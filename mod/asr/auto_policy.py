"""Local ASR language-to-model routing.

The preferred routes reflect *measured local FLEURS subsets*, not universal
model rankings. Every language supported by the installed Whisper language
identifier has a runnable Whisper fallback. No network calls or uploads.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
APP_ROOT = Path(os.environ.get(
    "LZASR_APP_HOME", str(Path(HERE.anchor) / "APP" / "LzASR")
))
MODEL_ROOT = APP_ROOT / "models"

@dataclass(frozen=True)
class Route:
    engine: str
    model_name: str
    sample_error: float | None
    metric: str | None
    sample: str
    validated: bool

# Tested routes only. WER and CER are different measurements.
# Validation subsets contain just 7 utterances, English test has 40.
MEASURED: dict[str, tuple[str, float, str, str]] = {
    "en": ("qwen17hf", 4.01, "WER", "test40"),
    "ko": ("buzzkorean", 10.25, "CER", "val7"),
    "de": ("granite470", 5.15, "WER", "val7"),
    "cs": ("parakeet", 14.66, "WER", "val7"),
    "da": ("parakeet", 38.26, "WER", "val7"),
    "el": ("parakeet", 31.40, "WER", "val7"),
    "es": ("granite470", 3.33, "WER", "val7"),
    "fr": ("granite2b", 3.61, "WER", "val7"),
    "ja": ("whisper", 2.39, "CER", "val7"),
    "nl": ("parakeet", 13.51, "WER", "val7"),
    "pl": ("parakeet", 17.24, "WER", "val7"),
    "pt": ("whisper", 8.39, "WER", "val7"),
    "ro": ("parakeet", 33.70, "WER", "val7"),
    "ru": ("parakeet", 18.64, "WER", "val7"),
    "sv": ("parakeet", 20.27, "WER", "val7"),
}

OUTPUT_SUFFIX = {
    "qwen17hf": "_qwen3-asr-1.7b-hf.srt",
    "buzzkorean": "_buzzasr-korean-beam5.srt",
    "granite470": "_granite-speech-5.0-470m-turboctc.srt",
    "granite2b": "_granite-speech-5.0-2b.srt",
    "parakeet": "_parakeet-tdt-0.6b-v3.srt",
    "whisper": "_whisper-large-v3-turbo.srt",
}

ENGINE_MODELS = {
    "qwen17hf": "Qwen3-ASR-1.7B-hf",
    "buzzkorean": "BuzzASR-korean",
    "granite470": "granite-speech-5.0-470m-turboctc",
    "granite2b": "granite-speech-5.0-2b",
    "parakeet": "parakeet-tdt-0.6b-v3",
    "whisper": "faster-whisper-large-v3-turbo",
}

LANGUAGE_NAMES = {
    "en": "English", "ko": "Korean", "de": "German", "cs": "Czech",
    "da": "Danish", "el": "Greek", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sv": "Swedish", "ar": "Arabic",
    "zh": "Chinese (Simplified)", "yue": "Cantonese", "fa": "Persian",
    "tl": "Filipino", "fi": "Finnish", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "mk": "Macedonian",
    "ms": "Malay", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}

# Official FLEURS locale IDs; other Whisper language codes use their ISO form.
LOCALE_ALIASES = {
    "ar_eg": "ar", "cmn_hans_cn": "zh", "cs_cz": "cs",
    "da_dk": "da", "de_de": "de", "el_gr": "el", "en_us": "en",
    "es_419": "es", "fa_ir": "fa", "fil_ph": "tl", "fi_fi": "fi",
    "fr_fr": "fr", "hi_in": "hi", "hu_hu": "hu", "id_id": "id",
    "it_it": "it", "ja_jp": "ja", "ko_kr": "ko", "mk_mk": "mk",
    "ms_my": "ms", "nl_nl": "nl", "pl_pl": "pl", "pt_br": "pt",
    "ro_ro": "ro", "ru_ru": "ru", "sv_se": "sv", "th_th": "th",
    "tr_tr": "tr", "vi_vn": "vi", "yue_hant_hk": "yue",
}


def supported_languages() -> frozenset[str]:
    """Read all supported languages from installed official Whisper config."""
    path = MODEL_ROOT / "whisper-large-v3" / "generation_config.json"
    if not path.is_file():
        raise FileNotFoundError("Whisper language detector weights/config missing")
    config = json.loads(path.read_text(encoding="utf-8"))
    lang_to_id = config.get("lang_to_id")
    if not isinstance(lang_to_id, dict) or len(lang_to_id) < 90:
        raise ValueError("Unexpected Whisper language detector configuration")
    return frozenset(
        token[2:-2] for token in lang_to_id
        if re.fullmatch(r"<\|[a-z]{2,4}\|>", token)
    )


def normalize_language(value: str, allowed: frozenset[str] | None = None) -> str:
    """Normalize ISO tags, FLEURS locales, names or Whisper tokens."""
    languages = allowed if allowed is not None else supported_languages()
    key = value.strip().lower().replace("-", "_")
    for name, title in LANGUAGE_NAMES.items():
        if key == title.lower():
            key = name
            break
    if re.fullmatch(r"<\|[a-z]{2,4}\|>", key):
        key = key[2:-2]
    key = LOCALE_ALIASES.get(key, key)
    if key not in languages and "_" in key:
        key = key.split("_", 1)[0]
    if key not in languages:
        raise ValueError(f"Unsupported language: {value!r}")
    return key


def model_ready(engine: str) -> bool:
    if engine not in ENGINE_MODELS:
        return False
    if engine == "whisper":
        return bool(
            (MODEL_ROOT / ENGINE_MODELS[engine] / "model.bin").is_file()
            and (APP_ROOT / "engine" / "Faster-Whisper-XXL"
                 / "faster-whisper-xxl.exe").is_file()
        )
    folder = MODEL_ROOT / ENGINE_MODELS[engine]
    return (folder / "model.safetensors").is_file()


def choose_route(language: str, *, allow_fallback: bool = True) -> Route:
    code = normalize_language(language)
    recorded = MEASURED.get(code)
    target = recorded[0] if recorded else "whisper"
    selected = target
    if not model_ready(selected):
        if not allow_fallback or not model_ready("whisper"):
            raise FileNotFoundError(f"Local ASR model unavailable: {selected}")
        selected = "whisper"
    is_evidenced = recorded is not None and selected == target
    return Route(
        engine=selected,
        model_name=ENGINE_MODELS[selected],
        sample_error=recorded[1] if is_evidenced else None,
        metric=recorded[2] if is_evidenced else None,
        sample=recorded[3] if is_evidenced else "unmeasured-fallback",
        validated=is_evidenced,
    )
