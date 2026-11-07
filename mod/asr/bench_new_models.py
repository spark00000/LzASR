"""Local ASR comparison runner for installed open-weight Transformers models.

Preserves input media. Writes exactly one model-named .srt alongside input,
without overwriting existing SRT. This program never runs translation or
downloads model weights. Requires the configured runtime benchmark environment on Windows.

These full-clip SRTs provide text recognition evidence, not word-level timing
accuracy. All output timestamps span the original WAV.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import os
import sys
import time

HOME = Path(os.environ.get("LZASR_APP_HOME", str(Path(Path(__file__).resolve().anchor) / "APP" / "LzASR"))) / "models"
MODELS = {
    "granite-speech-5.0-470m-turboctc": "ctc",
    "granite-speech-5.0-2b": "granite",
    "parakeet-tdt-0.6b-v3": "tdt",
}
LANGUAGES = {
    "en_us": "English", "de_de": "German", "es_419": "Spanish",
    "fr_fr": "French", "ja_jp": "Japanese", "pt_br": "Portuguese",
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stamp(seconds: float) -> str:
    t = int(round(seconds * 1000))
    h, t = divmod(t, 3600000)
    m, t = divmod(t, 60000)
    s, ms = divmod(t, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def decode(model_name: str, wav, rate: int, language: str) -> str:
    import torch
    from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoModelForTDT, AutoProcessor

    folder = HOME / model_name
    if not (folder / "model.safetensors").is_file():
        raise FileNotFoundError(f"Model weights missing: {folder}")

    print("[LOADING_MODEL]", model_name, flush=True)
    proc = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
    family = MODELS[model_name]
    if family == "ctc":
        model = AutoModelForCTC.from_pretrained(str(folder), local_files_only=True).to("cuda").eval()
        tensors = proc(wav, sampling_rate=rate, device="cuda").to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(**tensors)
        result = proc.batch_decode(ids, skip_special_tokens=True)[0]
    elif family == "tdt":
        model = AutoModelForTDT.from_pretrained(str(folder), local_files_only=True).to("cuda").eval()
        tensors = proc(wav, sampling_rate=rate).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(**tensors, return_dict_in_generate=True)
        result = proc.decode(ids.sequences, skip_special_tokens=True)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, list):
            result = result[0]
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(folder), local_files_only=True, dtype=torch.bfloat16
        ).to("cuda").eval()
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language for Granite 2B: {language}")
        prompt = f"<|audio|>The speech is in {LANGUAGES[language]}. Transcribe it exactly; do not translate. Can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": prompt}]
        prepared = proc.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        samples = torch.as_tensor(wav).unsqueeze(0)
        tensors = proc(prepared, samples, device="cuda", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = model.generate(**tensors, max_new_tokens=1024, do_sample=False, num_beams=1)
        prefix = tensors["input_ids"].shape[-1]
        generated = ids[:, prefix:]
        result = proc.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(f"Empty ASR result: {model_name}")
    return " ".join(result.split())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run an already-downloaded open-weight ASR model; output SRT only")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--input", type=Path, required=True, help="16kHz mono WAV")
    parser.add_argument("--language", required=True, help="FLEURS locale, e.g. es_419")
    args = parser.parse_args()

    media = args.input.resolve()
    if not media.is_file():
        parser.error(f"Input not found: {media}")
    if args.model == "parakeet-tdt-0.6b-v3" and args.language in ("ja_jp", "ko_kr"):
        parser.error("Parakeet v3 does not support Japanese/Korean")
    if args.model.startswith("granite") and args.language == "ko_kr":
        parser.error("Granite Speech 5.0 does not support Korean")

    output = media.with_name(f"{media.stem}_{args.model}.srt")
    if output.exists():
        parser.error(f"Output exists; refusing to overwrite: {output}")
    import soundfile as sf
    samples, rate = sf.read(media, dtype="float32", always_2d=False)
    if rate != 16000 or samples.ndim != 1 or len(samples) < 8000:
        parser.error("Input must be 16kHz mono WAV of nonzero speech duration")

    before = sha256(media)
    print(f"[START] {args.model} language={args.language} source_sha256={before} seconds={len(samples)/rate:.3f}", flush=True)
    start = time.perf_counter()
    text = decode(args.model, samples, rate, args.language)
    elapsed = time.perf_counter() - start
    if sha256(media) != before:
        raise RuntimeError("Source audio changed during inference")
    payload = f"1\n00:00:00,000 --> {stamp(len(samples)/rate)}\n{text}\n"
    with output.open("x", encoding="utf-8", newline="\n") as target:
        target.write(payload)
    print(f"[PASS] {output} elapsed_s={elapsed:.3f} sha256={sha256(output)}", flush=True)
    print(f"[TRANSCRIPT] {text}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
