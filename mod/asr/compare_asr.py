"""Read-only ASR text comparison from official FLEURS 7-utterance SRT references.

This does not perform inference, translation, model download, or file writes.
Subsets are NOT the published full FLEURS test scores. For each locale:
- Latin-space-separated languages: word edit rate (WER)
- Korean and selected no-whitespace scripts: character edit rate (CER)
Standardize NFKC/casefold; replace Unicode punctuation and symbols with
spaces; collapse spaces; for CER ignore all spaces.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import unicodedata
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2] / "test"
NAMES = {
    "whisper-large-v3-turbo": "Whisper turbo",
    "whisper-large-v3-turbo-native": "Whisper turbo HF native",
    "whisper-large-v3": "Whisper large-v3 (chunk)",
    "whisper-large-v3-native": "Whisper large-v3 (native)",
    "BuzzASR-korean": "BuzzASR Korean (existing)",
    "BuzzASR-korean-beam5": "BuzzASR Korean beam5 (chunk)",
    "BuzzASR-korean-beam5-native": "BuzzASR Korean beam5 (native)",
    "nemotron-3.5-asr-streaming-0.6b": "Nemotron 3.5 streaming",
    "qwen3-asr-0.6b": "Qwen 0.6B",
    "qwen3-asr-1.7b": "Qwen 1.7B (legacy)",
    "qwen3-asr-1.7b-hf-native": "Qwen 1.7B (HF native)",
    "granite-speech-5.0-470m-turboctc": "Granite 470M",
    "granite-speech-5.0-2b": "Granite 2B",
    "parakeet-tdt-0.6b-v3": "Parakeet v3",
    "Fun-ASR-MLT-Nano-2512": "Fun-ASR 800M",
}
CER = {"ja_jp", "ko_kr", "cmn_hans_cn", "yue_hant_hk", "th_th"}


def cue_texts(path: Path) -> list[str]:
    blocks = re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip())
    text = []
    for index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(index) or "-->" not in lines[1]:
            raise ValueError(f"Invalid SRT cue {index} in {path}")
        text.append(" ".join(lines[2:]))
    return text


def text_of_srt(path: Path) -> str:
    return " ".join(cue_texts(path))


def units(text: str, metric: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    text = " ".join(text.split())
    if metric == "CER":
        return list(text.replace(" ", ""))
    return text.split()


def edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    last = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(row[-1] + 1, last[j] + 1, last[j - 1] + (a != b)))
        last = row
    return last[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="WER/CER from locally stored FLEURS reference and ASR SRTs")
    ap.add_argument("--locale", action="append", help="Locale directory; repeatable (default: en_us, ko_kr)")
    ap.add_argument("--all-locales", action="store_true", help="Inspect every available locale directory")
    args = ap.parse_args()
    if args.locale and args.all_locales:
        ap.error("--locale and --all-locales are mutually exclusive")
    priority = ("en_us", "ko_kr", "de_de")
    locales = (
        sorted(
            (p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "tv1"),
            key=lambda locale: (priority.index(locale) if locale in priority else len(priority), locale),
        )
        if args.all_locales else (args.locale or list(priority))
    )
    print("| 언어 | 공식 평가 표본 | 지표 | " + " | ".join(NAMES.values()) + " |", flush=True)
    print("|---|---|---" + "|---:" * len(NAMES) + "|", flush=True)
    for locale in locales:
        folder = ROOT / locale
        if not folder.is_dir():
            print(f"[SKIP] {locale}: no reference folder", flush=True)
            continue
        references = sorted(folder.glob("*_reference.srt"))
        if not references:
            print(f"[SKIP] {locale}: no reference SRT", flush=True)
            continue
        metric = "CER" if locale in CER else "WER"
        for ref in references:
            prefix = ref.name.removesuffix("_reference.srt")
            per_utterance = prefix.endswith("_test_40")
            ref_texts = cue_texts(ref)
            baseline = units(" ".join(ref_texts), metric)
            if not baseline:
                raise ValueError(f"Empty reference: {ref}")
            cells = []
            for suffix in NAMES:
                srt = folder / (prefix + "_" + suffix + ".srt")
                if not srt.is_file():
                    cells.append("—")
                    continue
                if per_utterance:
                    predicted_cues = cue_texts(srt)
                    if len(predicted_cues) != len(ref_texts):
                        raise ValueError(f"Mismatched utterance count: {srt} ({len(predicted_cues)}/{len(ref_texts)})")
                    errors = sum(
                        edit_distance(units(reference, metric), units(prediction, metric))
                        for reference, prediction in zip(ref_texts, predicted_cues)
                    )
                    total = sum(len(units(reference, metric)) for reference in ref_texts)
                else:
                    predicted = units(text_of_srt(srt), metric)
                    errors = edit_distance(baseline, predicted)
                    total = len(baseline)
                cells.append(f"{100*errors/total:.2f}%")
            dataset = ("test, 40 separate utterances" if per_utterance else "validation, 7 concatenated utterances")
            print("| " + locale + " | " + dataset + " | " + metric + " | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
