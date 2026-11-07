"""Reproducible local Fun-ASR-MLT-Nano ASR runner (no translation).

Model: FunAudioLLM/Fun-ASR-MLT-Nano-2512 (SHA256 pinned in comparison doc)
Local dependencies: isolated runtime Python environment, with read-only
torch/transformers packages supplied from existing GPU environments.
Writes precisely one SRT per input, refuses overwrites. Current checkpoint
does not load its optional CTC timestamp head; SRT timing is whole-clip only.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import os
import sys
import time

APP_ROOT = Path(os.environ.get("LZASR_APP_HOME", str(Path(Path(__file__).resolve().anchor) / "APP" / "LzASR")))
MODEL = APP_ROOT / "models" / "Fun-ASR-MLT-Nano-2512"
BASE_PACKAGES = (
    str(APP_ROOT / ".venv-bench" / "Lib" / "site-packages"),
    str(APP_ROOT / ".venv-qwen" / "Lib" / "site-packages"),
)
LANGUAGES = {
    "ko_kr": "韩文",
    "en_us": "英文",
    "ja_jp": "日文",
    "cmn_hans_cn": "中文",
    "yue_hant_hk": "粤语",
}


def filehash(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp(s: float) -> str:
    n = round(s * 1000)
    h, n = divmod(n, 3600000)
    m, n = divmod(n, 60000)
    sec, ms = divmod(n, 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run official Fun-ASR multilingual weights on GPU (SRT only)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--language", choices=sorted(LANGUAGES), required=True)
    args = parser.parse_args()
    media = args.input.resolve()
    output = media.with_name(media.stem + "_Fun-ASR-MLT-Nano-2512.srt")
    if not media.is_file() or media.suffix.lower() != ".wav":
        parser.error(f"Input must be an existing WAV: {media}")
    if output.exists():
        parser.error(f"Output already exists; refusing overwrite: {output}")
    if not (MODEL / "model.pt").is_file():
        parser.error(f"Checkpoint missing: {MODEL}")
    import soundfile as sf
    info = sf.info(str(media))
    if info.samplerate != 16000 or info.channels != 1:
        parser.error("Requires 16kHz mono WAV")

    # The isolated environment's dependencies were installed separately.
    # Reuse the other venvs in read-only mode without changing their contents.
    sys.path.extend(BASE_PACKAGES)
    from funasr.models.sense_voice.model import SenseVoiceEncoderSmall  # noqa: F401 -- class registration
    from funasr import AutoModel
    before = filehash(media)
    started = time.monotonic()
    print(f"[START] model=Fun-ASR-MLT-Nano language={args.language} input_sha256={before}", flush=True)
    model = AutoModel(
        model=str(MODEL), hub="hf", trust_remote_code=False,
        device="cuda:0", disable_update=True,
    )
    output_rows = model.generate(
        input=[str(media)], cache={}, batch_size=1,
        language=LANGUAGES[args.language], itn=False,
        llm_kwargs={"do_sample": False},
    )
    if not output_rows or not isinstance(output_rows[0], dict):
        raise RuntimeError(f"No ASR output: {output_rows!r}")
    transcription = str(output_rows[0].get("text", "")).strip()
    if not transcription:
        raise RuntimeError("Empty transcript")
    if filehash(media) != before:
        raise RuntimeError("Input changed during recognition")
    srt = f"1\n00:00:00,000 --> {stamp(info.duration)}\n{transcription}\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(srt)
    print(f"[PASS] {output} seconds={time.monotonic()-started:.2f} sha256={filehash(output)}", flush=True)
    print(f"[TEXT] {transcription}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
