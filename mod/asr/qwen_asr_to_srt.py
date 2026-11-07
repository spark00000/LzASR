"""Qwen3-ASR with forced alignment -> SRT, fully local after model download.

Models:
    Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B
    Qwen/Qwen3-ForcedAligner-0.6B
Installed model files, temporary audio, and runtime cache live under the project-level bin directory.
Only the finished .srt is saved beside the original media file.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

APP = Path(__file__).resolve().parent
BIN = Path(os.environ.get("LZASR_APP_HOME", str(Path(Path(__file__).resolve().anchor) / "APP" / "LzASR")))
MODELS = BIN / "models"
TEMP_ROOT = BIN / "temp-qwen"
REPOS = {
    "1.7B": ("Qwen/Qwen3-ASR-1.7B", MODELS / "Qwen3-ASR-1.7B"),
    "0.6B": ("Qwen/Qwen3-ASR-0.6B", MODELS / "Qwen3-ASR-0.6B"),
}
ALIGNER = ("Qwen/Qwen3-ForcedAligner-0.6B", MODELS / "Qwen3-ForcedAligner-0.6B")
CHUNK_SECONDS = 120
MAX_CUE_SECONDS = 5.5
MAX_CUE_CHARS = 77
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def model_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    index = path / "model.safetensors.index.json"
    if index.is_file():
        import json
        try:
            items = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (KeyError, ValueError, OSError):
            return False
        return bool(items) and all(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in items
        )
    full = path / "model.safetensors"
    return full.is_file() and full.stat().st_size > 0


def load_or_download(repo: str, folder: Path) -> str:
    if model_ready(folder):
        print(f"[MODEL] Local: {folder}", flush=True)
        return str(folder)
    print(f"[DOWNLOAD] {repo} -> {folder} (one-time; multi-GB)", flush=True)
    # Only new directories are created; existing files are not deleted.
    from huggingface_hub import snapshot_download

    import time
    for attempt in range(1, 7):
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(folder),
                max_workers=2,
                etag_timeout=30,
            )
            break
        except Exception as exc:
            if attempt >= 6:
                raise
            delay = min(2 ** attempt, 30)
            print(f"[RETRY] {repo} attempt={attempt} error={type(exc).__name__}, wait={delay}s", flush=True)
            time.sleep(delay)
    if not model_ready(folder):
        raise RuntimeError(f"Model files are incomplete in {folder}")
    return str(folder)


def probe_audio(media: Path, destination: Path) -> list[Path]:
    executable = os.environ.get("LZASR_FFMPEG") or shutil.which("ffmpeg")
    if not executable and (BIN / "engine" / "ffmpeg" / "ffmpeg.exe").is_file():
        executable = str(BIN / "engine" / "ffmpeg" / "ffmpeg.exe")
    if not executable:
        raise RuntimeError("ffmpeg.exe is required but missing from PATH")
    # ffmpeg segments a long video in one pass to avoid excessive RAM use.
    output_pattern = str(destination / "chunk_%05d.wav")
    cmd = [
        executable, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(media), "-map", "0:a:0", "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
        "-segment_format", "wav", "-reset_timestamps", "1",
        "-y", output_pattern,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed: {completed.stderr.strip()}")
    parts = sorted(destination.glob("chunk_*.wav"))
    if not parts:
        raise RuntimeError("ffmpeg did not create any WAV audio chunks")
    return parts


def duration_of(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def _join(previous: str, word: str, language: str | None) -> str:
    word = word.strip()
    if not word:
        return previous
    if not previous:
        return word
    if word[0] in ".,!?;:%)]}、。，！？：；」』" or previous[-1] in "([{'\"“「『":
        return previous + word
    if language in {"Chinese", "Japanese", "Cantonese"} and CJK_RE.search(word):
        return previous + word
    if language != "Korean" and CJK_RE.search(previous[-1]) and CJK_RE.search(word[0]):
        return previous + word
    return previous + " " + word


def original_spacing(stamps: list, transcript: str) -> list | None:
    """Project the recognizer's exact spacing/punctuation onto timed aligner tokens.

    Important for Korean: ForcedAligner splits morphemes (e.g., '인식' + '으로'),
    while ASR text contains the intended word boundaries ('인식으로').
    """
    if not transcript.strip() or not stamps:
        return None
    decorated: list = []
    cursor = 0
    for stamp in stamps:
        surface = str(getattr(stamp, "text", "")).strip()
        if not surface:
            continue
        position = transcript.find(surface, cursor)
        if position < 0:
            match = re.search(re.escape(surface), transcript[cursor:], flags=re.IGNORECASE)
            position = cursor + match.start() if match else -1
        if position < 0:
            return None
        padding = transcript[cursor:position]
        if decorated:
            decorated[-1].text += padding
        raw = transcript[position:position + len(surface)]
        decorated.append(SimpleNamespace(
            text=(padding if not decorated else "") + raw,
            start_time=stamp.start_time,
            end_time=stamp.end_time,
        ))
        cursor = position + len(surface)
    if not decorated:
        return None
    decorated[-1].text += transcript[cursor:]
    return decorated


def make_cues(stamps: list, offset: float, language: str | None,
              transcript: str | None = None) -> list[Cue]:
    decorated = original_spacing(stamps, transcript) if transcript else None
    exact_text = decorated is not None
    if exact_text:
        stamps = decorated
    cues: list[Cue] = []
    group: list = []
    for item in stamps:
        text = str(getattr(item, "text", ""))
        if not text.strip():
            continue
        start = float(getattr(item, "start_time"))
        end = float(getattr(item, "end_time"))
        if end <= start:
            continue
        if group:
            projected = end - group[0][1]
            approx = sum(len(t[0]) for t in group) + len(text) + len(group)
            gap = start - group[-1][2]
            if projected > MAX_CUE_SECONDS or approx > MAX_CUE_CHARS or gap > 0.85:
                cues.append(build_cue(group, offset, language, exact_text))
                group = []
        group.append((text, start, end))
        if text.rstrip().endswith((".", "?", "!", "。", "！", "？")) and end - group[0][1] >= 0.6:
            cues.append(build_cue(group, offset, language, exact_text))
            group = []
    if group:
        cues.append(build_cue(group, offset, language, exact_text))
    return cues


def build_cue(group: list, offset: float, language: str | None,
              exact_text: bool = False) -> Cue:
    if exact_text:
        text = "".join(word for word, _, _ in group).strip()
    else:
        text = ""
        for word, _, _ in group:
            text = _join(text, word, language)
    return Cue(start=max(0.0, group[0][1] + offset),
               end=max(group[0][1] + 0.01, group[-1][2]) + offset,
               text=text.strip())


def srt_timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hh, ms = divmod(ms, 3_600_000)
    mm, ms = divmod(ms, 60_000)
    ss, ms = divmod(ms, 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"


def render_srt(cues: list[Cue]) -> str:
    result = []
    for index, cue in enumerate(cues, 1):
        if not cue.text:
            continue
        result.extend([
            str(index),
            f"{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}",
            cue.text,
            "",
        ])
    return "\n".join(result).rstrip() + "\n"


def check_selftest() -> None:
    rows = [
        SimpleNamespace(text="Hello", start_time=0.05, end_time=0.45),
        SimpleNamespace(text="world.", start_time=0.48, end_time=0.91),
        SimpleNamespace(text="Again!", start_time=1.2, end_time=1.6),
    ]
    output = render_srt(make_cues(rows, 12.0, "English"))
    assert "00:00:12,050 --> 00:00:12,910" in output, output
    assert "Hello world." in output, output
    assert "00:00:13,200 --> 00:00:13,600" in output, output
    assert len(re.findall(r"--> ", output)) == 2, output
    print("[OK] Qwen SRT formatter self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen3 ASR -> SRT")
    parser.add_argument("input", nargs="?", type=Path, help="Video or audio file")
    parser.add_argument("--model", choices=REPOS.keys(), default="1.7B")
    parser.add_argument("--output", type=Path, help="SRT path (same dir as source)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        check_selftest()
        return 0
    if args.input is None:
        parser.error("An input media file is required")
    media = args.input.expanduser().resolve(strict=True)
    if not media.is_file():
        raise RuntimeError(f"Input is not a regular file: {media}")
    suffix = "_qwen3-asr-1.7b.srt" if args.model == "1.7B" else "_qwen3-asr-0.6b.srt"
    output = args.output or media.with_name(media.stem + suffix)
    output = output.resolve(strict=False)
    if output.parent != media.parent or output.suffix.lower() != ".srt":
        raise RuntimeError("SRT output must be in the input media directory")
    if output.exists():
        print(f"[SKIP] Existing SRT: {output}")
        return 0

    os.environ.setdefault("HF_HOME", str(MODELS / ".hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS / ".hf" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(MODELS / ".hf" / "xet"))
    os.environ.setdefault("TORCH_HOME", str(MODELS / ".torch"))
    os.environ.setdefault("TMP", str(TEMP_ROOT))
    os.environ.setdefault("TEMP", str(TEMP_ROOT))
    TEMP_ROOT.mkdir(exist_ok=True, parents=True)

    import torch
    from qwen_asr import Qwen3ASRModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; Qwen GPU inference cannot start")
    print(f"[GPU] {torch.cuda.get_device_name(0)} / BF16", flush=True)
    repo, folder = REPOS[args.model]
    asr_dir = load_or_download(repo, folder)
    align_dir = load_or_download(*ALIGNER)
    print("[LOAD] Qwen3 ASR + ForcedAligner", flush=True)

    model = Qwen3ASRModel.from_pretrained(
        asr_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=512,
        forced_aligner=align_dir,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    )

    cues: list[Cue] = []
    with tempfile.TemporaryDirectory(dir=TEMP_ROOT, prefix="segments_") as temp:
        chunks = probe_audio(media, Path(temp))
        offset = 0.0
        for number, part in enumerate(chunks, 1):
            dur = duration_of(part)
            print(f"[TRANSCRIBE] {number}/{len(chunks)} t={offset:.1f}s {dur:.1f}s", flush=True)
            result = model.transcribe(audio=str(part), language=None, return_time_stamps=True)
            if not result or not result[0]:
                raise RuntimeError(f"ASR did not return a result for chunk {number}")
            item = result[0]
            print(f"[LANGUAGE] {item.language}", flush=True)
            if not item.text.strip():
                print(f"[SILENCE] chunk={number} has no transcript; no alignment required", flush=True)
                offset += dur
                continue
            stamps = item.time_stamps
            if stamps is None or not stamps:
                # A single bounded resegmentation is permitted for failed alignment.
                # Do not suppress any nonempty transcript with missing timestamps.
                if dur < 15:
                    raise RuntimeError(f"Unaligned speech chunk {number}; duration={dur:.2f}s")
                print(f"[ALIGN_RESEGMENT] chunk={number} duration={dur:.2f}s, two subchunks", flush=True)
                split_at = dur / 2
                rescued = []
                for index, (begin, length) in enumerate(((0.0, split_at), (split_at, dur - split_at)), 1):
                    sub = Path(temp) / f"rescue_{number:05d}_{index}.wav"
                    ffmpeg = os.environ.get("LZASR_FFMPEG") or shutil.which("ffmpeg") or str(BIN / "engine" / "ffmpeg" / "ffmpeg.exe")
                    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                               "-ss", str(begin), "-t", str(length), "-i", str(part),
                               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                               "-y", str(sub)]
                    conversion = subprocess.run(command, capture_output=True, text=True)
                    if conversion.returncode != 0 or not sub.is_file():
                        raise RuntimeError(f"Cannot split unaligned chunk {number}: {conversion.stderr[-300:]}")
                    children = model.transcribe(audio=str(sub), language=None, return_time_stamps=True)
                    if not children or not children[0]:
                        raise RuntimeError(f"No ASR result from split chunk {number}.{index}")
                    segment = children[0]
                    if not segment.text.strip():
                        print(f"[SILENCE] split={number}.{index}", flush=True)
                        continue
                    if segment.time_stamps is None or not segment.time_stamps:
                        raise RuntimeError(f"Unaligned speech after single split: chunk {number}.{index}")
                    rescued.extend(make_cues(segment.time_stamps, offset + begin,
                                            segment.language, transcript=segment.text))
                cues.extend(rescued)
            else:
                cues.extend(make_cues(stamps, offset, item.language, transcript=item.text))
            offset += dur

    if not cues:
        raise RuntimeError("No subtitle cues generated; not writing an empty SRT")
    payload = render_srt(cues)
    # Exclusive creation guarantees existing subtitles are never overwritten.
    try:
        with output.open("x", encoding="utf-8-sig", newline="\r\n") as stream:
            stream.write(payload)
    except FileExistsError:
        print(f"[SKIP] Existing SRT appeared while processing: {output}")
        return 0
    print(f"[OK] {output} ({len(cues)} subtitles)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
