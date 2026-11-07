"""Automatic speech language ID, model routing and SRT-only local ASR.

- Decode 16 kHz mono PCM through ffmpeg without writing intermediate audio.
- Scan 12-second windows for RMS loudness plus voice-band spectral energy.
- Whisper language ID votes over distinct high-activity windows.
- Choose one local ASR model for the entire media file (no code switching).
- Save only a model-named SRT beside input; never replace existing files.

The FLEURS-based language preferences are preliminary. Broad support is given by
the 100 language IDs in installed Whisper large-v3; unmeasured codes fall back
to Whisper turbo rather than claiming an untested model is best.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import auto_policy as policy

RATE = 16_000
PROBE_SECONDS = 12
INFER_SECONDS = 12
VOICE_MIN_RMS = 0.002
TRANSCRIBE_MIN_RMS = 0.00020  # preserve quieter speech in full-video ASR
SAMPLE_COUNT = 5
MIN_SPACING = 45.0
DRIVER = Path(__file__).resolve().with_name("ASR.bat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"


def ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    bundled = policy.APP_ROOT / "engine" / "ffmpeg" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    raise FileNotFoundError("ffmpeg is not installed; no audio was extracted")


def iter_audio(media: Path, block_seconds: int):
    """Yield duration, absolute start and PCM chunks; no temporary files."""
    import numpy as np

    command = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-v", "error",
        "-i", str(media), "-map", "0:a:0", "-vn", "-ac", "1",
        "-ar", str(RATE), "-f", "f32le", "pipe:1",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, bufsize=1024 * 1024,
    )
    assert process.stdout is not None and process.stderr is not None
    count = 0
    byte_count = RATE * block_seconds * 4
    try:
        while True:
            raw = process.stdout.read(byte_count)
            if not raw:
                break
            if len(raw) % 4:
                raise RuntimeError("Malformed float32 audio output")
            audio = np.frombuffer(raw, dtype="<f4")
            if audio.size == 0:
                break
            yield count / RATE, audio
            count += audio.size
        status = process.wait(timeout=60)
        error = process.stderr.read().decode("utf-8", "replace")
        if status != 0:
            raise RuntimeError(f"Audio decode failed (exit {status}): {error[-300:]}")
        if count < RATE:
            raise RuntimeError("No usable speech audio stream")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=20)
        process.stdout.close()
        process.stderr.close()


def speech_metrics(samples) -> tuple[float, float, float]:
    """RMS volume, speech-band energy fraction, amplitude modulation proxy."""
    import numpy as np

    if samples.size < RATE:
        return 0.0, 0.0, 0.0
    wave = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(wave * wave)))
    if not np.isfinite(rms) or rms <= 0:
        return 0.0, 0.0, 0.0
    # Compute one FFT spectrum in the middle of each candidate block.
    mid = wave.size // 2
    frame = wave[max(0, mid - RATE // 2):mid + RATE // 2]
    if frame.size < RATE:
        frame = wave[:RATE]
    fft = abs(np.fft.rfft(frame[:RATE] * np.hanning(RATE))) ** 2
    frequencies = np.fft.rfftfreq(RATE, 1.0 / RATE)
    useful = float(np.sum(fft[(frequencies >= 120) & (frequencies <= 3800)]))
    total = float(np.sum(fft[(frequencies >= 50) & (frequencies <= 7600)]))
    band_fraction = useful / total if total > 0 else 0.0
    steps = wave[:(wave.size // (RATE // 5)) * (RATE // 5)]
    power = steps.reshape(-1, RATE // 5)
    levels = np.sqrt(np.mean(power * power, axis=1))
    modulation = float(np.std(levels) / max(1e-5, np.mean(levels)))
    return rms, band_fraction, modulation


def choose_samples(media: Path, count: int = SAMPLE_COUNT):
    """Keep only small high-volume speech-like windows in RAM."""
    import numpy as np

    if count < 1 or count > 10:
        raise ValueError("Probe sample count outside 1..10")
    candidates: list[tuple[float, float, "np.ndarray", float]] = []
    windows = 0
    duration = 0.0
    for start, audio in iter_audio(media, PROBE_SECONDS):
        windows += 1
        duration = start + len(audio) / RATE
        rms, band, modulation = speech_metrics(audio)
        if rms < VOICE_MIN_RMS or band < 0.30:
            continue
        score = rms * (0.50 + min(band, 1.0)) * (
            0.65 + min(modulation, 1.2)
        )
        candidates.append((score, start, audio.copy(), rms))
        # Bound retained RAM irrespective of recording duration.
        if len(candidates) > 80:
            candidates.sort(key=lambda c: c[0], reverse=True)
            candidates = candidates[:45]
    if not candidates:
        raise RuntimeError("No audible speech-like regions; cannot determine language")
    candidates.sort(key=lambda c: c[0], reverse=True)
    picks: list[tuple[float, float, "np.ndarray", float]] = []
    for candidate in candidates:
        if all(abs(candidate[1] - p[1]) >= MIN_SPACING for p in picks):
            picks.append(candidate)
            if len(picks) == count:
                break
    if not picks:
        raise RuntimeError("No distinct language detection samples")
    picks.sort(key=lambda c: c[1])
    print(
        "[AUDIO_PROFILE] duration_s=%.1f windows=%d speech_candidates=%d "
        "sample_count=%d" % (duration, windows, len(candidates), len(picks)),
        flush=True,
    )
    print(
        "[AUDIO_PEAKS] " + " ".join(
            f"{start:.1f}s:{rms:.4f}rms" for _, start, _, rms in picks
        ),
        flush=True,
    )
    return picks


def detect_language(samples) -> tuple[str, int, int]:
    """Whisper's language-ID head, with voting. Votes are not probabilities."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    folder = policy.MODEL_ROOT / "whisper-large-v3"
    if not (folder / "model.safetensors").is_file():
        raise FileNotFoundError("Whisper large-v3 language detector weights missing")
    processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(folder), local_files_only=True, dtype=torch.float16
    ).to("cuda").eval()
    ids: list[str] = []
    try:
        for _, start, sample, rms in samples:
            features = processor(
                np.asarray(sample, dtype=np.float32),
                sampling_rate=RATE, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.float16)
            with torch.inference_mode():
                prediction = model.detect_language(input_features=features)
            tokens = processor.tokenizer.convert_ids_to_tokens(
                prediction.flatten().tolist()
            )
            if len(tokens) != 1:
                raise RuntimeError("Expected one language token")
            language = policy.normalize_language(tokens[0])
            ids.append(language)
            print(
                f"[LID_SAMPLE] start_s={start:.1f} language={language}",
                flush=True,
            )
        votes = Counter(ids)
        winner, majority = votes.most_common(1)[0]
        if len(votes) > 1:
            # Video may contain multiple languages; the user wants one
            # language for the whole file. Ignore later language switches.
            # Vote first; resolve a tie by the summed peak RMS.
            loudness = {
                code: sum(
                    float(samples[i][3]) for i, item in enumerate(ids)
                    if item == code
                )
                for code in votes
            }
            winner = max(
                votes, key=lambda code: (
                    votes[code], loudness[code], -ids.index(code)
                )
            )
            majority = votes[winner]
            print(
                f"[LID_MIXED_WARNING] votes={dict(votes)} "
                f"selected={winner} rule=votes_then_peak_rms "
                "language_switching=ignored",
                flush=True,
            )
        if len(ids) < 3:
            print("[LID_WARNING] Too few distinct speech samples to confirm language", flush=True)
        print(
            f"[LID_RESULT] language={winner} votes={majority}/{len(ids)} "
            "criterion=majority_vote_not_calibrated_probability",
            flush=True,
        )
        return winner, majority, len(ids)
    finally:
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()


def has_speech(wave) -> tuple[bool, float, float]:
    """Conservative energy gate, plus start/end of audible subsection."""
    import numpy as np

    audio = np.asarray(wave, dtype=np.float32)
    if audio.size < RATE // 2:
        return False, 0.0, 0.0
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms < TRANSCRIBE_MIN_RMS:
        return False, 0.0, 0.0
    frame = RATE // 10
    count = audio.size // frame
    if not count:
        return False, 0.0, 0.0
    sections = audio[:count * frame].reshape(count, frame)
    energy = np.sqrt(np.mean(sections * sections, axis=1))
    threshold = max(TRANSCRIBE_MIN_RMS * 0.6, min(0.02, rms * 0.32))
    active = np.flatnonzero(energy > threshold)
    if active.size < 2:
        return False, 0.0, 0.0
    begin = max(0.0, (float(active[0]) * frame / RATE) - .10)
    end = min(len(audio) / RATE, (float(active[-1] + 1) * frame / RATE) + .10)
    return True, begin, end


def clean_text(value: str) -> str:
    text = " ".join(value.split())
    return re.sub(r"[\x00-\x1f]", " ", text).strip()


class SpeechEngine:
    """One GPU model instance handles all inference chunks of one file."""

    def __init__(self, engine: str, language: str):
        self.engine = engine
        self.language = language
        import torch
        from transformers import (
            AutoProcessor, AutoModelForMultimodalLM, AutoModelForCTC,
            AutoModelForSpeechSeq2Seq, AutoModelForTDT,
            WhisperForConditionalGeneration,
        )

        self.torch = torch
        folder = policy.MODEL_ROOT / policy.ENGINE_MODELS[engine]
        self.processor = AutoProcessor.from_pretrained(
            str(folder), local_files_only=True,
        )
        if engine == "qwen17hf":
            self.model = AutoModelForMultimodalLM.from_pretrained(
                str(folder), local_files_only=True, dtype=torch.bfloat16,
            ).to("cuda").eval()
        elif engine == "buzzkorean":
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(folder), local_files_only=True, dtype=torch.float16,
            ).to("cuda").eval()
        elif engine == "granite470":
            self.model = AutoModelForCTC.from_pretrained(
                str(folder), local_files_only=True,
            ).to("cuda").eval()
        elif engine == "parakeet":
            self.model = AutoModelForTDT.from_pretrained(
                str(folder), local_files_only=True,
            ).to("cuda").eval()
        elif engine == "granite2b":
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(folder), local_files_only=True, dtype=torch.bfloat16,
            ).to("cuda").eval()
            name = policy.LANGUAGE_NAMES.get(language, language)
            user = (
                f"<|audio|>The speech is in {name}. Transcribe it exactly; "
                "do not translate. Can you transcribe the speech into "
                "a written format?"
            )
            messages = [{"role": "user", "content": user}]
            self.prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            raise ValueError(f"Unsupported native engine: {engine}")

    def transcribe(self, wave) -> str:
        torch = self.torch
        with torch.inference_mode():
            if self.engine == "qwen17hf":
                values = self.processor.apply_transcription_request(
                    audio=[wave],
                    language=[policy.LANGUAGE_NAMES.get(self.language, "English")],
                ).to("cuda", torch.bfloat16)
                predicted = self.model.generate(
                    **values, max_new_tokens=256, do_sample=False
                )
                decoded = self.processor.decode(
                    predicted[:, values["input_ids"].shape[1]:],
                    return_format="transcription_only",
                )
                result = decoded[0]
            elif self.engine == "buzzkorean":
                features = self.processor(
                    wave, sampling_rate=RATE, return_tensors="pt",
                ).input_features.to("cuda", dtype=torch.float16)
                ids = self.model.generate(features, num_beams=5, max_new_tokens=256)
                result = self.processor.batch_decode(
                    ids, skip_special_tokens=True,
                )[0]
            elif self.engine == "granite470":
                features = self.processor(
                    wave, sampling_rate=RATE, device="cuda",
                ).to(self.model.device, dtype=self.model.dtype)
                ids = self.model.generate(**features)
                result = self.processor.batch_decode(
                    ids, skip_special_tokens=True,
                )[0]
            elif self.engine == "parakeet":
                features = self.processor(
                    wave, sampling_rate=RATE,
                ).to(self.model.device, dtype=self.model.dtype)
                ids = self.model.generate(
                    **features, return_dict_in_generate=True
                )
                result = self.processor.decode(
                    ids.sequences, skip_special_tokens=True,
                )
                if isinstance(result, (tuple, list)):
                    result = result[0]
            else:
                features = self.processor(
                    self.prompt, torch.as_tensor(wave).unsqueeze(0),
                    device="cuda", return_tensors="pt",
                ).to("cuda")
                ids = self.model.generate(
                    **features, max_new_tokens=256,
                    do_sample=False, num_beams=1,
                )
                result = self.processor.tokenizer.batch_decode(
                    ids[:, features["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                )[0]
        if not isinstance(result, str):
            raise RuntimeError(f"Unexpected {self.engine} transcript type")
        return clean_text(result)

    def close(self):
        del self.model
        del self.processor
        gc.collect()
        self.torch.cuda.empty_cache()


def write_subtitles(media: Path, output: Path, engine: str, language: str):
    import numpy as np

    original = sha256(media)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing subtitle: {output}")
    started = time.perf_counter()
    asr = SpeechEngine(engine, language)
    captions: list[tuple[float, float, str]] = []
    duration = 0.0
    windows = 0
    try:
        for start, audio in iter_audio(media, INFER_SECONDS):
            windows += 1
            duration = start + len(audio) / RATE
            speaking, begin, end = has_speech(audio)
            if not speaking:
                continue
            prediction = asr.transcribe(np.asarray(audio, dtype=np.float32))
            if not prediction or re.fullmatch(r"[\W_]+", prediction):
                continue
            # These are acoustic chunk boundaries, not word-level alignment.
            captions.append((start + begin, start + end, prediction))
            if windows % 40 == 0:
                print(
                    f"[ASR_PROGRESS] audio_min={duration / 60:.1f} "
                    f"cues={len(captions)}",
                    flush=True,
                )
    finally:
        asr.close()
    if not captions:
        raise RuntimeError("No speech was recognized, no SRT created")
    if sha256(media) != original:
        raise RuntimeError("Input media changed during ASR; no SRT saved")
    lines = [
        f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n"
        for i, (a, b, text) in enumerate(captions, 1)
    ]
    payload = "\n".join(lines) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(payload)
    print(
        f"[ASR_FINISHED] cues={len(captions)} duration_s={duration:.1f} "
        f"elapsed_s={time.perf_counter() - started:.1f} "
        "timing=acoustic_chunk_not_word_alignment",
        flush=True,
    )


def run_whisper_existing(media: Path, output: Path):
    if output.exists():
        print("[ASR_SKIPPED] Existing Whisper SRT preserved", flush=True)
        return
    if not DRIVER.is_file():
        raise FileNotFoundError("Whisper standalone launcher missing")
    proc = subprocess.run(
        ["cmd.exe", "/d", "/c", str(DRIVER),
         "--engine", "whisper", str(media)],
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Whisper engine exited with code {proc.returncode}")
    if not output.is_file():
        raise RuntimeError("Whisper exited successfully without its SRT")


def check_environment() -> list[str]:
    missing = []
    try:
        ffmpeg_executable()
    except FileNotFoundError as e:
        missing.append(str(e))
    try:
        supported = policy.supported_languages()
        if len(supported) < 90:
            missing.append("Whisper language list is incomplete")
    except (FileNotFoundError, ValueError) as e:
        missing.append(str(e))
    if not (policy.MODEL_ROOT / "whisper-large-v3" / "model.safetensors").is_file():
        missing.append("Whisper language detector weights")
    if not policy.model_ready("whisper"):
        missing.append("Whisper turbo fallback engine/weights")
    try:
        import numpy, torch, transformers
        if not torch.cuda.is_available():
            missing.append("CUDA GPU unavailable")
    except ImportError as e:
        missing.append(f"ASR benchmark environment: {e}")
    return missing


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Automatic language ID + language-routed local ASR (SRT only)"
    )
    parser.add_argument("media", nargs="?", type=Path)
    parser.add_argument(
        "--source-language", "--language", default="auto",
        help="Auto detect or supply ISO / BCP-47 language (en, ja, ko, de, ...)",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-routing", action="store_true",
                        help="All installed Whisper LID language codes and local routes")
    args = parser.parse_args(argv)

    if args.check:
        missing = check_environment()
        for item in missing:
            print("[MISSING]", item, flush=True)
        if not missing:
            print(
                f"[PASS] Auto ASR ready; {len(policy.supported_languages())} "
                "Whisper language IDs available",
                flush=True,
            )
        return 0 if not missing else 2
    if args.list_routing:
        for lang in sorted(policy.supported_languages()):
            route = policy.choose_route(lang)
            print(
                f"{lang}\t{route.engine}\t{route.sample}\t"
                f"{route.sample_error if route.validated else '-'}",
                flush=True,
            )
        return 0
    if args.media is None:
        parser.print_help()
        return 2
    media = args.media.expanduser().resolve()
    if not media.is_file():
        print("[ERROR] Input media missing", file=sys.stderr)
        return 2

    missing = check_environment()
    if missing:
        for item in missing:
            print("[MISSING]", item, file=sys.stderr)
        return 2
    try:
        if args.source_language.lower() == "auto":
            candidate_samples = choose_samples(media)
            language, votes, count = detect_language(candidate_samples)
        else:
            language = policy.normalize_language(args.source_language)
            print(f"[LID_OVERRIDE] language={language}", flush=True)
        route = policy.choose_route(language)
        print(
            f"[AUTO_SELECT] language={language} engine={route.engine} "
            f"evidence={route.sample} measured_error="
            f"{str(route.sample_error) + '%' if route.validated else 'unmeasured'} "
            f"metric={route.metric or 'n/a'}",
            flush=True,
        )
        print(
            "[LANGUAGE] " + policy.LANGUAGE_NAMES.get(language, language),
            flush=True,
        )
        target = media.with_name(
            media.stem + policy.OUTPUT_SUFFIX[route.engine]
        )
        if args.probe_only:
            print("[PROBE_ONLY] No ASR result was written", flush=True)
            return 0
        if target.is_file():
            # Existing valid subtitles are always retained.
            from cli import validate_srt
            count = validate_srt(target)
            print(f"[ASR_SKIPPED] Valid existing SRT cues={count}", flush=True)
        elif route.engine == "whisper":
            run_whisper_existing(media, target)
        else:
            write_subtitles(media, target, route.engine, language)
        from cli import validate_srt
        cues = validate_srt(target)
        print(f"[AUTO_OUTPUT] {target}", flush=True)
        print(f"[AUTO_DONE] cues={cues}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[CANCELLED] Interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
