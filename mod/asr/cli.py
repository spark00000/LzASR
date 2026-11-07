"""Local ASR plus Korean subtitles. All outputs are created next to input media.

One file with no flags: Whisper large-v3-turbo + external translator MiLMMT-4B.
No remote inference or automatic installation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ENGINES = {
    "auto": "_selected-model.srt",  # preview only; the selected model writes its own SRT
    "whisper": "_whisper-large-v3-turbo.srt",
    "qwen17": "_qwen3-asr-1.7b.srt",
    "qwen06": "_qwen3-asr-0.6b.srt",
}


def output_for(media: Path, engine: str) -> Path:
    return media.with_name(media.stem + ENGINES[engine])


def validate_srt(path: Path) -> int:
    """Reject malformed or empty subtitle cues without external dependencies."""
    contents = path.read_text(encoding="utf-8-sig")
    blocks = [s.strip() for s in re.split(r"\r?\n\s*\r?\n", contents.strip()) if s.strip()]
    if not blocks:
        raise ValueError("SRT has no subtitle blocks")
    for number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(number) or not STAMP.fullmatch(lines[1].strip()):
            raise ValueError(f"Malformed SRT cue {number}")
        if not any(line.strip() for line in lines[2:]):
            raise ValueError(f"Empty SRT cue {number}")
    return len(blocks)



APP_ROOT = Path(os.environ.get("LZASR_APP_HOME", str(Path(Path(__file__).resolve().anchor) / "APP" / "LzASR")))
HERE = Path(__file__).resolve().parent
DRIVER = HERE / "ASR.bat"
INSTALLER = HERE / "install.ps1"
# An optional local translator can be selected explicitly, or discovered
# by its generic launcher name when exactly one sibling installation exists.
_translator_override = os.environ.get("LZASR_TRANSLATOR_CLI")
_translator_candidates = sorted(HERE.parents[2].glob("*/out/translate.bat"))
TRANSLATE_BAT = (
    Path(_translator_override)
    if _translator_override else
    (_translator_candidates[0] if len(_translator_candidates) == 1 else HERE / "unconfigured-translator.bat")
)
TRANSLATE_RUNTIME = Path(os.environ.get(
    "LZASR_TRANSLATOR_HOME", str(APP_ROOT.parent / TRANSLATE_BAT.parent.parent.name)
))
DEFAULT_ENGINE = "auto"
DEFAULT_TRANSLATOR = "mi4"
EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".mpg", ".mpeg",
    ".ts", ".m2ts", ".wmv", ".mp3", ".wav", ".flac", ".m4a",
    ".aac", ".ogg", ".wma", ".aif", ".aiff",
}
KANA = re.compile(r"[\u3040-\u30ff]")
HANGUL = re.compile(r"[\uac00-\ud7af]")
HAN = re.compile(r"[\u3400-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
STAMP = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")
LANGUAGES = {
    "Chinese": "Chinese (Simplified)",
    "Mandarin": "Chinese (Simplified)",
    "Cantonese": "Chinese (Traditional)",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "English": "English",
    "French": "French",
    "German": "German",
    "Spanish": "Spanish",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Hindi": "Hindi",
    "Vietnamese": "Vietnamese",
    "Indonesian": "Indonesian",
    "Thai": "Thai",
    "Turkish": "Turkish",
    "Dutch": "Dutch",
    "Polish": "Polish",
    "Czech": "Czech",
}
# Explicit source-language is still recommended for Latin-script languages other than English.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def translation_output(source: Path, model: str) -> Path:
    return source.with_name(f"{source.stem}.{model}.ko.srt")


def requirements(engine: str, translate: bool = False) -> list[str]:
    missing: list[str] = []
    if not DRIVER.is_file():
        missing.append(str(DRIVER))
    if engine == "auto":
        py = APP_ROOT / ".venv-bench" / "Scripts" / "python.exe"
        if not py.is_file():
            missing.append("Local ASR benchmark Python environment")
        if not (HERE / "auto_asr.py").is_file():
            missing.append("Automatic ASR source")
        if not (HERE / "auto_policy.py").is_file():
            missing.append("Automatic ASR routing policy")
        if not (APP_ROOT / "models" / "whisper-large-v3" / "model.safetensors").is_file():
            missing.append("Whisper language detector weights")
        if not shutil.which("ffmpeg") and not (APP_ROOT / "engine" / "ffmpeg" / "ffmpeg.exe").is_file():
            missing.append("ffmpeg")
        if not (APP_ROOT / "models" / "faster-whisper-large-v3-turbo" / "model.bin").is_file():
            missing.append("Whisper default fallback weights")
        if not (APP_ROOT / "engine" / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe").is_file():
            missing.append("Whisper default fallback executable")
    elif engine == "whisper":
        if not (APP_ROOT / "engine" / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe").is_file():
            missing.append("Faster-Whisper-XXL executable")
        if not (APP_ROOT / "models" / "faster-whisper-large-v3-turbo" / "model.bin").is_file():
            missing.append("Whisper large-v3-turbo weights")
    else:
        if not (APP_ROOT / ".venv-qwen" / "Scripts" / "python.exe").is_file():
            missing.append("Qwen Python environment")
        if not (HERE / "qwen_asr_to_srt.py").is_file():
            missing.append("Qwen ASR source")
        try:
            from qwen_asr_to_srt import model_ready
            model = "Qwen3-ASR-1.7B" if engine == "qwen17" else "Qwen3-ASR-0.6B"
            for name in (model, "Qwen3-ForcedAligner-0.6B"):
                if not model_ready(APP_ROOT / "models" / name):
                    missing.append(f"{name} weights")
        except ImportError:
            missing.append("Qwen model validation module")
        if not shutil.which("ffmpeg") and not (APP_ROOT / "engine" / "ffmpeg" / "ffmpeg.exe").is_file():
            missing.append("ffmpeg.exe")
    if translate:
        if not TRANSLATE_BAT.is_file():
            missing.append(str(TRANSLATE_BAT))
        if not TRANSLATE_RUNTIME / "engine" / "venv" / "Scripts" / "python.exe".is_file():
            missing.append("external translator Python environment")
    return missing


def inputs(paths: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for text in paths:
        path = Path(text).expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            scan = path.rglob("*") if recursive else path.iterdir()
            candidates = sorted(
                (p for p in scan if p.is_file() and p.suffix.lower() in EXTENSIONS),
                key=lambda p: str(p).casefold(),
            )
        else:
            raise ValueError(f"Input does not exist: {path}")
        for candidate in candidates:
            if candidate.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"Unsupported media type: {candidate}")
            key = str(candidate).casefold()
            if key not in seen:
                found.append(candidate)
                seen.add(key)
    return found


def clock() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def gpu_sample() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "unavailable"
    try:
        result = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "unavailable"


def install_missing(engine: str) -> bool:
    if not INSTALLER.is_file():
        print("[ERROR] ASR installer missing", flush=True)
        return False
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALLER), "-Install", "-Engine", engine], check=False,
    )
    return process.returncode == 0


def run_asr(media: Path, engine: str) -> tuple[int, str | None]:
    """Stream subprocess output without swallowing errors; collect Qwen language labels."""
    command = ["cmd.exe", "/d", "/c", str(DRIVER), "--engine", engine, str(media)]
    found: list[str] = []
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        command, cwd=str(HERE), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("[LANGUAGE] "):
            found.append(line.split("]", 1)[1].strip())
    code = proc.wait()
    language = Counter(found).most_common(1)[0][0] if found else None
    return code, language


def run_auto_asr(media: Path, override: str) -> tuple[int, str | None, Path | None, bool]:
    """Auto LID + pre-tested local route; capture actual selected output SRT."""
    py = APP_ROOT / ".venv-bench" / "Scripts" / "python.exe"
    cmd = [str(py), "-B", str(HERE / "auto_asr.py"),
           str(media), "--source-language", override]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    found_output: Path | None = None
    language: str | None = None
    skipped = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("[AUTO_OUTPUT] "):
            found_output = Path(line.split("]", 1)[1].strip())
        elif line.startswith("[LANGUAGE] "):
            language = line.split("]", 1)[1].strip()
        elif line.startswith("[ASR_SKIPPED] "):
            skipped = True
    return proc.wait(), language, found_output, skipped


def detect_language(srt: Path, emitted: str | None, override: str) -> str:
    if override != "auto":
        return override
    if emitted and emitted in LANGUAGES:
        return LANGUAGES[emitted]
    # For already-created subtitles, deterministic script cues are preferable to
    # inventing the source language. Latin-script languages must use --source-language
    # unless an ASR language identifier is available.
    body = []
    for line in srt.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.isdigit() or STAMP.fullmatch(line):
            continue
        body.append(line)
        if len(body) >= 50:
            break
    sample = " ".join(body)
    if len(KANA.findall(sample)) >= 2:
        return "Japanese"
    if len(HANGUL.findall(sample)) >= 2:
        return "Korean"
    if len(HAN.findall(sample)) >= 5:
        return "Chinese (Simplified)"
    if len(LATIN.findall(sample)) >= 15:
        print("[LANGUAGE_WARNING] Latin script without ASR language label: assuming English. "
              "Use --source-language for other Latin-script languages.", flush=True)
        return "English"
    raise ValueError("Cannot determine source language; use --source-language explicitly")


def srt_rows(path: Path) -> list[tuple[str, str]]:
    """Return ordered (timestamp, caption) pairs from a structurally valid SRT."""
    value = path.read_text(encoding="utf-8-sig")
    blocks = [block.strip() for block in re.split(r"\r?\n\s*\r?\n", value.strip()) if block.strip()]
    result: list[tuple[str, str]] = []
    for idx, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(idx) or not STAMP.fullmatch(lines[1].strip()):
            raise ValueError(f"Malformed subtitle cue {idx}")
        text = " ".join(line.strip() for line in lines[2:]).strip()
        if not text:
            raise ValueError(f"Empty subtitle cue {idx}")
        result.append((lines[1].strip(), text))
    if not result:
        raise ValueError(f"Empty subtitle file: {path}")
    return result


def nonlexical_only(text: str) -> bool:
    """A punctuation/prolonged-sound cue is not a lexical translation failure.

    Japanese elongated vowel marks (ーーー!) can be preserved verbatim; words
    or other letters must still translate to actual Korean script.
    """
    return bool(text.strip()) and re.sub(r"[\s\-—–―ーｰ~〜～!！?？.,。、…：:;；・'\"“”]+", "", text) == ""


def check_alignment(original: Path, korean: Path) -> int:
    """Validate source/target numbering, timestamps and Korean lexical text."""
    src, ko = srt_rows(original), srt_rows(korean)
    if len(src) != len(ko):
        raise ValueError(f"Translation cue count changed: {len(src)} vs {len(ko)}")
    for idx, ((stamp, old), (kstamp, text)) in enumerate(zip(src, ko), 1):
        if stamp != kstamp:
            raise ValueError(f"Translation timestamp mismatch at cue {idx}")
        if not HANGUL.search(text) and not (nonlexical_only(old) and text == old):
            raise ValueError(f"Untranslated lexical content at cue {idx}")
    return len(src)


def recover_nonlexical_trace(source: Path, target: Path, model: str,
                             trace: Path, summary: Path) -> int:
    """Recover only a fully audited MiLMMT trace with punctuation-only exceptions.

    Never invent missing translations, retry uncertain requests or modify a
    failed source SRT. external translator's original FAILED evidence is retained.
    """
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")
    if not trace.is_file() or not summary.is_file():
        raise RuntimeError("Incomplete external translator trace; cannot recover")
    report = json.loads(summary.read_text(encoding="utf-8"))
    rows = srt_rows(source)
    expected = "milmmt4" if model == "mi4" else model
    if (report.get("status") != "FAILED" or report.get("model") != expected
            or report.get("source") != str(source)
            or report.get("source_sha256", "").casefold() != sha256(source)
            or report.get("cue_count") != len(rows)):
        raise RuntimeError("external translator report fingerprint/model does not match; refuse recovery")
    entries = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != len(rows):
        raise RuntimeError("Translation trace incomplete; cannot recover")
    recovered: list[str] = []
    exceptions = 0
    valid_count = 0
    for idx, ((timestamp, original), item) in enumerate(zip(rows, entries), 1):
        rendered = str(item.get("output_ko", "")).replace("\r", " ").replace("\n", " ").strip()
        if (item.get("cue") != idx or item.get("timecode") != timestamp
                or item.get("source") != original):
            raise RuntimeError(f"Trace does not match source cue {idx}")
        if item.get("status") == "VALID" and HANGUL.search(rendered):
            valid_count += 1
        elif (item.get("status") == "NOT_KOREAN"
              and nonlexical_only(original) and rendered == original):
            exceptions += 1
            print(f"[NONLEXICAL_CUE] {idx}: punctuation/sound marker preserved verbatim", flush=True)
        else:
            raise RuntimeError(f"Untranslated lexical content in trace cue {idx}: STOP")
        recovered.append(f"{idx}\r\n{timestamp}\r\n{rendered}")
    if exceptions < 1 or report.get("valid_korean") != valid_count:
        raise RuntimeError("Recovery requires a verified nonlexical-only exception")
    payload = "\r\n\r\n".join(recovered) + "\r\n"
    with target.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)
    checked = check_alignment(source, target)
    print(f"[TRANSLATE_RECOVERED] {target} cues={checked} nonlexical={exceptions}; "
          "original external translator FAILED report retained for audit", flush=True)
    return checked


def archive_translation_diagnostics(source: Path, target: Path) -> None:
    """Keep translation JSONL/reports out of the media directory.

    Preserve their exact bytes under external translator's already existing runtime
    log directory, with source-path and content-derived collision-proof names.
    The public output directory then contains only video/audio inputs and SRT.
    """
    sidecars = (
        target.with_suffix(target.suffix + ".jsonl"),
        target.with_suffix(target.suffix + ".report.json"),
    )
    present = [path for path in sidecars if path.is_file()]
    if not present:
        return
    log_root = TRANSLATE_RUNTIME / "logs"
    if not log_root.is_dir():
        raise RuntimeError(f"Translation diagnostic log directory not found: {log_root}")
    source_key = hashlib.sha256(str(source.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    for path in present:
        digest = sha256(path)
        archive = log_root / f"asr_{source_key}_{digest[:16]}_{path.name}"
        if archive.exists():
            if sha256(archive) != digest:
                raise FileExistsError(f"Translation diagnostic archive collision: {archive}")
            # Same exact contents are already in the archive, confirmed by SHA256.
            path.unlink()
        else:
            path.replace(archive)
            if sha256(archive) != digest:
                raise RuntimeError(f"Translation archive integrity check failed: {archive}")
        print(f"[DIAGNOSTIC_ARCHIVED] {archive}", flush=True)


def run_translate(source: Path, model: str, language: str) -> tuple[str, Path]:
    target = translation_output(source, model)
    if target.exists():
        count = check_alignment(source, target)
        print(f"[TRANSLATE_SKIPPED] Existing Korean SRT verified: {target} cues={count}", flush=True)
        return "SKIPPED", target
    if not TRANSLATE_BAT.is_file():
        raise RuntimeError(f"external translator CLI unavailable: {TRANSLATE_BAT}")
    before = sha256(source)
    trace = target.with_suffix(target.suffix + ".jsonl")
    report = target.with_suffix(target.suffix + ".report.json")
    if trace.exists() or report.exists():
        # Earlier failed translation: verify the complete existing trace.
        try:
            recover_nonlexical_trace(source, target, model, trace, report)
        finally:
            archive_translation_diagnostics(source, target)
        return "RECOVERED_NONLEXICAL", target
    cmd = ["cmd.exe", "/d", "/c", str(TRANSLATE_BAT), "--input", str(source),
           "--models", model, "--source-language", language]
    print(f"[TRANSLATE_START] model={model} source_language={language}", flush=True)
    try:
        exit_code = subprocess.run(cmd, cwd=str(HERE), check=False).returncode
        if sha256(source) != before:
            raise RuntimeError("ASR source SRT changed while translating")
        if exit_code != 0:
            try:
                recover_nonlexical_trace(source, target, model, trace, report)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"external translator failed with exit {exit_code}; "
                                   f"no safe recovery: {exc}") from exc
            return "RECOVERED_NONLEXICAL", target
        if not target.is_file():
            raise RuntimeError(f"external translator did not produce Korean SRT: {target}")
        count = check_alignment(source, target)
        print(f"[TRANSLATE_DONE] {target} cues={count}", flush=True)
        return "COMPLETED", target
    finally:
        archive_translation_diagnostics(source, target)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="LzASR: local video/audio -> model-named SRT + Korean SRT via external translator"
    )
    parser.add_argument("media", nargs="*", help="Video/audio files or folders")
    parser.add_argument("--engine", choices=tuple(ENGINES), default=DEFAULT_ENGINE,
                        help=f"ASR model (default: {DEFAULT_ENGINE})")
    parser.add_argument("--asr-only", "--no-translate", action="store_true",
                        help="Only create source-language SRT; skip Korean translation")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATOR,
                        help=f"external translator model ID (default: {DEFAULT_TRANSLATOR})")
    parser.add_argument("--source-language", default="auto",
                        help="ASR language override for auto routing; also used for optional translation")
    parser.add_argument("--recursive", action="store_true", help="Include nested folders")
    parser.add_argument("--check", action="store_true", help="Check local runtime without inference")
    parser.add_argument("--list-routing", action="store_true",
                        help="List all 100 language codes and their local model routes")
    parser.add_argument("--probe-only", action="store_true",
                        help="Detect language and select local model; write no output")
    parser.add_argument("--install-missing", action="store_true",
                        help="Explicitly install missing ASR dependencies (not external translator)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing")
    args = parser.parse_args()

    if args.check and args.media:
        parser.error("--check cannot be combined with media")
    if args.list_routing:
        from auto_policy import supported_languages, choose_route
        langs = supported_languages()
        for lang in sorted(langs):
            route = choose_route(lang)
            print(f"{lang}\t{route.engine}\t{route.sample}\t"
                  f"{route.sample_error if route.validated else '-'}", flush=True)
        print(f"[ROUTES] {len(langs)} languages; observed results are not full-test rankings",
              flush=True)
        return 0
    if args.probe_only and args.engine != "auto":
        parser.error("--probe-only requires --engine auto")
    if not args.media and not args.check:
        parser.print_help()
        return 2
    need_translation = not args.asr_only and not args.probe_only
    missing = requirements(args.engine, translate=need_translation)
    if args.check:
        print(f"[CHECK] engine={args.engine} translation={'off' if args.asr_only else args.translation_model}")
        for entry in missing:
            print("[MISSING]", entry)
        if not missing:
            print("[PASS] Entrypoints, engine, model files and translation runtime present")
        return 0 if not missing else 2

    try:
        media = inputs(args.media, args.recursive)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    if not media:
        print("[ERROR] No supported media files found", file=sys.stderr)
        return 2
    print(f"[PLAN] engine={args.engine}, translation={'off' if args.asr_only else args.translation_model}, files={len(media)}", flush=True)
    for index, item in enumerate(media, 1):
        if args.engine == "auto":
            print(f"[FILE {index}/{len(media)}] {item} -> selected-model SRT after language detection", flush=True)
            if need_translation:
                print("[KOREAN] After language/model selection", flush=True)
        else:
            source = output_for(item, args.engine)
            print(f"[FILE {index}/{len(media)}] {item} -> {source}", flush=True)
            if need_translation:
                print(f"[KOREAN] {translation_output(source, args.translation_model)}", flush=True)
    if args.dry_run:
        print("[DRY-RUN] No audio inference, model loading or output creation")
        return 0

    if missing and args.install_missing:
        if not install_missing("whisper" if args.engine == "auto" else args.engine):
            return 2
        missing = requirements(args.engine, translate=need_translation)
    if missing:
        for entry in missing:
            print(f"[MISSING] {entry}", file=sys.stderr)
        print("[ERROR] Install missing dependencies before execution", file=sys.stderr)
        return 2

    if args.probe_only:
        python = APP_ROOT / ".venv-bench" / "Scripts" / "python.exe"
        for item in media:
            code = subprocess.run(
                [str(python), "-B", str(HERE / "auto_asr.py"),
                 str(item), "--source-language", args.source_language,
                 "--probe-only"], check=False,
            ).returncode
            if code:
                return code
        return 0

    records: list[dict[str, object]] = []
    start = time.monotonic()
    failed = False
    for index, item in enumerate(media, 1):
        original_sha = sha256(item)
        source = output_for(item, args.engine)
        stamp = clock()
        began = time.monotonic()
        status = "SKIPPED"
        translated = "DISABLED"
        korean_output: Path | None = None
        diagnostic = ""
        emitted_language: str | None = None
        print(f"[START {index}/{len(media)}] {stamp} {item}", flush=True)
        try:
            if args.engine == "auto":
                exit_code, emitted_language, produced, was_skipped = run_auto_asr(
                    item, args.source_language
                )
                if exit_code != 0 or produced is None:
                    raise RuntimeError(f"Automatic ASR failed with exit {exit_code}")
                if not produced.is_file():
                    raise RuntimeError("Automatic ASR did not create its selected SRT")
                source = produced
                count = validate_srt(source)
                print(f"[ASR_DONE] {source} cues={count}", flush=True)
                status = "SKIPPED" if was_skipped else "COMPLETED"
            elif source.exists():
                print(f"[ASR_SKIPPED] {validate_srt(source)} existing cues; original unchanged", flush=True)
            else:
                exit_code, emitted_language = run_asr(item, args.engine)
                if exit_code != 0:
                    raise RuntimeError(f"ASR engine failed with exit {exit_code}")
                if not source.is_file():
                    raise RuntimeError(f"ASR did not generate expected SRT: {source}")
                count = validate_srt(source)
                print(f"[ASR_DONE] {source} cues={count}", flush=True)
                status = "COMPLETED"
            if sha256(item) != original_sha:
                raise RuntimeError("Input media changed while running")
            if need_translation:
                language = detect_language(source, emitted_language, args.source_language)
                translated, korean_output = run_translate(source, args.translation_model, language)
            if sha256(item) != original_sha:
                raise RuntimeError("Input media changed during translation")
        except (OSError, ValueError, RuntimeError, UnicodeError) as error:
            diagnostic = f"{type(error).__name__}: {error}"
            status = "FAILED"
            failed = True
            print(f"[ERROR] {diagnostic}", flush=True)
        except KeyboardInterrupt:
            print("[CANCELLED] Interrupted by user", flush=True)
            return 130
        elapsed = round(time.monotonic() - began, 3)
        print(f"[RESULT {index}/{len(media)}] {status} translation={translated} "
              f"elapsed_s={elapsed} gpu_snapshot={gpu_sample()} {diagnostic}", flush=True)
        records.append({
            "input": str(item), "source": str(source), "korean": str(korean_output) if korean_output else None,
            "engine": args.engine, "translation_model": args.translation_model if need_translation else None,
            "started": stamp, "elapsed_s": elapsed, "status": status, "translated": translated,
            "error": diagnostic,
        })
        if failed:
            print("[STOP] First failing file stops the batch; earlier outputs are preserved", flush=True)
            break
    total = round(time.monotonic() - start, 3)
    print(f"[DONE] files={len(records)}/{len(media)} failed={int(failed)} elapsed_s={total}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
