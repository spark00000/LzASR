# ASR module — Automatic Speech Recognition

Speech recognition converts audio into written text. SRT generation additionally requires timing. Transcript accuracy and subtitle time-alignment accuracy are separate quality measures.

| Module | Role |
|---|---|
| cli.py | Local batch CLI; auto language routing is the default, optional separate translation |
| auto_asr.py | ffmpeg streaming RMS/FFT peak selection, 5-window Whisper LID vote, full-file model-specific SRT |
| auto_policy.py | Measured-language model preferences, 30 FLEURS locale aliases and all 100 Whisper ID-code fallbacks |
| gui.py | Tkinter GUI |
| ASR.bat | Model-specific driver for Whisper/Qwen |
| qwen_asr_to_srt.py | Qwen3-ASR and forced alignment |
| install.ps1 | Explicit external dependency check / installation |
| compare_asr.py | Read-only WER/CER comparison from model-named SRTs |
| bench_new_models.py | Additional open-weight model evaluations |
| bench_funasr.py | Fun-ASR experiment with known timestamp-head warning |

Language routing selects **one language and one model for the whole file**, without switching mid-video. RMS and speech-band FFT energy identify high-volume candidates; large music peaks may still cause false language votes. Different engines have different decoder behavior, and non-Whisper SRT timing uses coarse audio chunk boundaries rather than forced word alignment. Unmeasured languages run through the installed Whisper turbo fallback; no model download is initiated automatically.

Run the CLI or GUI from the repository root:

~~~bat
out\asr.bat --help
out\asr.bat --list-routing
out\install.bat -Check -Engine all
out\asr.bat "video.mp4" --probe-only
out\asr.bat "video.mp4" --asr-only
out\asr.bat "video.mp4" --source-language ja --asr-only
out\asr_gui.bat
~~~

Use LZASR_APP_HOME to configure an independent runtime directory containing the model weights, Python dependencies, and third-party executables. The default Windows fallback is APP\LzASR on the checkout drive. No test media, models or virtual environments are distributed in public source.

**Qualification:** selected transcript outputs and benchmark checks were tested; full multilingual benchmark and exact subtitle timestamps were not. See [the model comparison](../../doc/ASR_Model_Comparison_20261106.md).
