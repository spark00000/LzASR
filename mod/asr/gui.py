"""Tkinter GUI for LzASR; all model execution runs in a cancellable child process."""
from __future__ import annotations

from pathlib import Path
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
CLI = HERE / "cli.py"
INSTALLER = HERE / "install.ps1"


class ASRGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LzASR — Local Video / Audio Transcription")
        root.geometry("930x670")
        root.minsize(740, 530)
        self.items: list[str] = []
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.running = False
        self.engine = tk.StringVar(value="auto")
        self.translate = tk.BooleanVar(value=True)
        self.recursive = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="준비됨 — 입력 영상 또는 폴더를 선택하십시오.")
        self.counter = tk.StringVar(value="0 / 0")
        self._build()
        root.after(100, self._poll)

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="LzASR  ·  로컬 음성 인식 / 자막 생성",
                  font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 10))
        select = ttk.LabelFrame(frame, text="1. 입력 파일 / 폴더", padding=8)
        select.pack(fill="both", expand=True)
        bar = ttk.Frame(select)
        bar.pack(fill="x", pady=(0, 5))
        self.add_files_btn = ttk.Button(bar, text="파일 추가", command=self.add_files)
        self.add_files_btn.pack(side="left", padx=(0, 5))
        self.add_folder_btn = ttk.Button(bar, text="폴더 추가", command=self.add_folder)
        self.add_folder_btn.pack(side="left", padx=(0, 5))
        self.remove_btn = ttk.Button(bar, text="선택 제거", command=self.remove_selected)
        self.remove_btn.pack(side="left", padx=(0, 5))
        self.clear_btn = ttk.Button(bar, text="전체 비우기", command=self.clear)
        self.clear_btn.pack(side="left")
        ttk.Checkbutton(bar, text="하위 폴더 포함", variable=self.recursive).pack(side="right")
        chooser = ttk.Frame(select)
        chooser.pack(fill="both", expand=True)
        self.filelist = tk.Listbox(chooser, selectmode="extended", height=8,
                                  font=("Segoe UI", 10), exportselection=False)
        self.filelist.pack(side="left", fill="both", expand=True)
        scrolling = ttk.Scrollbar(chooser, orient="vertical", command=self.filelist.yview)
        scrolling.pack(side="right", fill="y")
        self.filelist.configure(yscrollcommand=scrolling.set)

        options = ttk.LabelFrame(frame, text="2. 모델 / 실행", padding=8)
        options.pack(fill="x", pady=(9, 0))
        row = ttk.Frame(options)
        row.pack(fill="x")
        ttk.Label(row, text="ASR 모델:").pack(side="left", padx=(0, 6))
        self.models = ttk.Combobox(
            row, state="readonly", textvariable=self.engine,
            values=("auto", "whisper", "qwen17", "qwen06"), width=14,
        )
        self.models.pack(side="left", padx=(0, 8))
        ttk.Label(row, text="auto: 언어 감지 후 언어별 모델 선택").pack(side="left")
        controls = ttk.Frame(options)
        controls.pack(fill="x", pady=(9, 0))
        self.start_btn = ttk.Button(controls, text="▶ 시작", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 7))
        ttk.Checkbutton(controls, text="한국어 자막도 생성", variable=self.translate).pack(side="left", padx=(0, 7))
        self.stop_btn = ttk.Button(controls, text="■ 중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 7))
        self.check_btn = ttk.Button(controls, text="환경 검사", command=self.check)
        self.check_btn.pack(side="left", padx=(0, 7))
        self.install_btn = ttk.Button(controls, text="누락 항목 설치", command=self.install)
        self.install_btn.pack(side="left")
        ttk.Label(controls, textvariable=self.counter).pack(side="right")
        self.progress = ttk.Progressbar(options, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 0))
        ttk.Label(options, textvariable=self.status, wraplength=830).pack(anchor="w", pady=(6, 0))

        logs = ttk.LabelFrame(frame, text="3. 실시간 실행 기록", padding=8)
        logs.pack(fill="both", expand=True, pady=(9, 0))
        container = ttk.Frame(logs)
        container.pack(fill="both", expand=True)
        self.log = tk.Text(container, height=10, font=("Consolas", 9),
                           wrap="word", state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        logscroll = ttk.Scrollbar(container, command=self.log.yview)
        logscroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=logscroll.set)
        ttk.Label(frame,
                  text="SRT는 입력 파일과 같은 폴더에 저장되며 기존 자막은 덮어쓰지 않습니다.",
                  foreground="#555555").pack(anchor="w", pady=(8, 0))

    def _add(self, paths: tuple[str, ...] | list[str]) -> None:
        seen = {s.casefold() for s in self.items}
        for entry in paths:
            full = str(Path(entry).resolve())
            if full.casefold() not in seen:
                seen.add(full.casefold())
                self.items.append(full)
                self.filelist.insert("end", full)
        self.status.set(f"{len(self.items)}개 입력 경로 선택됨")

    def add_files(self) -> None:
        if self.running:
            return
        selected = filedialog.askopenfilenames(
            title="영상 / 음성 파일 다중 선택",
            filetypes=[("Audio / Video", "*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                       ("모든 파일", "*.*")],
        )
        self._add(selected)

    def add_folder(self) -> None:
        if self.running:
            return
        selected = filedialog.askdirectory(title="영상 / 음성 파일 폴더 선택")
        if selected:
            self._add([selected])

    def remove_selected(self) -> None:
        if self.running:
            return
        for index in reversed(self.filelist.curselection()):
            self.filelist.delete(index)
            del self.items[index]
        self.status.set(f"{len(self.items)}개 입력 경로 선택됨")

    def clear(self) -> None:
        if self.running:
            return
        self.items.clear()
        self.filelist.delete(0, "end")
        self.status.set("입력 목록을 비웠습니다.")

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        match = re.search(r"\[(?:START|RESULT)\s+(\d+)/(\d+)\]", text)
        if match:
            self.counter.set(match.group(1) + " / " + match.group(2))

    def _execute(self, args: list[str], label: str) -> None:
        if self.running:
            return
        self.running = True
        self.status.set(label)
        for btn in (self.start_btn, self.check_btn, self.install_btn,
                    self.add_files_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            btn.configure(state="disabled")
        self.models.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(12)
        self._append("> " + subprocess.list2cmdline(args))
        thread = threading.Thread(target=self._worker, args=(args,), daemon=True)
        thread.start()

    def _worker(self, cmd: list[str]) -> None:
        result = 1
        try:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
                creationflags=flags,
            )
            self.process = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                self.messages.put(("log", line.rstrip()))
            result = proc.wait()
        except (OSError, ValueError) as exc:
            self.messages.put(("log", f"[ERROR] {type(exc).__name__}: {exc}"))
        finally:
            self.process = None
            self.messages.put(("finished", result))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    self._append(str(value))
                elif kind == "finished":
                    self.running = False
                    self.progress.stop()
                    self.stop_btn.configure(state="disabled")
                    self.models.configure(state="readonly")
                    for btn in (self.start_btn, self.check_btn, self.install_btn,
                                self.add_files_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
                        btn.configure(state="normal")
                    self.status.set("완료" if value == 0 else f"종료: 반환 코드 {value} (로그를 확인하십시오)")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def start(self) -> None:
        if not self.items:
            messagebox.showwarning("입력 필요", "음성 또는 동영상 파일을 추가하십시오.")
            return
        python = str(Path(sys.executable).with_name("python.exe")) if Path(sys.executable).name.lower() == "pythonw.exe" else sys.executable
        args = [python, "-B", str(CLI), "--engine", self.engine.get()]
        if not self.translate.get():
            args.append("--asr-only")
        if self.recursive.get():
            args.append("--recursive")
        args.extend(self.items)
        self.counter.set("0 / 0")
        self._execute(args, "로컬 모델로 처리 중입니다.")

    def check(self) -> None:
        python = str(Path(sys.executable).with_name("python.exe")) if Path(sys.executable).name.lower() == "pythonw.exe" else sys.executable
        self._execute([python, "-B", str(CLI), "--check", "--engine",
                       self.engine.get()], "설치된 환경을 검사하는 중입니다.")

    def install(self) -> None:
        if not messagebox.askyesno(
            "설치 확인",
            "선택한 모델의 누락된 엔진/모델/의존성을 로컬 런타임 폴더에 다운로드하겠습니다.\n"
            "수 GB의 저장 공간과 네트워크 사용이 발생할 수 있습니다. 진행하시겠습니까?",
        ):
            return
        self._execute(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(INSTALLER), "-Install", "-Engine",
             "all" if self.engine.get() == "auto" else self.engine.get()],
            "누락된 외부 의존성을 설치하는 중입니다.",
        )

    def stop(self) -> None:
        proc = self.process
        if proc is None or proc.poll() is not None:
            self._append("[CANCEL] 시작/완료 중이므로 종료할 실행 프로세스가 없습니다.")
            return
        self.status.set("중지 요청 중입니다...")
        self._append("[CANCEL] 현재 처리의 프로세스 트리만 종료합니다.")
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)


def main() -> int:
    root = tk.Tk()
    application = ASRGui(root)
    if "--self-test" in sys.argv:
        root.update_idletasks()
        assert application.engine.get() == "auto"
        assert application.start_btn.instate(["!disabled"])
        assert len(application.items) == 0
        root.destroy()
        print("[PASS] GUI built and controls initialized")
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
