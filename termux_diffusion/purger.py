# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Clean-Room Purger module for terminating non-essential memory hogs.

Strictly preserves critical system daemons, user shells, SSH sessions, and multiplexers
while reclaiming physical RAM from zombie or idle inference processes before model loading.
Operates exclusively as an opt-in CLI flag (--kill [threshold_mb]).
"""

import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple

logger = logging.getLogger("termux_diffusion.purger")

# 절대 불사 화이트리스트: 사용자 원격 세션 및 OS 필수 프로세스 영구 보호
IMMORTAL_WHITELIST_COMM: Set[str] = {
    "sshd",
    "bash",
    "sh",
    "zsh",
    "fish",
    "tmux",
    "tmux: server",
    "screen",
    "tailscaled",
    "tailscale",
    "init",
    "zygote",
    "zygote64",
    "system_server",
}


@dataclass
class ProcessKillRecord:
    pid: int
    name: str
    rss_mb: float


@dataclass
class PurgeSummary:
    threshold_mb: float
    scanned_count: int
    killed_processes: List[ProcessKillRecord] = field(default_factory=list)

    @property
    def total_reclaimed_mb(self) -> float:
        return round(sum(p.rss_mb for p in self.killed_processes), 1)


def get_process_rss_mb(pid: int) -> float:
    """Read RSS memory deterministically from /proc/[pid]/statm in MB."""
    statm_path = Path(f"/proc/{pid}/statm")
    if not statm_path.exists():
        return 0.0
    try:
        with open(statm_path, "r", encoding="utf-8") as f:
            parts = f.readline().split()
            if len(parts) >= 2:
                # 2nd field is resident pages. 1 page = 4096 bytes on ARM64 Linux.
                resident_pages = int(parts[1])
                return round((resident_pages * 4096) / (1024 * 1024), 2)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0.0
    return 0.0


def get_process_comm(pid: int) -> str:
    """Read executable name deterministically from /proc/[pid]/comm."""
    comm_path = Path(f"/proc/{pid}/comm")
    if not comm_path.exists():
        return ""
    try:
        with open(comm_path, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def purge_idle_processes(threshold_mb: float = 50.0) -> PurgeSummary:
    """Scan and terminate processes exceeding threshold_mb, preserving whitelist."""
    self_pid = os.getpid()
    self_ppid = os.getppid()
    proc_dir = Path("/proc")
    summary = PurgeSummary(threshold_mb=threshold_mb, scanned_count=0)

    if not proc_dir.exists():
        logger.warning("/proc filesystem not accessible. Clean-room purge aborted.")
        return summary

    targets_to_terminate: List[Tuple[int, str, float]] = []

    for entry in proc_dir.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue

        pid = int(entry.name)
        summary.scanned_count += 1

        # 1. 자기 자신 및 직속 부모 프로세스 영구 보호
        if pid == self_pid or pid == self_ppid:
            continue

        comm = get_process_comm(pid)

        # 2. 불사 화이트리스트 보호
        if comm in IMMORTAL_WHITELIST_COMM:
            continue

        # 3. RSS 측정 및 임계치 검사
        rss_mb = get_process_rss_mb(pid)
        if rss_mb >= threshold_mb:
            targets_to_terminate.append((pid, comm, rss_mb))

    # 프로세스 종료 실행: SIGTERM 전송 후 0.5초 대기, 생존 시 SIGKILL 단행
    for pid, comm, rss_mb in targets_to_terminate:
        try:
            logger.info("Purging process %d (%s) consuming %.1f MB", pid, comm, rss_mb)
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue

    if targets_to_terminate:
        time.sleep(0.5)

    for pid, comm, rss_mb in targets_to_terminate:
        try:
            # 여전히 생존해 있는지 확인 (0 시그널 프로빙)
            os.kill(pid, 0)
            logger.warning("Process %d (%s) still alive, escalating to SIGKILL", pid, comm)
            os.kill(pid, signal.SIGKILL)
            summary.killed_processes.append(ProcessKillRecord(pid=pid, name=comm, rss_mb=rss_mb))
        except ProcessLookupError:
            # 정상 종료됨
            summary.killed_processes.append(ProcessKillRecord(pid=pid, name=comm, rss_mb=rss_mb))
        except PermissionError:
            pass

    return summary
