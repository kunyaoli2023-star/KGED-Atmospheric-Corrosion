from __future__ import annotations
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    """Atomic checkpoint: no half-written JSON after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(root, name, **values):
    row = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=name, **values)
    with (Path(root)/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


@contextmanager
def run_lock(root):
    """OS releases the lock on crash; stale files do not prevent resumption."""
    root = Path(root).resolve()
    root.parent.mkdir(parents=True,exist_ok=True)
    path = root.parent/('.'+root.name+'.run.lock')
    with path.open('a+b') as f:
        if not f.tell():
            f.write(b'0');f.flush()
        f.seek(0)
        locked = False
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise RuntimeError('RUN_ALREADY_OPEN_BY_ANOTHER_PROCESS') from None
        try:
            yield
        finally:
            if locked:
                f.seek(0)
                if os.name=='nt':
                    msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    fcntl.flock(f,fcntl.LOCK_UN)
