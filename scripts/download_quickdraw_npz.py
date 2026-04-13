#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import contextlib
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'decoder'))
from utils import Category  # noqa: E402


BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn/'


def remote_size_for(name: str, timeout: int) -> int | None:
    url = BASE_URL + urllib.parse.quote(name) + '.full.npz'
    req = urllib.request.Request(url, method='HEAD')
    with contextlib.closing(urllib.request.urlopen(req, timeout=timeout)) as response:
        length = response.headers.get('Content-Length')
    return int(length) if length is not None else None


def is_valid_npz(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, 'missing'
    if path.stat().st_size == 0:
        return False, 'empty'
    try:
        with np.load(path, allow_pickle=True, encoding='latin1') as data:
            keys = set(data.files)
        required = {'train', 'valid', 'test'}
        if not required.issubset(keys):
            return False, f'missing_keys:{sorted(required - keys)}'
        return True, 'ok'
    except Exception as exc:  # pragma: no cover - defensive integrity check
        return False, f'corrupt:{exc}'


def verify_with_remote(name: str, path: Path, timeout: int) -> tuple[bool, str]:
    valid, reason = is_valid_npz(path)
    if not valid:
        return valid, reason
    try:
        remote_size = remote_size_for(name, timeout)
        if remote_size is not None and path.stat().st_size != remote_size:
            return False, f'size_mismatch:local={path.stat().st_size},remote={remote_size}'
    except Exception as exc:
        return False, f'remote_check_failed:{exc}'
    return True, 'ok'


def download_one(folder: Path, timeout: int) -> tuple[str, int]:
    name = folder.name
    url = BASE_URL + urllib.parse.quote(name) + '.full.npz'
    dest = folder / 'data.full.npz'
    tmp = folder / 'data.full.npz.part'
    folder.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open('wb') as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    valid, reason = is_valid_npz(tmp)
    if not valid:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'invalid download for {name}: {reason}')
    tmp.replace(dest)
    return name, dest.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description='Download or repair QuickDraw data.full.npz files.')
    parser.add_argument('--root', default='/workspace/home/oujingfeng/project/Latent-Sketchpad/decoder/QuickDraw')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--skip-remote-size-check', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    categories = [c.value for c in Category]
    status = []
    for name in categories:
        folder = root / name
        if args.skip_remote_size_check:
            valid, reason = is_valid_npz(folder / 'data.full.npz')
        else:
            valid, reason = verify_with_remote(name, folder / 'data.full.npz', args.timeout)
        status.append((name, valid, reason))

    missing_or_bad = [(root / name, reason) for name, valid, reason in status if not valid]
    ok_count = sum(1 for _, valid, _ in status if valid)
    print({'total_categories': len(categories), 'valid': ok_count, 'needs_download': len(missing_or_bad)}, flush=True)
    if missing_or_bad:
        print('NEEDS_DOWNLOAD_START', flush=True)
        for folder, reason in missing_or_bad:
            print(f'{folder.name}: {reason}', flush=True)
        print('NEEDS_DOWNLOAD_END', flush=True)

    if args.verify_only or not missing_or_bad:
        return

    completed = 0
    failed = 0
    lock = threading.Lock()
    start = time.time()

    def wrapped(item: tuple[Path, str]) -> tuple[str, int]:
        folder, _reason = item
        return download_one(folder, args.timeout)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(wrapped, item): item[0].name for item in missing_or_bad}
        total = len(futures)
        for future in cf.as_completed(futures):
            name = futures[future]
            try:
                _name, size = future.result()
                with lock:
                    completed += 1
                    elapsed = time.time() - start
                    print(f'[{completed}/{total}] ok: {name} ({size} bytes) elapsed={elapsed:.1f}s', flush=True)
            except Exception as exc:
                with lock:
                    failed += 1
                    completed += 1
                    elapsed = time.time() - start
                    print(f'[{completed}/{total}] fail: {name} -> {exc} elapsed={elapsed:.1f}s', flush=True)

    # Final verification pass.
    bad_after = []
    for name in categories:
        path = root / name / 'data.full.npz'
        if args.skip_remote_size_check:
            valid, reason = is_valid_npz(path)
        else:
            valid, reason = verify_with_remote(name, path, args.timeout)
        if not valid:
            bad_after.append((name, reason))
    print({'final_invalid': len(bad_after), 'failed_downloads': failed}, flush=True)
    if bad_after:
        print('FINAL_INVALID_START', flush=True)
        for name, reason in bad_after:
            print(f'{name}: {reason}', flush=True)
        print('FINAL_INVALID_END', flush=True)


if __name__ == '__main__':
    main()
