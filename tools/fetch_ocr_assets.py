"""Fetch Tesseract's WASM core and English language data for offline OCR.

The document reader runs entirely in the browser so that a patient's lab report
is never transmitted anywhere. That promise has a hole in it by default:
tesseract.js fetches its WASM core and language data from a CDN the first time
it runs, so the *first* scan needs the network even though the file itself does
not leave the machine.

This script closes it, by putting those assets in ``frontend/public/ocr/`` where
``readDocument.ts`` looks for them first. After running it, OCR works with the
network off.

Not needed for PDFs that carry a text layer, which is most lab reports - those
are read by pdf.js, which is bundled, and no OCR runs at all.

The files are gitignored: they are third-party build artefacts, not source, and
re-fetching them is one command.

Usage::

    python tools/fetch_ocr_assets.py
    python tools/fetch_ocr_assets.py --check    # report status, download nothing
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "frontend" / "public" / "ocr"

# Pinned to the versions in frontend/package.json. tesseract.js loads a core
# matching its own major version, so these move together.
TESSERACT_CORE = "6.0.0"

ASSETS = {
    # Language data. The "best_int" build is the accuracy/size compromise the
    # project recommends: ~2.9 MB against ~12 MB for the float model.
    "eng.traineddata.gz": (
        "https://cdn.jsdelivr.net/gh/naptha/tessdata@gh-pages/"
        "4.0.0_best_int/eng.traineddata.gz"
    ),
    # SIMD and non-SIMD cores: tesseract.js picks at runtime based on what the
    # browser supports, so both have to be present for this to work offline.
    "tesseract-core-simd.wasm": (
        f"https://cdn.jsdelivr.net/npm/tesseract.js-core@{TESSERACT_CORE}/"
        "tesseract-core-simd.wasm"
    ),
    "tesseract-core.wasm": (
        f"https://cdn.jsdelivr.net/npm/tesseract.js-core@{TESSERACT_CORE}/"
        "tesseract-core.wasm"
    ),
    "tesseract-core-simd.wasm.js": (
        f"https://cdn.jsdelivr.net/npm/tesseract.js-core@{TESSERACT_CORE}/"
        "tesseract-core-simd.wasm.js"
    ),
    "tesseract-core.wasm.js": (
        f"https://cdn.jsdelivr.net/npm/tesseract.js-core@{TESSERACT_CORE}/"
        "tesseract-core.wasm.js"
    ),
}

# Anything smaller than this came back as an error page, not an asset.
MIN_BYTES = 10_000


def human(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB" if n >= 1_048_576 else f"{n / 1024:.0f} KB"


def status() -> list[tuple[str, Path, bool]]:
    rows = []
    for name in ASSETS:
        path = OUT_DIR / name
        rows.append((name, path, path.exists() and path.stat().st_size >= MIN_BYTES))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what is present, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    print(f"[*] {OUT_DIR}")

    if args.check:
        missing = 0
        for name, path, ok in status():
            size = human(path.stat().st_size) if path.exists() else "-"
            print(f"    [{'ok' if ok else '  '}] {name:<30} {size}")
            missing += 0 if ok else 1
        if missing:
            print(f"\n[!] {missing} missing. OCR will fall back to a CDN on first use.")
            print("    Run `python tools/fetch_ocr_assets.py` to keep it offline.")
        else:
            print("\n[ok] OCR assets present; scanning works with the network off.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, url in ASSETS.items():
        path = OUT_DIR / name
        if path.exists() and path.stat().st_size >= MIN_BYTES and not args.force:
            print(f"    [skip] {name:<30} {human(path.stat().st_size)}")
            total += path.stat().st_size
            continue
        print(f"    [get ] {name:<30} {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"[FAIL] could not fetch {name}: {exc}") from exc
        if len(data) < MIN_BYTES:
            raise SystemExit(
                f"[FAIL] {name} came back as only {len(data)} bytes, which is an "
                "error page rather than the asset. Refusing to write it."
            )
        path.write_bytes(data)
        total += len(data)
        print(f"           -> {human(len(data))}")

    print(f"\n[+] {len(ASSETS)} assets, {human(total)} total, in {OUT_DIR}")
    print("    OCR now runs with no network. PDFs with a text layer never needed it.")


if __name__ == "__main__":
    main()
