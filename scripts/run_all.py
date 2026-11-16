"""Chạy tuần tự 01 -> 08, dừng ngay khi có bước lỗi.

Cách dùng (từ thư mục gốc repo):
    python scripts/run_all.py
    python scripts/run_all.py --skip 05 06    # bỏ qua các bước chạy lâu
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

logger = logging.getLogger("run_all")

STEPS = [
    "01_preprocess.py",
    "02_eda_overview.py",
    "03_eda_features.py",
    "04_baseline_xgb.py",
    "05_imbalance_compare.py",
    "06_tune_xgb.py",
    "07_evaluate.py",
    "08_explain.py",
]


def parse_args() -> argparse.Namespace:
    valid = [step[:2] for step in STEPS]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="NN",
        help=f"Số đầu tên script cần bỏ qua, ví dụ: --skip 05 06. Hợp lệ: {' '.join(valid)}",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.skip) - set(valid))
    if unknown:
        parser.error(f"--skip có giá trị không hợp lệ {unknown}; hợp lệ: {valid}")
    return args


def print_summary(results: list[tuple[str, str, float | None]]) -> None:
    print("\nBảng thời gian từng bước:")
    for name, status, seconds in results:
        shown = "" if seconds is None else f"{seconds:8.1f} giây"
        print(f"  {name:<26} {status:<14} {shown}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    skip = set(args.skip)
    results: list[tuple[str, str, float | None]] = []

    for step in STEPS:
        if step[:2] in skip:
            logger.info("BỎ QUA %s", step)
            results.append((step, "bỏ qua", None))
            continue

        path = config.SCRIPTS_DIR / step
        if not path.exists() or path.stat().st_size == 0:
            reason = "không tồn tại" if not path.exists() else "đang rỗng (chưa có code)"
            results.append((step, "LỖI", None))
            print_summary(results)
            logger.error("%s %s, dừng. Dùng --skip %s nếu muốn bỏ qua.", step, reason, step[:2])
            sys.exit(1)

        logger.info("=== CHẠY %s ===", step)
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, str(path)], cwd=config.ROOT)
        elapsed = time.perf_counter() - started

        if proc.returncode != 0:
            results.append((step, f"LỖI (mã {proc.returncode})", elapsed))
            print_summary(results)
            logger.error("%s lỗi (mã %d), dừng.", step, proc.returncode)
            sys.exit(proc.returncode)
        results.append((step, "xong", elapsed))

    print_summary(results)
    logger.info("Đã chạy xong toàn bộ pipeline.")


if __name__ == "__main__":
    main()