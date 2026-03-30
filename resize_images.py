"""
resize_images.py
────────────────
data/images/ 의 모든 .jpg 이미지를 1024×576 으로 리사이즈하여
data/images_resize/ 에 저장합니다.

- 원본 종횡비 유지
- 비율이 다를 경우 검정 패딩(letterbox)으로 채움
- 원본 파일·폴더 구조는 그대로 보존
- 멀티스레드(기본 CPU 코어 수)로 병렬 처리하여 속도 향상
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image, ImageFile
except ImportError:
    raise SystemExit(
        "[ERROR] Pillow 가 설치되어 있지 않습니다.\n"
        "  pip install pillow  을 실행한 뒤 다시 시도하세요."
    )

# 불완전하게 저장된(truncated) 이미지도 읽을 수 있는 부분까지 처리
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent / "data"
SRC_DIR     = BASE_DIR / "images"
DST_DIR     = BASE_DIR / "images_resize"
TARGET_W    = 1024
TARGET_H    = 576
WORKERS     = os.cpu_count() or 4   # 병렬 스레드 수

# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
def resize_letterbox(src_path: Path, dst_path: Path) -> str:
    """
    이미지를 TARGET_W × TARGET_H 로 리사이즈.
    - 종횡비 유지
    - 비율이 다를 경우 검정 패딩 추가 (letterbox)
    반환값: 'ok' | 'skip' | 'error:<msg>'
    """
    if dst_path.exists():
        return "skip"

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            # 비율 계산
            scale = min(TARGET_W / orig_w, TARGET_H / orig_h)
            new_w = round(orig_w * scale)
            new_h = round(orig_h * scale)

            # 고품질 리사이즈
            resized = img.resize((new_w, new_h), Image.LANCZOS)

            # letterbox 캔버스 (검정)
            canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
            offset_x = (TARGET_W - new_w) // 2
            offset_y = (TARGET_H - new_h) // 2
            canvas.paste(resized, (offset_x, offset_y))

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(dst_path, "JPEG", quality=95)
        return "ok"

    except Exception as e:
        return f"error:{e}"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"[ERROR] 원본 폴더가 없습니다: {SRC_DIR}")

    # 변환 대상 목록 수집
    src_files = sorted(SRC_DIR.rglob("*.jpg"))
    total = len(src_files)
    if total == 0:
        print("[INFO] 변환할 이미지가 없습니다.")
        return

    print("=" * 55)
    print(f"  resize_images  {TARGET_W}×{TARGET_H}  (letterbox)")
    print("=" * 55)
    print(f"  원본 경로  : {SRC_DIR}")
    print(f"  저장 경로  : {DST_DIR}")
    print(f"  대상 파일  : {total:,}장")
    print(f"  병렬 스레드: {WORKERS}")
    print()

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 원본 폴더 구조를 유지하면서 목적지 경로 매핑
    tasks = []
    for src in src_files:
        rel     = src.relative_to(SRC_DIR)
        dst     = DST_DIR / rel
        tasks.append((src, dst))

    # 병렬 처리
    done = ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(resize_letterbox, s, d): s for s, d in tasks}
        for future in as_completed(futures):
            result = future.result()
            done  += 1
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                err += 1
                print(f"  [ERR] {futures[future].name} → {result}")

            # 진행률 출력 (500장마다)
            if done % 500 == 0 or done == total:
                pct = done / total * 100
                print(f"  [{done:,}/{total:,}] {pct:.1f}%  "
                      f"(완료={ok}, 스킵={skip}, 오류={err})")

    print()
    print("=" * 55)
    print(f"  완료  : {ok:,}장 저장")
    print(f"  스킵  : {skip:,}장 (이미 존재)")
    print(f"  오류  : {err:,}장")
    print(f"  출력  : {DST_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
