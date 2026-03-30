"""
JSON to YOLO Dataset Converter
- 5 classes : 형광등, 캔류, 비닐, 유리병, 고철류
- 조건 1 : 해상도 1920×1080 (16:9) 이미지만 사용
           → 실시간 감지 카메라 입력과 종횡비 통일
- 조건 2 : 1건(세션) 당 유효 이미지 5장 이상인 세션만 사용
           → 원거리/근거리/4방향 등 다각도가 모두 갖춰진 건만 학습에 활용
- 조건 3 : PIL 무결성 검사 통과한 이미지만 사용
           → 훼손(truncated/corrupt) 이미지를 사전 차단해 gradient 오염 방지
- 클래스당 2000장, 총 10,000장 목표 (세션 단위 랜덤 추출)
- Output:
    data/images/  ← hard link (같은 드라이브, 추가 용량 0)
    data/labels/  ← YOLO .txt 어노테이션
    data/dataset.yaml
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "[ERROR] Pillow 가 설치되어 있지 않습니다.\n"
        "  pip install pillow  을 실행한 뒤 다시 시도하세요."
    )

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent / "data" / "org"
LABEL_DIR  = BASE_DIR / "Training_라벨링데이터"
OUTPUT_DIR = Path(__file__).parent / "data"   # images/, labels/ 바로 생성

SAMPLES_PER_CLASS    = 2000
MIN_IMAGES_PER_CASE  = 5          # 1건(세션)당 최소 유효 이미지 수
ALLOWED_RESOLUTION   = "1920*1080"
RANDOM_SEED          = 42

CLASS_MAP = {
    "고철류": 0,
    "비닐":   1,
    "유리병": 2,
    "캔류":   3,
    "형광등": 4,
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def find_image(json_path: Path, label_root: Path, image_root: Path) -> Path | None:
    """
    JSON 라벨 경로 → 원천 이미지 경로 매핑
    label : Training_라벨링데이터/{class}/{subclass}/{session}/{name}.Json
    image : [T원천]{class}_{subclass}_{subclass}/{session}/{name}.jpg
    """
    rel   = json_path.relative_to(label_root)
    parts = rel.parts                           # (class, subclass, session, filename)
    if len(parts) != 4:
        return None
    cls, subclass, session, fname = parts
    stem       = Path(fname).stem
    img_folder = f"[T원천]{cls}_{subclass}_{subclass}"
    img_path   = image_root / img_folder / session / (stem + ".jpg")
    return img_path if img_path.exists() else None


def parse_resolution(res_str: str) -> tuple[int, int]:
    """'1920*1080' → (1920, 1080)  |  실패 시 (0, 0)"""
    try:
        w, h = res_str.strip().split("*")
        return int(w), int(h)
    except Exception:
        return 0, 0


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2) pixel → YOLO (cx,cy,w,h) normalized [0,1]"""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def is_image_valid(img_path: Path) -> bool:
    """
    PIL로 이미지를 완전히 디코딩해 무결성 검사.
    truncated / corrupt 파일은 False 반환.

    LOAD_TRUNCATED_IMAGES 를 False(기본값)로 유지하여
    불완전한 파일을 엄격하게 걸러냄.
    """
    try:
        with Image.open(img_path) as img:
            img.load()   # 픽셀 데이터 전체 강제 디코딩
        return True
    except Exception:
        return False


def hard_link_or_symlink(src: Path, dst: Path) -> bool:
    """
    Hard link 우선 (같은 드라이브 → 추가 용량 0),
    실패 시 symlink, 둘 다 실패하면 False
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return True
    except OSError:
        pass
    try:
        os.symlink(src.resolve(), dst)
        return True
    except OSError:
        return False


# ──────────────────────────────────────────────
# Step 1: Collect valid sessions per class
# ──────────────────────────────────────────────
ValidEntry = tuple[Path, Path, list[str]]  # (json_path, img_path, yolo_lines)

def collect_valid_sessions(cls: str, cls_id: int) -> dict[tuple, list[ValidEntry]]:
    """
    클래스 디렉토리를 순회하며 아래 3가지 조건을 모두 만족하는
    항목을 세션 단위로 그룹핑하여 반환.
      조건 1. 해상도 == ALLOWED_RESOLUTION
      조건 2. 매칭 이미지 존재 + 바운딩 박스 유효
      조건 3. PIL 무결성 검사 통과 (훼손 이미지 제외)
    """
    cls_label_dir = LABEL_DIR / cls
    if not cls_label_dir.exists():
        print(f"  [WARN] 라벨 디렉토리 없음: {cls_label_dir}")
        return {}

    sessions: dict[tuple, list[ValidEntry]] = defaultdict(list)
    skipped_corrupt = 0

    for json_path in cls_label_dir.rglob("*.[Jj]son"):
        rel   = json_path.relative_to(LABEL_DIR)
        parts = rel.parts
        if len(parts) != 4:
            continue
        _, subclass, session, _ = parts
        session_key = (subclass, session)

        # ── 매칭 이미지 확인 ──────────────────
        img_path = find_image(json_path, LABEL_DIR, BASE_DIR)
        if img_path is None:
            continue

        # ── JSON 파싱 ─────────────────────────
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # ── 조건 1: 해상도 필터 ───────────────
        if data.get("RESOLUTION", "").strip() != ALLOWED_RESOLUTION:
            continue
        img_w, img_h = parse_resolution(data["RESOLUTION"])
        if img_w == 0:
            continue

        # ── 조건 3: PIL 무결성 검사 ───────────
        if not is_image_valid(img_path):
            skipped_corrupt += 1
            continue

        # ── 바운딩 박스 변환 ──────────────────
        bounding_list = data.get("Bounding", [])
        if not bounding_list:
            continue

        yolo_lines = []
        for bbox in bounding_list:
            try:
                x1 = int(bbox["x1"])
                y1 = int(bbox["y1"])
                x2 = int(bbox["x2"])
                y2 = int(bbox["y2"])
            except (KeyError, ValueError):
                continue
            bbox_cls_name = bbox.get("CLASS", cls)
            bbox_cls_id   = CLASS_MAP.get(bbox_cls_name, cls_id)
            cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            yolo_lines.append(
                f"{bbox_cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

        if not yolo_lines:
            continue

        sessions[session_key].append((json_path, img_path, yolo_lines))

    if skipped_corrupt:
        print(f"  [필터] 훼손(corrupt) 이미지 제외: {skipped_corrupt}장")

    return sessions


# ──────────────────────────────────────────────
# Step 2: Filter sessions (≥ MIN_IMAGES_PER_CASE)
# ──────────────────────────────────────────────
def filter_sessions(
    sessions: dict[tuple, list[ValidEntry]]
) -> list[list[ValidEntry]]:
    """
    1건당 유효 이미지가 MIN_IMAGES_PER_CASE 이상인 세션만 반환.
    반환값: 세션 리스트 (각 원소가 해당 세션의 ValidEntry 리스트)
    """
    return [
        entries
        for entries in sessions.values()
        if len(entries) >= MIN_IMAGES_PER_CASE
    ]


# ──────────────────────────────────────────────
# Step 3: Sample sessions → 2000 images per class
# ──────────────────────────────────────────────
def sample_entries(qualified_sessions: list[list[ValidEntry]]) -> list[ValidEntry]:
    """
    세션을 랜덤 셔플 후 순서대로 추가하여 SAMPLES_PER_CLASS 장 수집.
    세션 전체를 포함하므로 다각도 세트가 항상 완전히 유지됨.
    """
    random.shuffle(qualified_sessions)
    selected: list[ValidEntry] = []

    for session_entries in qualified_sessions:
        if len(selected) >= SAMPLES_PER_CLASS:
            break
        remaining = SAMPLES_PER_CLASS - len(selected)
        selected.extend(session_entries[:remaining])

    return selected


# ──────────────────────────────────────────────
# Step 4: Write YOLO files
# ──────────────────────────────────────────────
def write_yolo_files(
    entries: list[ValidEntry],
    cls: str,
    images_out: Path,
    labels_out: Path,
) -> tuple[int, int]:
    """이미지 hard link + 라벨 .txt 저장. (converted, skipped) 반환"""
    converted = 0
    skipped   = 0

    for _, img_path, yolo_lines in entries:
        out_name = f"{cls}_{img_path.stem}"

        dst_img = images_out / (out_name + ".jpg")
        if not hard_link_or_symlink(img_path, dst_img):
            skipped += 1
            continue

        dst_lbl = labels_out / (out_name + ".txt")
        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")
        converted += 1

    return converted, skipped


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def run():
    images_out = OUTPUT_DIR / "images"
    labels_out = OUTPUT_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    total_converted = 0

    for cls, cls_id in CLASS_MAP.items():
        print(f"\n[{cls}] 세션 수집 중... (PIL 무결성 검사 포함, 시간 소요)")

        # 1) 유효 항목 수집 (3가지 조건 필터링)
        sessions = collect_valid_sessions(cls, cls_id)
        total_sessions  = len(sessions)
        total_valid_img = sum(len(v) for v in sessions.values())

        # 2) 1건당 ≥5장 세션만
        qualified = filter_sessions(sessions)
        q_sessions = len(qualified)
        q_images   = sum(len(v) for v in qualified)

        print(
            f"  전체세션={total_sessions:,} / 유효이미지={total_valid_img:,}  →  "
            f"5장이상세션={q_sessions:,} / 사용가능={q_images:,}장"
        )

        if q_sessions == 0:
            print(f"  [WARN] 조건을 만족하는 세션 없음 → 스킵")
            continue

        # 3) 세션 단위 랜덤 샘플링
        entries = sample_entries(qualified)

        # 4) YOLO 파일 저장
        converted, skipped = write_yolo_files(entries, cls, images_out, labels_out)

        status = "✓" if converted >= SAMPLES_PER_CLASS else f"⚠ {converted}장 (목표 미달)"
        print(
            f"  [{status}] 변환={converted}장, 링크실패={skipped}장"
            + (f"  ← 가용 이미지 {q_images}장뿐" if converted < SAMPLES_PER_CLASS else "")
        )
        total_converted += converted

    # dataset.yaml 저장
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"""\
# YOLO Dataset — Waste Classification
# 조건 1: 해상도={ALLOWED_RESOLUTION}
# 조건 2: 1건당 유효이미지 ≥{MIN_IMAGES_PER_CASE}장 세션만
# 조건 3: PIL 무결성 검사 통과 이미지만 (훼손 파일 제외)
# 총 샘플 수: {total_converted}

path: {OUTPUT_DIR.as_posix()}
train: images
val:   images   # 학습 후 train/val 분리 권장

nc: {len(CLASS_MAP)}
names:
  0: 고철류
  1: 비닐
  2: 유리병
  3: 캔류
  4: 형광등
""",
        encoding="utf-8",
    )

    print(f"\n{'='*55}")
    print(f"  총계   : {total_converted}장")
    print(f"  images → {images_out}")
    print(f"  labels → {labels_out}")
    print(f"  yaml   → {yaml_path}")
    print(f"{'='*55}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  JSON → YOLO Converter")
    print(f"  조건 1 해상도     : {ALLOWED_RESOLUTION}")
    print(f"  조건 2 최소이미지  : {MIN_IMAGES_PER_CASE}장 이상 세션만")
    print(f"  조건 3 무결성검사  : PIL img.load() 통과 이미지만")
    print(f"  클래스당 목표      : {SAMPLES_PER_CLASS}장")
    print("=" * 55)
    print(f"  라벨 루트    : {LABEL_DIR}")
    print(f"  이미지 루트  : {BASE_DIR}")
    print(f"  출력         : {OUTPUT_DIR / 'images'}  /  {OUTPUT_DIR / 'labels'}")
    print()
    run()
    print("\n완료!")
