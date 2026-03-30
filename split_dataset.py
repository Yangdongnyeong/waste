"""
split_dataset.py
────────────────────────────────────────────────────────────────
원본 데이터
  data/images_resize/  ← 1024×576 리사이즈 이미지 (10,000장)
  data/labels/         ← YOLO .txt 어노테이션 (10,000개)

파일명 규칙
  {class}_{session}_{idx}.jpg   예) 캔류_22_X001_C013_1020_0.jpg

분할 전략  ← 세션(1건) 단위로 분할 → 데이터 누수 방지
  ┌─────────────────────────────────────────────────────────┐
  │ 동일 세션(같은 물체의 다각도 사진)이 train/valid/test   │
  │ 중 하나에만 속하게 하여 train↔valid 교차 오염을 차단    │
  └─────────────────────────────────────────────────────────┘

출력 구조
  data/
  └── dataset/
      ├── train/
      │   ├── images/
      │   └── labels/
      ├── valid/
      │   ├── images/
      │   └── labels/
      ├── test/
      │   ├── images/
      │   └── labels/
      └── data.yaml

분할 비율 : train 70% / valid 15% / test 15%
────────────────────────────────────────────────────────────────
"""

import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
IMAGES_SRC  = DATA_DIR / "images_resize"
LABELS_SRC  = DATA_DIR / "labels"
DATASET_DIR = DATA_DIR / "dataset"          # ← 최종 출력 루트
YAML_OUT    = DATASET_DIR / "data.yaml"

SPLITS = {
    "train": 0.70,
    "valid": 0.15,
    "test":  0.15,
}
RANDOM_SEED = 42

CLASS_NAMES = ["고철류", "비닐", "유리병", "캔류", "형광등"]

# ──────────────────────────────────────────────
# 세션 키 추출
#   "캔류_22_X001_C013_1020_0"  →  "캔류_22_X001_C013_1020"
#   마지막 _숫자 를 제거
# ──────────────────────────────────────────────
_SESSION_RE = re.compile(r"^(.+)_\d+$")

def session_key(stem: str) -> str:
    m = _SESSION_RE.match(stem)
    return m.group(1) if m else stem


def hard_link_or_copy(src: Path, dst: Path) -> None:
    """Hard link 우선(추가 용량 0), 실패 시 copy2 로 폴백."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ──────────────────────────────────────────────
# 1. 이미지·라벨 쌍 수집 & 세션 단위 그룹핑
# ──────────────────────────────────────────────
def collect_pairs() -> dict[str, list[tuple[Path, Path]]]:
    """
    images_resize 의 .jpg 파일마다 매칭 .txt 라벨을 찾아
    세션 키 → [(img_path, lbl_path), ...] 딕셔너리로 반환.
    라벨이 없는 이미지는 제외.
    """
    sessions: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    missing = 0

    for img_path in sorted(IMAGES_SRC.glob("*.jpg")):
        lbl_path = LABELS_SRC / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing += 1
            continue
        key = session_key(img_path.stem)
        sessions[key].append((img_path, lbl_path))

    if missing:
        print(f"  [WARN] 라벨 없는 이미지 {missing}장 제외")
    return sessions


# ──────────────────────────────────────────────
# 2. 세션 단위 분할 (클래스별 균형 유지)
# ──────────────────────────────────────────────
def split_sessions(
    sessions: dict[str, list[tuple[Path, Path]]]
) -> dict[str, list[tuple[Path, Path]]]:
    """
    세션을 클래스별로 나눠 각각 train/valid/test 로 분할한 뒤
    합산 반환 → 클래스 간 분포 균형 유지.
    """
    result: dict[str, list[tuple[Path, Path]]] = {
        "train": [], "valid": [], "test": []
    }

    # 클래스별로 세션 키 분리
    class_sessions: dict[str, list[str]] = defaultdict(list)
    for key in sessions:
        cls = key.split("_")[0]   # 파일명 첫 번째 토큰 = 클래스명
        class_sessions[cls].append(key)

    for cls, keys in class_sessions.items():
        random.shuffle(keys)
        n    = len(keys)
        n_tr = round(n * SPLITS["train"])
        n_va = round(n * SPLITS["valid"])
        # test = 나머지 (반올림 오차 흡수)
        splits_keys = {
            "train": keys[:n_tr],
            "valid": keys[n_tr : n_tr + n_va],
            "test":  keys[n_tr + n_va :],
        }
        for split, skeys in splits_keys.items():
            for k in skeys:
                result[split].extend(sessions[k])

        n_te = len(splits_keys["test"])
        print(
            f"  [{cls}]  세션 {n:4d}개 →  "
            f"train {len(splits_keys['train']):3d} / "
            f"valid {n_va:3d} / "
            f"test  {n_te:3d}"
        )

    return result


# ──────────────────────────────────────────────
# 3. 파일 복사/링크 → data/dataset/{split}/
# ──────────────────────────────────────────────
def write_split(
    split_name: str,
    pairs: list[tuple[Path, Path]],
) -> None:
    img_dir = DATASET_DIR / split_name / "images"
    lbl_dir = DATASET_DIR / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_src, lbl_src in pairs:
        hard_link_or_copy(img_src, img_dir / img_src.name)
        hard_link_or_copy(lbl_src, lbl_dir / lbl_src.name)


# ──────────────────────────────────────────────
# 4. data.yaml 저장 → data/dataset/data.yaml
# ──────────────────────────────────────────────
def write_yaml(counts: dict[str, int]) -> None:
    content = f"""\
# YOLO Dataset — Waste Classification (split dataset)
# 분할 비율 : train {SPLITS['train']*100:.0f}% / valid {SPLITS['valid']*100:.0f}% / test {SPLITS['test']*100:.0f}%
# 분할 단위 : 세션(1건) 단위 — 동일 물체 train↔valid 교차 방지
# train {counts['train']:,}장 / valid {counts['valid']:,}장 / test {counts['test']:,}장

path: {DATASET_DIR.as_posix()}
train: train/images
val:   valid/images
test:  test/images

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"

    YAML_OUT.write_text(content, encoding="utf-8")
    print(f"  data.yaml 저장 → {YAML_OUT}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  split_dataset.py — 세션 단위 train/valid/test 분할")
    print(f"  이미지 소스 : {IMAGES_SRC}")
    print(f"  라벨  소스  : {LABELS_SRC}")
    print(f"  출력  루트  : {DATASET_DIR}")
    print(f"  분할 비율   : train {SPLITS['train']*100:.0f}% /"
          f" valid {SPLITS['valid']*100:.0f}% /"
          f" test  {SPLITS['test']*100:.0f}%")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    # 1) 수집
    print("\n[1/4] 이미지·라벨 쌍 수집 중...")
    sessions = collect_pairs()
    total_imgs = sum(len(v) for v in sessions.values())
    print(f"  총 세션 {len(sessions):,}개 / 이미지 {total_imgs:,}장")

    # 2) 분할
    print("\n[2/4] 세션 단위 분할 중...")
    split_data = split_sessions(sessions)

    # 3) 파일 쓰기
    print("\n[3/4] 파일 링크/복사 중...")
    counts: dict[str, int] = {}
    for split_name, pairs in split_data.items():
        write_split(split_name, pairs)
        counts[split_name] = len(pairs)
        print(f"  {split_name:5s} : {len(pairs):5,}장  →  "
              f"{DATASET_DIR / split_name / 'images'}")

    # 4) yaml
    print("\n[4/4] data.yaml 저장 중...")
    write_yaml(counts)

    # 결과 요약
    total = sum(counts.values())
    print("\n" + "=" * 60)
    print(f"  완료!")
    print(f"  train : {counts['train']:,}장  ({counts['train']/total*100:.1f}%)")
    print(f"  valid : {counts['valid']:,}장  ({counts['valid']/total*100:.1f}%)")
    print(f"  test  : {counts['test']:,}장  ({counts['test']/total*100:.1f}%)")
    print(f"  합계  : {total:,}장")
    print("=" * 60)


if __name__ == "__main__":
    main()
