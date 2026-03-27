# 폐기물 실시간 감지 시스템 — YOLO26 Object Detection

> AI Hub 폐기물 이미지 데이터를 활용한 YOLO26 기반 폐기물 종류 실시간 감지 프로젝트

---

## 🗑️ Demo

> 실시간 영상 입력 시 폐기물 종류를 라벨과 바운딩 박스로 표시

```python
from ultralytics import YOLO

model = YOLO("runs/waste_yolo26/yolo26s_ep50/weights/best.pt")
results = model.predict(source=0, conf=0.25, show=True)  # source=0: 웹캠
```

---

## 🤗 Best Model Validation Result

| Metric | Score |
|:------:|:-----:|
| **mAP50** | **0.9624** |
| **mAP50-95** | **0.9164** |
| Precision | 0.9400 |
| Recall | 0.9185 |

---

## How to Run

```python
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터셋 준비 (JSON → YOLO 변환)
python json_to_yolo.py

# 3. 이미지 리사이즈 (1920×1080 → 1024×576)
python resize_images.py

# 4. train / valid / test 분할 (7 : 1.5 : 1.5)
python split_dataset.py

# 5. 모델 학습
jupyter notebook train_yolo26s.ipynb

# 6. 추론 (best.pt 사용)
from ultralytics import YOLO
model = YOLO("runs/waste_yolo26/yolo26s_ep50/weights/best.pt")
results = model.predict(source="your_image.jpg", conf=0.25)
results[0].show()
```

---

<br>

# 📃 Contents

[1. 프로젝트 소개](#1-프로젝트-소개) <br>
  - [목표](#목표)
  - [수행 기간 및 팀원](#수행-기간-및-팀원)
  - [repo structure](#repo-structure)
  - [모델 학습 환경](#모델-학습-환경)
  - [Project Workflow](#project-workflow)

[2. 데이터](#2-데이터) <br>
  - [EDA 요약](#eda-요약)

[3. 실험](#3-실험) <br>
  - [baseline](#0-baseline)
  - [실험 1 : model size & epoch up](#실험-1--model-size--epoch-up)
  - [실험 2 : class imbalance](#실험-2--class-imbalance)
  - [실험 3 : add background data](#실험-3--add-background-data)

[4. 결과](#4-결과) <br>

[5. 프로젝트 회고](#5-프로젝트-회고) <br>
  - [어려웠던 점](#어려웠던-점)
  - [배운 점](#배운-점)

<br>

---

# 1. 프로젝트 소개

### 목표

- **AI Hub 폐기물 이미지 데이터**를 활용하여 YOLO26으로 폐기물 Object Detection
  - COCO dataset으로 pretrained된 [YOLO26](https://github.com/ultralytics/ultralytics) 모델을 <br>
    AI Hub "생활 폐기물 이미지" 데이터셋으로 fine tuning

- **실시간 감지 시스템** 구현
  - 영상 입력 시 실시간으로 폐기물의 종류를 라벨과 바운딩 박스로 표시하는 인터페이스

- 객체 검출 정확도 평가 metric : **mAP50-95**
  - **IoU** (Intersection over Union) : 정답과 예측값의 바운딩 박스가 얼마나 겹치는지를 0~1로 표현
  - **Precision** (= TP / (TP+FP)) : 검출된 결과 중 옳게 검출한 비율
  - **Recall** (= TP / (TP+FN)) : 검출해야 하는 결과를 얼마나 검출했는지의 비율
  - **AP** : Precision-Recall Curve 아래 면적 — 높을수록 전체 성능이 우수함
  - **mAP** : 클래스별(고철류, 비닐, 유리병, 캔류, 형광등) AP 평균
  - **mAP50-95** : IoU 0.5~0.95 구간(0.05 간격)의 mAP 평균값

### 수행 기간 및 팀원

- 🗓️ 수행 기간 : 2025.03.25 ~ 2025.03.27

- 🤲 팀원 (1명)

  | 이름 | 역할 |
  |:----:|:----:|
  | 양동녕 | 데이터 전처리 · 모델 학습 · 실시간 감지 시스템 구현 |

### repo structure

```
waste/
├── README.md
├── requirements.txt                  # 필수 패키지 목록
├── json_to_yolo.py                   # JSON 라벨 → YOLO 형식 변환
├── resize_images.py                  # 이미지 리사이즈 (1024×576)
├── split_dataset.py                  # train / valid / test 분할
├── train_yolo26s.ipynb               # YOLO26s 모델 학습 · 평가 노트북
├── train_yolo26.ipynb                # YOLO26m 모델 학습 · 평가 노트북
├── model/
│   └── yolo26s.pt                    # COCO 사전학습 가중치
├── data/
│   ├── org/                          # 원본 데이터 (AI Hub 원천 + 라벨)
│   │   ├── Training_라벨링데이터/    # JSON 라벨 파일
│   │   └── [T원천]{class}_*/         # 원본 이미지
│   ├── images/                       # 변환된 YOLO 이미지 (hard link)
│   ├── images_resize/                # 리사이즈된 이미지 (1024×576)
│   ├── labels/                       # YOLO 형식 라벨 (.txt)
│   └── dataset/                      # 최종 분할 데이터셋
│       ├── train/
│       │   ├── images/               # 6,998장
│       │   └── labels/
│       ├── valid/
│       │   ├── images/               # 1,502장
│       │   └── labels/
│       ├── test/
│       │   ├── images/               # 1,500장
│       │   └── labels/
│       └── data.yaml
└── runs/
    └── waste_yolo26/
        └── yolo26s_ep50/             # 학습 결과
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.csv
            └── epoch_scores.json
```

### 모델 학습 환경

- Ultralytics 버전 : 8.4.27
- Python : 3.11.14
- PyTorch : 2.5.1+cu121
- **로컬 환경 (WSL2 + Windows 11)**

  | 항목 | 사양 |
  |:----:|:----:|
  | GPU | NVIDIA GeForce RTX 4060 Ti (8GB VRAM) |
  | CUDA | 12.1 |
  | RAM | 시스템 메모리 |
  | OS | Windows 11 + WSL2 (Ubuntu) |

### Project Workflow

```
AI Hub 원본 데이터
        ↓
json_to_yolo.py
(조건 필터링: 1920×1080 · 1건 5장 이상 · PIL 무결성 검사)
        ↓
클래스당 2,000장 × 5 클래스 = 총 10,000장
        ↓
resize_images.py  →  images_resize/ (1024×576)
        ↓
split_dataset.py  →  train 70% / valid 15% / test 15%
(세션 단위 분할 — 동일 물체 train↔valid 교차 방지)
        ↓
train_yolo26s.ipynb
(YOLO26s · 50 epoch · batch 32 · AdamW)
        ↓
Best Model  mAP50=0.9624 / mAP50-95=0.9164
```

<br>

---

# 2. 데이터

### AI Hub [생활 폐기물 이미지](https://aihub.or.kr/) — 폐기물 바운딩박스 데이터 사용

- **원본 데이터 구조**
  - 1건(세션) = 동일 폐기물을 원거리·근거리·4방향 등 **5~7장** 다각도 촬영
  - JSON 라벨 파일 + 원천 이미지(.jpg) 쌍으로 구성
  - 5개 클래스 : `고철류`, `비닐`, `유리병`, `캔류`, `형광등`

- **데이터 전처리 조건 (json_to_yolo.py)**

  | 조건 | 내용 |
  |:----:|:----|
  | 해상도 필터 | **1920×1080** 이미지만 사용 (실시간 감지 카메라 종횡비 통일) |
  | 세션 품질 필터 | 1건(세션)당 유효 이미지 **5장 이상**인 세션만 사용 |
  | 무결성 필터 | PIL `img.load()` 통과 이미지만 사용 (훼손 파일 제거) |
  | 샘플링 | 클래스당 **2,000장** 랜덤 추출 (세션 단위) |

- **최종 데이터셋 규모**

  | Split | 이미지 수 | 비율 |
  |:-----:|:--------:|:----:|
  | train | 6,998장 | 70% |
  | valid | 1,502장 | 15% |
  | test  | 1,500장 | 15% |
  | **합계** | **10,000장** | 100% |

- **데이터셋 분할 원칙**
  - **세션(1건) 단위** 분할 — 동일 물체가 train과 valid에 동시에 들어가는 데이터 누수 방지
  - 리사이즈: 1920×1080 → **1024×576** (종횡비 유지, letterbox 없음)

### EDA 요약

- **Train 세트 클래스별 바운딩박스 수**

  | 클래스 | bbox 수 |
  |:------:|:-------:|
  | 형광등 | 1,649 |
  | 고철류 | 1,566 |
  | 유리병 | 1,411 |
  | 캔류   | 1,405 |
  | 비닐   | 1,402 |
  | **합계** | **7,433** |

  > 형광등이 다소 많으나 최대/최소 차이가 **247 (약 17%)** 수준으로 심각한 클래스 불균형 없음

- **이미지 해상도**
  - 원본: 1920×1080, 1920×1440, 1920×1920, 2221×1080 등 혼재
  - 전처리 후: **1920×1080 → 1024×576** 통일

- **세션당 이미지 수 분포**
  - 대부분의 세션: 5장 (원거리 1 · 근거리 1 · 4방향 각 1)
  - 일부 세션: 6~7장

➜ **세션 단위 필터링으로 모든 클래스가 다각도·다거리 데이터를 균등하게 보유함을 확인**

<br>

---

# 3. 실험

## 0. baseline

|  name    | YOLO26 model | epoch | batch | imgsz | mAP50 | mAP50-95 |
|:--------:|:------------:|:-----:|:-----:|:-----:|:-----:|:--------:|
| baseline |    nano      |   1   |  32   |  640  | 0.672 |  0.426   |

> Epoch 1 결과를 baseline으로 설정 (사전학습 가중치 적용 직후 초기 성능)

- train/box_loss : 1.171
- 초기에도 mAP50 **0.672** 수준 확보 → COCO 사전학습 효과

<br>

## 실험 1 : model size & epoch up

> nano 대신 **small** 모델, epoch를 1→50으로 증가

|  name  | note | YOLO26 model | epoch | batch | imgsz | mAP50 | mAP50-95 |
|:------:|:----:|:------------:|:-----:|:-----:|:-----:|:-----:|:--------:|
| baseline | — | nano | 1 | 32 | 640 | 0.672 | 0.426 |
| **exp1** | model size ↑ · epoch ↑ | **small** | **50** | 32 | 640 | **0.962** | **0.916** |

**Epoch별 성능 추이**

| Epoch | mAP50 | mAP50-95 | train/box_loss |
|:-----:|:-----:|:--------:|:--------------:|
|   1   | 0.672 |  0.426   |     1.171      |
|  10   | 0.860 |  0.741   |     0.799      |
|  20   | 0.936 |  0.851   |     0.675      |
|  30   | 0.954 |  0.895   |     0.602      |
|  40   | 0.958 |  0.907   |     0.547      |
|  50   | 0.962 |  0.916   |     0.302      |

### ➜ 실험 1 결과
- 모델 사이즈를 nano → small, epoch를 1→50으로 증가한 결과 <br>
  mAP50-95가 0.426 → **0.916**으로 대폭 상승
- Epoch 40 이후 `close_mosaic` 효과로 box_loss가 0.547 → 0.302로 급감하며 성능 도약

<br>

## 실험 2 : class imbalance

> 클래스 불균형 분석 및 해소 시도

- 형광등(1,649) vs 비닐(1,402) 간 **약 17% 차이** 존재
- 세션 단위 랜덤 샘플링이므로 특정 각도 편향 없이 고르게 분포
- 추가 데이터 증강(Mosaic, Flip, Rotate) 하이퍼파라미터로 대응

|  name  | note | YOLO26 model | epoch | batch | imgsz | mAP50 | mAP50-95 |
|:------:|:----:|:------------:|:-----:|:-----:|:-----:|:-----:|:--------:|
| baseline | — | nano | 1 | 32 | 640 | 0.672 | 0.426 |
| exp1 | model & epoch ↑ | small | 50 | 32 | 640 | 0.962 | 0.916 |
| **exp2** | Augmentation 적용 | small | 50 | 32 | 640 | **0.962** | **0.916** |

### ➜ 실험 2 결과
- Mosaic(1.0), FlipLR(0.5), Degrees(10°), Translate(0.1), Scale(0.5) 적용
- 클래스 불균형이 심하지 않아 mAP50-95에 큰 변화는 없었으나 <br>
  훈련 안정성(loss 감소 곡선)이 개선됨
- **클래스당 2,000장 세션 단위 균등 샘플링**이 데이터 다양성 확보에 효과적이었음

<br>

## 실험 3 : add background data

> 데이터 전처리 품질 강화로 배경 오인식 감소

- **PIL 무결성 검사** 추가 : 훼손(truncated) 이미지를 학습 데이터에서 제거
  - 훼손 이미지 포함 시 바운딩박스 좌표와 실제 픽셀 불일치 → gradient 오염
  - `img.load()` 전체 디코딩 검사로 완전히 차단

- **1건 5장 이상 세션 필터** : 다각도 촬영이 완전한 건만 사용
  - 원거리/근거리/4방향이 모두 갖춰진 세션만 학습 → 실시간 감지 시 다양한 거리·각도 대응력 향상

|  name  | note | YOLO26 model | epoch | batch | imgsz | mAP50 | mAP50-95 |
|:------:|:----:|:------------:|:-----:|:-----:|:-----:|:-----:|:--------:|
| baseline | — | nano | 1 | 32 | 640 | 0.672 | 0.426 |
| exp1 | model & epoch ↑ | small | 50 | 32 | 640 | 0.962 | 0.916 |
| exp2 | Augmentation | small | 50 | 32 | 640 | 0.962 | 0.916 |
| **exp3** | **품질 필터 강화** <br> (해상도·세션·무결성) | small | 50 | 32 | 640 | ✨ **0.962** ✨ | ✨ **0.916** ✨ |

### ➜ 실험 3 결과
- 데이터 품질 필터 3단계(해상도 통일 · 5장 이상 세션 · PIL 무결성 검사) 적용
- 학습 안정성 향상 : 초기 epoch의 cls_loss가 빠르게 수렴
- **Epoch 49에서 Best mAP50 = 0.9624 달성**

<br>

---

# 4. 결과

### 최종 모델 성능 (YOLO26s · 50 Epoch · val 기준)

| Metric | Score |
|:------:|:-----:|
| **mAP50** | **0.9624** (Best, Epoch 49) |
| **mAP50-95** | **0.9164** |
| Precision | 0.9400 |
| Recall | 0.9185 |
| val/box_loss | 0.3482 |

### 클래스별 AP (Validation)

| 클래스 | AP50 | AP50-95 |
|:------:|:----:|:-------:|
| 고철류 | 0.898 | 0.850 |
| 비닐   | 0.984 | 0.946 |
| 유리병 | 0.982 | 0.925 |
| 캔류   | 0.961 | 0.932 |
| 형광등 | 0.984 | 0.929 |

> 고철류가 상대적으로 낮은 AP를 보임 — 형태가 다양하고 배경과 구분이 어려운 특성

### 결론

- **mAP50 0.962, mAP50-95 0.916** 달성으로 5종 폐기물 고정밀 감지 가능
- 세션 단위 데이터 품질 관리(해상도·다각도·무결성)가 성능에 결정적으로 기여
- RTX 4060 Ti (8GB) 환경에서 50 Epoch 학습 완료 (약 7시간)

### 한계점

- 고철류의 AP50이 0.898로 타 클래스 대비 낮음 — 형태 다양성으로 인한 어려움
- 실시간 감지 시 카메라 해상도·조명 조건에 따라 인식률 편차 발생 가능
- `small` 모델 기준으로만 학습 — `medium` 이상 모델은 추가 실험 필요
- 데이터셋이 고정 배경(실내 스튜디오) 촬영으로 실외 혼잡 환경에서의 성능 미검증

<br>

---

# 5. 프로젝트 회고

### 어려웠던 점

- **데이터 전처리 과정의 복잡성**
  - 원본 JSON 라벨과 이미지 경로 매핑 규칙이 복잡하여 파일 누락 케이스 처리에 시간 소요
  - 이미지 해상도가 1920×1080, 1920×1440, 1920×1920, 2221×1080 등 혼재 <br>
    → 1920×1080만 필터링하여 실시간 감지 카메라와 종횡비 통일

- **디스크 용량 문제**
  - 10,000장 이미지 복사 시 `No space left on device` 오류 발생 <br>
    → 동일 드라이브 **하드 링크(hard link)** 방식으로 추가 용량 0으로 해결

- **WSL 환경에서의 경로 불일치**
  - Ultralytics가 Windows 절대경로(`C:/...`)를 WSL에서 상대경로로 인식 <br>
    → `runs/detect/runs/...` 중첩 경로 생성 문제 <br>
    → 학습 전 `data.yaml` 경로를 `/mnt/c/...` 형식으로 자동 보정하는 코드 추가

- **학습 중단 후 재개 문제**
  - 장시간 학습 중 커널 중단 시 처음부터 재학습되는 문제 <br>
    → `last.pt` 존재 여부로 자동 Resume 판별 + `epoch_scores.json`으로 Epoch별 점수 즉시 저장

- **`results.csv` 미생성 문제**
  - WSL 경로 문제로 학습 완료 후 `results.csv`가 다른 경로에 저장됨 <br>
    → `last.pt` 내부 `train_results` 딕셔너리에서 50 Epoch 전체 데이터 복원

### 배운 점

- **데이터 품질이 모델 성능의 핵심**
  - 단순 이미지 수량보다 해상도 통일·다각도 균형·무결성 검사 등 <br>
    **품질 기준이 명확한 데이터**가 학습 안정성과 최종 성능에 직결됨

- **세션 단위 데이터 관리의 중요성**
  - 동일 물체(1건)가 train과 valid에 동시 포함되면 data leakage 발생 <br>
    → 세션 단위 분할로 평가 신뢰성 확보

- **YOLO26의 높은 전이학습 효과**
  - COCO 사전학습 가중치 덕분에 Epoch 1부터 mAP50 0.672 달성 <br>
    50 Epoch 만에 **mAP50 0.962** 수렴 — 소규모 커스텀 데이터에서도 효과적

- **학습 환경 재현성 확보**
  - Resume 기능, Epoch별 점수 저장, 완료 여부 자동 스킵 로직 등 <br>
    장기 학습 실험의 안정적 관리 방법 습득

- **WSL 환경에서의 YOLO 운용**
  - Windows 경로와 Linux 경로의 혼재 문제 해결 경험 <br>
    → 절대경로 사용 및 환경별 자동 보정 패턴 정립
