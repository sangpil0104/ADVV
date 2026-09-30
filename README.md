# ADVV

**Augmentation using DragFlow with VQA Verification**

원본에서 영역·방향·operation을 무작위로 선택해 DragFlow로 편집하고, 로컬 Qwen3.5-4B가 **현실성·물리적 개연성**과 **원본의 핵심 의미 보존**을 모두 YES로 판정한 새 이미지를 수집합니다. N은 입력 전체의 총 증강 수량이며, 정상적인 NO/UNCERTAIN·중복 반복에는 시도·시간 상한이 없습니다. 원본과 동일한 이미지도 수량에서 제외합니다.

## 구현 및 검증 상태

- Python 패키지, CLI, 무작위 sampler, 엄격한 JSON 판정, 총량 반복, 중단·재개, 격리·삭제, export/report, 편집 설정 시각화를 구현했습니다.
- 공식 DragFlow `Dragger` adapter와 별도 프로세스의 로컬 Qwen adapter를 연결했습니다. 가짜 생성기로 대체하는 실행 옵션은 없습니다.
- CPU 테스트 **79개**와 실제 Qwen의 원본 분석·단일 이미지 VQA·두 이미지 비교를 검증했습니다.
- **실제 DragFlow 생성 → 두 Qwen 검증 → 채택·저장·export까지 N=1 실행을 완료했습니다.** 고양이 원본 한 장으로 확인한 통합 사례이며, 산업 결함·사고·희귀 생물에 대한 성능 결과는 아닙니다. [검증 범위와 측정값](docs/RUN_VERIFICATION.md)을 참고하세요.
- FLUX.1-dev와 Qwen/InstantCharacter/SigLIP/DINOv2/CLIP 가중치는 별도 다운로드합니다. [구현 상태와 남은 검증](docs/IMPLEMENTATION_PLAN.md)에서 실제 모델의 중단·재개 검증과 도메인 평가를 구분합니다.

Qwen 판단은 시각적 추정입니다. 산업 결함 판정 정확도, 물리적 사실, 생물 종명이나 후속 모델 성능 개선을 입증한 상태는 아닙니다. Qwen 추가 학습과 GT 라벨은 실행의 필수 조건이 아닙니다.

## 실행하기

처음 사용하는 서버에서는 아래 [새 환경 설치 및 검사](#새-환경-설치-및-검사)를 먼저 진행하세요. 이후 저장소 루트에서 다음 명령으로 모델을 준비합니다. `hf auth login`이 안내하는 방법으로 인증하고, [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 사용 조건에 동의한 계정을 사용하세요. 토큰을 소스·설정에 적지 않습니다.

```bash
conda activate ADVV
hf auth login
advv prepare
```

`prepare`는 [고정 revision](configs/upstream.lock.json)의 모델만 다운로드합니다. 모두 준비되면 `models/weights.lock.json`과 `configs/advv.local.yaml`을 연결합니다. 기존 local 설정은 덮어쓰지 않습니다. 추론 worker는 오프라인 모드이며 실행 중 모델을 다운로드하지 않습니다.

원본 사진을 `assets/`에 넣으세요. 입력 사진·실험 결과·가중치·개인 설정·발표자료는 저장소에 포함하지 않습니다. 아래 N=10, GPU 0–7과 run ID는 예시이며 **전체 총량 N**과 사용 가능한 장치로 변경하세요.

```bash
advv preflight --config configs/advv.local.yaml --input-dir assets --target-count 10 --gpus 0,1,2,3,4,5,6,7
advv run 10 --config configs/advv.local.yaml --input-dir assets --gpus 0,1,2,3,4,5,6,7 --run-id pilot_001
```

`advv run N`의 숫자를 바꾸면 됩니다. 예를 들어 `advv run 50 ...`은 전체 입력에서 **두 VQA를 통과한 고유 이미지 50장**을 모읍니다. 탈락·중복 후보는 수량에서 제외하므로 실제 생성 시도는 N보다 많을 수 있습니다. 기존 `--target-count N` 형식도 지원하지만 두 형식을 동시에 지정하면 오류입니다. N은 양의 정수이며, 재개할 때는 숫자를 생략하고 저장된 목표를 사용합니다. 새 실행마다 다른 `--run-id`를 지정하세요.

MVP는 후보 한 장씩 순차 처리합니다. 선택한 목록의 앞 두 GPU를 DragFlow에, 첫 GPU를 Qwen에 교대로 노출합니다. 예를 들어 `--gpus 4,5,6,7`이면 생성은 4·5, 검증은 4를 사용합니다. 8장을 지정해도 8장 동시 병렬 실행은 하지 않습니다. 공식 DragFlow 경로에는 GPU 두 장이 필요합니다.

`Ctrl+C` 또는 SIGTERM으로 중단하면 partial 결과를 저장합니다. 이미 완료한 판정과 삭제한 NO 이미지를 다시 추론하지 않습니다. GPU ID만 바꿔 재개할 수 있습니다.

```bash
advv run --run-dir runs/pilot_001 --resume --gpus 0,1,2,3,4,5,6,7
advv report --run-dir runs/pilot_001
advv export --run-dir runs/pilot_001
```

원본·prompt·sampler·모델·환경·코드를 바꾸면 새 run이 필요합니다. 기존 run ID는 덮어쓰지 않습니다. 모델 로드 실패/OOM은 즉시 중단하고, JSON 오류·개별 timeout은 최대 1회 추가 시도합니다. 기술 오류가 연속 3개 후보에서 발생하면 장애로 종료합니다. 이는 정상적인 NO/UNCERTAIN의 시도 상한이 아닙니다.

## 결과와 편집 설정 보기

```text
runs/<run_id>/
├── config.json / manifest.json / sources.json / state.json
├── inputs/images/                 # EXIF 정규화한 원본 snapshot
├── profiles/                      # 생성 전 고정한 의미 기준과 응답 원문
├── edit_regions/                  # 선택한 binary area; GT mask가 아님
├── records/                       # 계획·실제 입력·두 VQA·오류·hash
├── candidates/                    # backend 입력, 실제 좌표/영역, raw receipt
├── accepted/                      # 두 YES + 중복 제외 + 저장 완료 이미지
├── quarantine/                    # UNCERTAIN·오류·중복·보관 설정의 NO
├── visualizations/<candidate_id>/
│   ├── drag_plan.png              # 원본 + start/end/area
│   ├── comparison.png             # 왼쪽 설정, 오른쪽 표시 없는 생성 결과
│   └── metadata.json              # 요청값/실제 적용값과 좌표 변환
├── exports/images.jsonl           # run 기준 상대 경로, image-only 계보 manifest
└── report.json / report.md / logs/
```

시작점은 빨간 원 S, 도착점은 초록 십자 E, 영역은 반투명 파랑, 방향은 노란 화살표, 회전 anchor는 보라색 A입니다. 실제 backend의 중심점 교체·좌표 반올림·리사이즈도 기록합니다. 화살표는 요청한 이동이지 실제 도달 위치의 측정값이 아닙니다. 원본과 overlay를 Qwen 입력이나 증강 데이터에 섞지 않습니다.

NO 후보는 먼저 판정·hash를 기록한 뒤 생성 이미지와 비교 PNG를 삭제합니다. 원본 설정 PNG는 남습니다. report에는 존재하는 파일만 링크합니다. 시각화 실패 후 재개하면 추가 생성 없이 그림만 복구합니다.

GPU 없이 **생성 전 요청 설정**을 미리 보려면 사진을 `assets/`에 넣고 새로운 출력 폴더를 지정합니다. 미리보기는 실제 생성 결과가 아닙니다.

```bash
python scripts/preview_edits.py --input-dir assets --output integration_checks/new_previews
```

## 입력과 선택 설명

폴더 입력은 PNG/JPEG/WebP/BMP/TIFF를 재귀적으로 읽고 EXIF 방향을 정규화합니다. 다중 프레임 이미지는 거부합니다. 기본 split은 `source`, group은 원본별 임시 ID이며 촬영 개체의 실제 독립성을 보장하지 않습니다.

원본마다 “차량 충돌 사고”, “특정 균열” 등의 설명을 추가하려면 [source manifest 예시](examples/source_manifest.jsonl)의 `preserve_hint`를 채우고 설정의 `dataset.root`·`manifest`를 지정하세요. 이 경우 `input_dir: null`로 두고 CLI의 `--input-dir`을 생략합니다. YAML 경로는 YAML 파일 위치, manifest 이미지 경로는 dataset root 기준입니다. `val/test`는 증강하지 않으며 group·정규화 픽셀의 split 중복은 오류입니다. 현재 설명 입력은 JSONL manifest로 지원합니다.

학습에는 `accepted/` 또는 임시 폴더를 직접 순회하기보다 `exports/images.jsonl`을 사용하세요. 검출 box·분할 GT·분류 label을 자동 승계하지 않는 image-only export입니다. 테스트용 fake backend는 production export가 차단됩니다.

## 새 환경 설치 및 검사

Python 3.10을 사용합니다. DragFlow와 Qwen은 Transformers 요구 버전이 달라 별도 환경을 유지합니다.

```bash
git clone https://github.com/sangpil0104/ADVV.git
cd ADVV
conda create -n ADVV python=3.10 pip -y
conda activate ADVV
pip install -e '.[dev,download]'
python -m venv .venv-dragflow
python -m venv .venv-qwen
.venv-dragflow/bin/python -m pip install -r environments/dragflow-requirements.txt
.venv-qwen/bin/python -m pip install -r environments/qwen-requirements.txt
.venv-dragflow/bin/python -m pip install --no-deps -e .
.venv-qwen/bin/python -m pip install --no-deps -e .
```

공식 checkout을 준비하지 않은 새 서버에서는 다음을 실행합니다. 기존 checkout이 있으면 덮어쓰지 말고 commit을 확인합니다.

```bash
git clone https://github.com/Edennnnnnnnnn/DragFlow.git third_party/DragFlow
git -C third_party/DragFlow checkout b3a8fa7136df5131d07f98425c0d189fb6ca74b0
git -C third_party/DragFlow submodule update --init --recursive
```

실제 설치 버전은 `environments/*.freeze.txt`에 있습니다. freeze의 editable 설치 경로는 검증 서버의 기록이며, 재설치는 위의 requirements 명령을 사용합니다. 현재 검증 환경은 DragFlow torch 2.5.1 / Transformers 4.48.0 / Diffusers 0.32.2, Qwen torch 2.6.0 / Transformers 5.17.0입니다. Qwen은 bf16, thinking off, greedy decoding을 사용합니다. 최적화 kernel 추가 설치 없이 기본 PyTorch 경로로 동작을 확인했습니다.

```bash
python -m pytest -q
python -m ruff check src scripts tests
python scripts/check_qwen.py --image assets/original_image.png --gpus 0,1 --output integration_checks/qwen_probe/results.json
```

마지막 명령만 실제 GPU/Qwen을 사용하며, 모델 준비와 `assets/original_image.png`가 필요합니다. 이 검사는 생성 데이터나 학습 export를 만들지 않습니다. N=1 실제 증강 실행의 측정값과 남은 검증은 [검증 기록](docs/RUN_VERIFICATION.md)에 정리했습니다.

## 설계 문서

[AGENTS.md](AGENTS.md) · [SPEC.md](SPEC.md) · [구조](docs/ARCHITECTURE.md) · [구현 상태](docs/IMPLEMENTATION_PLAN.md) · [시각화](docs/VISUALIZATION.md) · [평가 계획](docs/EVALUATION.md) · [공식 근거](docs/REFERENCES.md) · [설정 예시](configs/advv.example.yaml)
