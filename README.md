# ADVV

**Augmentation using DragFlow with VQA Verification**

원본에서 영역·방향·operation을 무작위로 선택해 DragFlow로 편집하고, 로컬 Qwen3.5-4B가 **현실성·물리적 개연성**과 **원본의 핵심 의미 보존**을 모두 YES로 판정한 새 이미지를 수집합니다. N은 입력 전체의 총 증강 수량이며, 정상적인 NO/UNCERTAIN·중복 반복에는 시도·시간 상한이 없습니다. 원본과 동일한 이미지도 수량에서 제외합니다.

## 구현 및 검증 상태

- Python 패키지, CLI, 무작위 sampler, 엄격한 JSON 판정, 총량 반복, 중단·재개, 격리·삭제, export/report, 편집 설정 시각화를 구현했습니다.
- 공식 DragFlow `Dragger` adapter와 별도 프로세스의 로컬 Qwen adapter를 연결했습니다. 가짜 생성기로 대체하는 실행 옵션은 없습니다.
- CPU 테스트 **273개**와 실제 Qwen의 원본 분석·단일 이미지 VQA·두 이미지 비교를 검증했습니다.
- **실제 DragFlow 생성 → 두 Qwen 검증 → 채택·저장·export까지 N=1 실행을 완료했습니다.** 고양이 원본 한 장으로 확인한 통합 사례이며, 산업 결함·사고·희귀 생물에 대한 성능 결과는 아닙니다. [검증 범위와 측정값](docs/RUN_VERIFICATION.md)을 참고하세요.
- FLUX.1-dev와 Qwen/InstantCharacter/SigLIP/DINOv2/CLIP 가중치는 별도 다운로드합니다. [구현 상태와 남은 검증](docs/IMPLEMENTATION_PLAN.md)에서 실제 모델의 중단·재개 검증과 도메인 평가를 구분합니다.
- 선택 단계로 켤 수 있는 사람 최종 검수 `advv review`를 추가했습니다. Qwen 두 검증 뒤 사람이 ←(통과)/→(불합격)로 판정합니다. 기본은 꺼짐입니다. 불합격은 삭제하고 재생성하지 않으며 export에는 사람 통과분만 들어갑니다. 실제 터미널·생성 이미지 검증은 아직입니다.
- **준비만, 미측정:** DragFlow 실행 가속 플래그 `generator.speedups`(기본 전부 꺼짐 = 공식 경로)와 측정 스크립트를 추가했습니다. CPU 테스트만 했고 GPU에서 속도·결과 동일성·메모리는 아직 확인하지 않았습니다([아래](#dragflow-가속-플래그-선택-미측정)).
- **부분 구현:** 무작위 사각형·타원 대신 객체(entity)·부위(part) mask를 편집 영역으로 쓰는 `object_region_v2` sampler를 추가했습니다([설계](docs/REGION_SAMPLER.md)). 기본은 기존 `random_geometry_v1`입니다. v2 run은 원본 profile을 고정한 뒤 SAM 3 worker(별도 `.venv-sam3`, 첫 선택 GPU)로 원본마다 한 번 proposal을 만들어 고정하고, 그 안에서 편집 영역을 고릅니다. 부위(part)는 profile이 subject마다 적은 부위 이름(v4 `parts`, 예: car → wheel, excavator → bucket)의 텍스트 prompt와 entity 내부 점 prompt를 함께 써서 얻고 출처를 기록합니다. 각 객체에는 자기 subject의 부위 이름만 묻습니다. rotation은 DragFlow feature grid에서 실제로 실행될 각도가 요청각의 허용 오차(기본 max(3°, 30%)) 안에 들도록 목표 칸을 고릅니다(CPU 테스트만). 텍스트 부위가 남은 객체에서는 점 부위를 쓰지 않습니다(설정으로 끌 수 있음). sampler, proposal 저장 계약·검증, `source_no_region` 보류, worker 프로세스 계약·receipt·오류 경로는 가짜 SAM 3로 CPU 테스트했습니다. SAM 3 모델 자체는 GPU 스모크(텍스트·점 prompt, 피크 약 6.2 GiB)로 확인했고, 부위 이름 추가 전 ADVV adapter를 5장에 실행해 점 prompt만으로는 부위가 거의 나오지 않음을 확인했고, 부위 이름 텍스트 prompt를 더한 adapter도 같은 5장에서 실행했습니다(부위 4–7개/장). 이후 바꾼 부위 필터(포함 중복, 면적 하한 0.02, 점 부위 억제)는 그 receipt를 CPU에서 다시 build해 확인했습니다. 텍스트 부위의 객체 포함 비율 하한은 0.85입니다(굴착기 트랙이 0.865/0.888로 빠져 0.90에서 낮춤, 이 값으로 다시 build한 결과는 아직 확인 전). 실험·디버깅용으로 `sampler.object_region.region_phrase_filter`(예: `{level: part, phrases_contain: [leg]}`)를 주면 phrase가 맞는 영역만 고르고, 맞는 영역이 없으면 그 원본을 보류합니다(일반 사용에는 필요 없음, [설계 §3.4](docs/REGION_SAMPLER.md#34-실험용-region-phrase-필터-t023), CPU 테스트만). **v4 profile(subject별 `parts`)로의 실행과 v2 증강(생성·판정) 실행은 아직 검증하지 않았습니다.** 비교용 Grounding DINO + SAM 2.1 worker는 미구현입니다.

Qwen 판단은 시각적 추정입니다. 산업 결함 판정 정확도, 물리적 사실, 생물 종명이나 후속 모델 성능 개선을 입증한 상태는 아닙니다. Qwen 추가 학습과 GT 라벨은 실행의 필수 조건이 아닙니다.

## 실행하기

처음 사용하는 서버에서는 아래 [새 환경 설치 및 검사](#새-환경-설치-및-검사)를 먼저 진행하세요. 이후 저장소 루트에서 다음 명령으로 모델을 준비합니다. `hf auth login`이 안내하는 방법으로 인증하고, [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 사용 조건에 동의한 계정을 사용하세요. 토큰을 소스·설정에 적지 않습니다.

```bash
conda activate ADVV
hf auth login
advv prepare
```

`prepare`는 [고정 revision](configs/upstream.lock.json)의 모델만 다운로드합니다. 모두 준비되면 `models/weights.lock.json`과 `configs/advv.local.yaml`을 연결합니다. 기존 local 설정은 덮어쓰지 않습니다. 추론 worker는 오프라인 모드이며 실행 중 모델을 다운로드하지 않습니다. `object_region_v2`에 쓰는 SAM 3는 HF 수동 승인이 필요한 gated 모델이라 기본 `prepare`에 포함하지 않습니다. 승인된 계정으로 `advv prepare --only sam3`를 실행하면 없는 파일만 받고, 이미 있는 파일은 SHA256만 검증합니다.

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

### DragFlow 가속 플래그 (선택, 미측정)

`generator.speedups`는 기본 전부 `false`이며 이때 공식 DragFlow 실행 경로 그대로입니다. 켜면 ADVV wrapper가 upstream 파일을 고치지 않고 프로세스 안에서 패치합니다. 결과 동일을 설계 근거로 둔 패치 4개와 결과가 달라지는 정밀도 변형 `tf32`가 있습니다([근거와 표](docs/ARCHITECTURE.md#dragflow-실행-가속-플래그-t017)). **아직 GPU에서 속도·동일성·메모리를 측정하지 않았습니다.** 켠 플래그는 `config.json`, 후보 `generation_info.speedups`, report, `exports/images.jsonl` 각 행의 `generator_speedups`(`official_execution_path`, `enabled`, `tf32`)에 남고, 켠 run은 공식 경로가 아니라고 표시됩니다.

```yaml
generator:
  speedups:
    skip_unused_single_blocks: true
    disable_gradient_checkpointing: true
    ip_adapter_on_transformer_device: true
    light_reclaim_memory: true
    tf32: false            # true는 정밀도 변형
```

측정용 스크립트는 GPU 두 장을 인자로 받고(기본값 없음) 기존 run의 후보 입력을 읽기만 합니다. 실행 전 GPU 소유자를 확인하세요. `CUDA_VISIBLE_DEVICES`를 명령에 `--gpus`와 같은 값으로 직접 적어야 하며, 없으면 스크립트가 설정하지 않고 종료합니다. 결과는 `_workspace/` 아래 새 디렉터리에만 씁니다(프로젝트 밖은 `--allow-external-out`). `--rounds`는 drag K-loop 반복 수이고 INTENSIFY 전인 `max_dragging_num`(공식 50) 이하만 받습니다.

```bash
CUDA_VISIBLE_DEVICES=A,B .venv-dragflow/bin/python scripts/profile_dragflow.py --gpus A,B \
    --run-dir runs/<run> --candidate <candidate_id> --out _workspace/<new_dir>
CUDA_VISIBLE_DEVICES=A,B .venv-dragflow/bin/python scripts/check_dragflow_equivalence.py --gpus A,B \
    --run-dir runs/<run> --candidate <candidate_id> --out _workspace/<new_dir>
```

`check_dragflow_equivalence.py`는 arm마다 `equivalence.json`을 다시 써서 중간에 멈춰도 그때까지 결과가 남습니다. 반복 기준(`control_baseline`)은 판정하지 않고 GPU 잡음 바닥으로 쓰며, 결과 동일 arm은 지표별로 `max(--rtol 1e-5, --noise-multiplier 10 × 잡음 바닥)` 안이면 통과입니다. 종료 코드: 0 전부 완료·통과, 2 완료된 결과 동일 arm이 허용오차 밖(수치 차이), 3 수치 차이는 없지만 OOM·오류로 끝난 arm이 있음(`incomplete_arms`), 4 기준 arm이 완료되지 않음, 1 인자·입력 오류. `tf32` arm은 진단 기준만 기록하고 종료 코드에 쓰지 않습니다. `profile_dragflow.py`는 arm이 OOM·오류면 JSON을 남기고 3으로 끝납니다.

### 사람 최종 검수 (선택)

기본은 꺼져 있어 위의 Qwen 전용 파이프라인 그대로 동작합니다. 켜려면 새 run을 만들 때 `--human-review`를 붙이거나 설정에 `human_review.enabled: true`를 둡니다. 값은 run마다 고정되며 재개할 때 바꿀 수 없습니다.

```bash
advv run 10 --config configs/advv.local.yaml --input-dir assets --gpus 0,1 --run-id pilot_002 --human-review
advv review --run-dir runs/pilot_002
```

Qwen 단계가 끝나면(`completed`) 두 검증을 통과한 이미지를 사람이 한 장씩 판정합니다.

- VS Code에서 `runs/pilot_001/review/current.png`를 열어 두세요. 판정할 때마다 다음 후보(왼쪽 원본, 오른쪽 생성 결과)로 바뀝니다.
- 터미널에서 **← 통과**, **→ 불합격**, **q 중단**입니다. 중단하면 같은 명령으로 이어서 검수합니다.
- 불합격은 판정 기록을 남긴 뒤 이미지를 삭제하며 **다시 생성하지 않습니다.** 최종 수량은 N보다 적을 수 있고, 부족하면 새 run을 만드세요. 검수를 시작한 run은 `--resume`할 수 없습니다.
- 검수를 켠 run의 `exports/images.jsonl`에는 사람이 통과시킨 이미지만 들어갑니다. 검수가 끝나면 export와 report를 자동으로 갱신합니다.

원본·prompt·sampler·모델·환경·코드를 바꾸면 새 run이 필요합니다. 기존 run ID는 덮어쓰지 않습니다. 모델 로드 실패/OOM은 즉시 중단하고, JSON 오류·개별 timeout은 최대 1회 추가 시도합니다. 기술 오류가 연속 3개 후보에서 발생하면 장애로 종료합니다. 이는 정상적인 NO/UNCERTAIN의 시도 상한이 아닙니다.

## 결과와 편집 설정 보기

```text
runs/<run_id>/
├── config.json / manifest.json / sources.json / state.json
├── inputs/images/                 # EXIF 정규화한 원본 snapshot
├── profiles/                      # 생성 전 고정한 의미 기준과 응답 원문
├── proposals/                     # object_region_v2만: 고정한 entity/part 영역 후보
├── edit_regions/                  # 선택한 binary area; GT mask가 아님
├── records/                       # 계획·실제 입력·두 VQA·오류·hash
├── candidates/                    # backend 입력, 실제 좌표/영역, raw receipt
├── accepted/                      # 두 YES + 중복 제외 + 저장 완료 이미지(사람 불합격은 삭제)
├── quarantine/                    # UNCERTAIN·오류·중복·보관 설정의 NO
├── visualizations/<candidate_id>/
│   ├── drag_plan.png              # 원본 + start/end/area
│   ├── comparison.png             # 왼쪽 설정, 오른쪽 표시 없는 생성 결과
│   └── metadata.json              # 요청값/실제 적용값과 좌표 변환
├── review/current.png             # 사람 검수 화면(원본 | 후보), 검수를 켠 run만
├── exports/images.jsonl           # 검수를 켠 run은 사람 통과분만, run 기준 상대 경로, image-only 계보 manifest
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

학습에는 `accepted/` 또는 임시 폴더를 직접 순회하지 말고, `exports/images.jsonl`을 사용하세요. 사람 검수를 켰다면 검수를 마친 뒤 사용합니다. 검출 box·분할 GT·분류 label을 자동 승계하지 않는 image-only export입니다. 테스트용 fake backend는 production export가 차단됩니다.

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

### SAM 3 proposal 환경 (object_region_v2)

`object_region_v2`를 쓸 때만 필요합니다. SAM 3는 Python ≥3.12·torch ≥2.7을 요구하므로 DragFlow·Qwen과 합치지 않고 `.venv-sam3`를 따로 만듭니다. 서버 드라이버가 580 미만이면 cu130 wheel은 쓸 수 없어 cu128을 씁니다(드라이버 535에서 동작 확인).

```bash
conda create -n advv-sam3-py python=3.12 -y
"$(conda run -n advv-sam3-py which python)" -m venv .venv-sam3
.venv-sam3/bin/python -m pip install --upgrade pip "setuptools<82" wheel
.venv-sam3/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sam3/bin/python -m pip install -r environments/sam3-requirements.txt
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
.venv-sam3/bin/python -m pip install -e third_party/sam3
.venv-sam3/bin/python -m pip install --no-deps -e .
advv prepare --only sam3
```

`environments/sam3-requirements.txt`에는 `numpy<2`, `setuptools<82`(sam3가 `pkg_resources`를 import), 그리고 sam3 commit이 선언하지 않았지만 import하는 `einops`·`pycocotools`·`psutil`이 들어 있습니다. 설정에는 `sampler.version: object_region_v2`, v4 profile prompt/schema(`text_parts: false`이면 v3도 가능), `sampler.object_region.rotation_max_error_degrees`·`rotation_max_error_fraction`·`rotation_min_executed_degrees`, `execution.proposal_python`, `region_proposal.model_path`·`model_revision`·`repo_path`·`code_revision`을 지정합니다([설정 예시](configs/advv.example.yaml), [설계 §5–6](docs/REGION_SAMPLER.md#5-환경과-gpu)).

실제 설치 버전은 `environments/*.freeze.txt`에 있습니다. freeze의 editable 설치 경로는 검증 서버의 기록이며, 재설치는 위의 requirements 명령을 사용합니다. 현재 검증 환경은 DragFlow torch 2.5.1 / Transformers 4.48.0 / Diffusers 0.32.2, Qwen torch 2.6.0 / Transformers 5.17.0, SAM 3 torch 2.10.0+cu128(Python 3.12)입니다. Qwen은 bf16, thinking off, greedy decoding을 사용합니다. 최적화 kernel 추가 설치 없이 기본 PyTorch 경로로 동작을 확인했습니다.

```bash
python -m pytest -q          # GPU 통합 테스트(-m integration)는 기본 실행에서 제외
python -m ruff check src scripts tests
python scripts/check_qwen.py --image assets/original_image.png --gpus 0,1 --output integration_checks/qwen_probe/results.json
```

마지막 명령만 실제 GPU/Qwen을 사용하며, 모델 준비와 `assets/original_image.png`가 필요합니다. 이 검사는 생성 데이터나 학습 export를 만들지 않습니다. N=1 실제 증강 실행의 측정값과 남은 검증은 [검증 기록](docs/RUN_VERIFICATION.md)에 정리했습니다.

## 설계 문서

[AGENTS.md](AGENTS.md) · [SPEC.md](SPEC.md) · [구조](docs/ARCHITECTURE.md) · [구현 상태](docs/IMPLEMENTATION_PLAN.md) · [영역 sampler v2](docs/REGION_SAMPLER.md) · [시각화](docs/VISUALIZATION.md) · [평가 계획](docs/EVALUATION.md) · [공식 근거](docs/REFERENCES.md) · [설정 예시](configs/advv.example.yaml)
