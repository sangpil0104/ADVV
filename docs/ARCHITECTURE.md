# 아키텍처와 모델 연결

MVP 코드는 아래 구조로 구현했다. CPU 및 실제 Qwen 검증을 마쳤으며, FLUX.1-dev를 포함한 모든 가중치 설치와 사전 검사를 완료했다. 실제 DragFlow 생성과 두 모델의 전체 통합 검증은 별도 진행한다.

## 모듈

| 경로 (`src/advv/` 기준) | 책임 |
| --- | --- |
| `cli.py`, `config.py` | CLI, YAML 경로 해석, 설정 검증 |
| `prepare.py`, `preflight.py` | revision 고정 다운로드, 로컬 자원·입력·환경 확인 |
| `contracts.py`, `ingest.py` | Source/EditRequest/응답 계약, EXIF 정규화, split·중복 검사 |
| `sampler.py` | seed/source/attempt 기반 연결된 area와 operation·점 선택 |
| `pipeline.py` | 원본 profile 고정, 총 N장 반복, 기록·재시도·재개·판정·격리 |
| `backends/process.py`, `worker.py` | 독립 Python 환경, GPU 노출, JSONL 요청/응답, durable receipt |
| `backends/dragflow.py` | 고정 upstream Dragger 호출과 입력/좌표 변환 |
| `backends/qwen.py` | 로컬 Qwen의 새 대화별 단일/두 이미지 추론 |
| `verifiers/parser.py`, `decision.py` | 엄격한 전체 JSON 파싱, 두 판정 AND |
| `storage.py` | atomic write, hash, 단일 writer lock, 소유 파일 삭제 |
| `visualization.py` | CPU 원본 overlay, 나란히 비교, 좌표·영역 metadata |
| `reporting.py` | 검증된 image-only export, 통계, 정적 report 링크 |

가중치·CUDA 없는 검사는 `tests/test_*.py`, 실제 Qwen 검사는 `scripts/check_qwen.py`, 요청값 그림은 `scripts/preview_edits.py`를 사용한다. fake backend는 tests fixture에만 있으며 production export가 거부한다.

## 실행 순서와 GPU

```text
원본 ingest/snapshot → 로컬 Qwen source profile 고정
while 저장된 unique accepted 수 < 전체 목표 N:
    원본을 round-robin 선택
    source별 attempt index로 무작위 계획·mask 생성 및 예약 저장
    Qwen worker 종료 → DragFlow load/generate → generation receipt → worker 종료
    실제 입력 metadata와 비교 PNG 저장
    Qwen load → candidate-only physical VQA
    physical == YES이면 original-first/candidate-second semantic VQA
    판정 record 영속화 → 채택/격리/삭제 → 최종 PNG 갱신
export/report 저장 → completed 확정
```

coordinator는 torch를 import하지 않는다. worker별 Python 실행 파일과 환경을 분리하고 시작 전에 `CUDA_VISIBLE_DEVICES`를 지정한다. 선택 목록 앞 두 GPU는 DragFlow의 논리 cuda:0/1에, 첫 GPU는 Qwen의 cuda:0에 대응한다. 나머지 선택 GPU의 병렬 활용은 후속 최적화다. 각 execution segment에 사용 GPU·Python·시작/종료를 기록한다.

프로파일과 두 VQA는 같은 Qwen checkpoint를 사용하지만 호출마다 새 메시지를 구성한다. 이전 응답을 대화 context에 누적하지 않는다. timeout 후에는 worker를 종료하고 제한된 기술 재시도에 새 worker를 사용한다. OOM·로드 실패는 즉시 partial failed로 종료한다. 연속 3개 후보의 기술 실패도 장애로 중단한다. 정상 NO/UNCERTAIN·중복에는 전체 시도 제한이 없다.

## 모델 revision과 연결

공식 DragFlow commit은 `b3a8fa7136df5131d07f98425c0d189fb6ca74b0`, FireFlow submodule은 `20ab81fb084c06a089bb8cf86726a28036c88d5b`다. 모델 revision은 `configs/upstream.lock.json`, 로컬 snapshot 경로는 `models/weights.lock.json`에 기록한다. `prepare`만 다운로드를 허용하며 worker는 offline/local-only로 실행한다.

`framework/bench_dragflow.py`는 import하지 않는다. 실제 `Dragger(conf, dtype=torch.float32)`, `load_pipeline()`, `dragger(raw_image, instruction, image_name)`를 호출한다. Transformer는 공식 qint8 경로이며 latent 최적화의 gradient를 유지한다. 공식 코드가 import 때 `./framework/config.yaml`을 읽으므로 worker는 upstream cwd를 사용하고, 모든 run 입출력은 절대 경로로 전달한다. upstream tracked 파일은 수정하지 않는다. 모델 경로를 고정하기 위한 ADVV wrapper 코드는 run manifest의 implementation hash에 포함한다.

| ADVV operation | upstream task | 입력 |
| --- | --- | --- |
| relocation | transformation | binary area, start/target centroid |
| deformation | deformation | binary area, start/target displacement; 독립 scale 인자 없음 |
| rotation | rotation | binary area, start/target, 별도 anchor |

원본과 mask, `instruction.json`은 후보별 `backend_input/`에 저장한다. sampler는 feature grid로 반올림했을 때 항등 이동/범위 이탈도 거부한다. 이미지 크기는 공식 코드에서 16의 배수로 낮춰 bicubic resize하며, 실제 시작점은 upstream이 feature 영역의 centroid로 교체한다. feature 영역의 bilinear soft mask를 NPY로 저장하고, nonzero 영역을 nearest로 원본 크기에 복원해 표시한다. 선택 영역과 확대 gradient mask를 혼동하지 않는다.

Qwen은 `Qwen3_5ForConditionalGeneration`과 `AutoProcessor`로 실제 이미지 tensor를 전달한다. profile은 원본 한 장, physical은 후보 한 장, semantic은 원본·후보 두 장 순서다. bf16, eval/inference mode, thinking off, greedy decoding을 사용한다. chat template의 비활성화 표기를 확인하고 새 토큰만 decode한다. prompt·ordered image hashes·profile/hint·모델·processor·전처리·decoding 조건이 검증 identity에 포함된다.

환경은 DragFlow torch 2.5.1 / Transformers 4.48.0 / Diffusers 0.32.2, Qwen torch 2.6.0 / Transformers 5.17.0으로 분리했다. 전체 설치 버전은 `environments/*.freeze.txt`와 run의 `config.json`에 남긴다.

## 저장·복구

```text
runs/<run_id>/
├── config.json / manifest.json / sources.json / state.json
├── inputs/images/<source_id>.png
├── profiles/<source_id>.json
├── profiles/<source_id>_receipts/
├── edit_regions/<candidate_id>.png
├── records/<candidate_id>.json
├── candidates/<candidate_id>/
│   ├── backend_input/             # 원본, binary mask, upstream instruction
│   ├── generation_receipt.json
│   ├── effective.json / effective_region.npy / effective_area.png
│   └── vqa/                       # 호출별 raw response receipt
├── accepted/<candidate_id>.png
├── quarantine/<candidate_id>/generated.png
├── visualizations/<candidate_id>/drag_plan.png / comparison.png / metadata.json
├── exports/images.jsonl
└── report.json / report.md / logs/
```

`config.json`에는 경로를 해석한 설정, prompt/schema 내용, 모델·환경·upstream snapshot이 있다. `manifest.json`은 코드·설정·source manifest hash를, `state.json`은 cursor·count·실행 segment를 가진다. 계획은 record의 `plan`, 실제 입력은 `effective`로 저장한다.

후보 예약을 모델 추론 전에 저장한다. 모델 응답 receipt를 worker가 coordinator 응답 전달 전에 원자적으로 쓰므로, 전달 직후 중단되어도 이미 완료한 호출을 반복하지 않는다. 재개 때 accepted 파일 hash와 unique count를 재계산하고, 미완료 reservation을 먼저 처리한다. completed check와 삭제한 NO는 재추론하지 않는다. 코드를 포함한 recipe가 바뀌면 새 run을 요구하며 GPU ID만 변경할 수 있다.

NO 판정·hash를 저장한 다음 run 소유의 생성 이미지·비교 PNG·해당 atomic 임시 파일만 삭제한다. 원본 설정 그림은 남긴다. symlink·hardlink·원본·accepted 경로 삭제를 거부한다. 실패하면 cleanup_pending으로 기록해 재개한다. 시각화 실패는 별도 오류이며 quota가 채워진 뒤 추가 생성 없이 복구한다. export/report까지 완료하기 전에는 completed 상태를 확정하지 않는다.

## CLI와 경로

```bash
advv prepare
advv preflight --config configs/advv.local.yaml --input-dir assets --target-count 10 --gpus 0,1
advv run 10 --config configs/advv.local.yaml --input-dir assets --gpus 0,1 --run-id pilot_001
advv run --run-dir runs/pilot_001 --resume --gpus 2,3
advv export --run-dir runs/pilot_001
advv report --run-dir runs/pilot_001
```

숫자는 실행 예시다. `run N`과 기존 `--target-count N`은 동일한 전체 채택 수량으로 정규화하고, 양의 정수 여부·중복 지정·재개 시 수량 입력을 parser에서 검사한다. 수량을 생략한 새 실행은 YAML 설정을 사용한다. preflight는 가중치를 다운로드하거나 추론하지 않는다. CLI 경로는 현재 디렉터리, YAML 경로는 YAML 파일 위치, manifest 이미지 경로는 dataset root 기준이다. `--input-dir`은 manifest 입력을 해제한다. Python 실행 파일의 venv symlink는 resolve하지 않아 독립 환경이 유지된다.

외부 replay_plan import, sidecar 설명, 독립 GPU 상주/후보 병렬화, supervised export는 후속 확장이다. 저장된 run의 미완료 계획 재개는 현재 구현에 포함된다.
