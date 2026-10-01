# 아키텍처와 모델 연결

MVP 코드는 아래 구조로 구현했다. CPU 및 실제 Qwen 검증을 마쳤으며, FLUX.1-dev를 포함한 모든 가중치 설치와 사전 검사를 완료했다. 실제 DragFlow 생성과 두 모델의 전체 통합 검증은 별도 진행한다.

## 모듈

| 경로 (`src/advv/` 기준) | 책임 |
| --- | --- |
| `cli.py`, `config.py` | CLI, YAML 경로 해석, 설정 검증 |
| `prepare.py`, `preflight.py` | revision 고정 다운로드, 로컬 자원·입력·환경 확인 |
| `contracts.py`, `ingest.py` | Source/EditRequest/응답 계약, EXIF 정규화, split·중복 검사 |
| `sampler.py` | seed/source/attempt 기반 연결된 area와 operation·점 선택. `random_geometry_v1`(기본)과 저장된 proposal에서 고르는 `object_region_v2`([설계](REGION_SAMPLER.md)) |
| `proposals.py` | v2 영역 proposal의 raw 수집 순서(`collect_raw`: entity → 부위 이름 텍스트 part의 entity 귀속 → 점 part), raw receipt 읽기·쓰기, mask 정리·필터·중복 제거(IoU 중복과 part 포함 중복, text part가 남은 entity의 point part 억제, part 출처 `text`/`point` 기록), `proposals/<source_id>/` 저장 계약과 검증 로더(CPU) |
| `pipeline.py` | 원본 profile 고정, 총 N장 반복, 기록·재시도·재개·판정·격리 |
| `backends/process.py`, `worker.py` | 독립 Python 환경, GPU 노출, JSONL 요청/응답, durable receipt |
| `backends/dragflow.py` | 고정 upstream Dragger 호출과 입력/좌표 변환 |
| `backends/dragflow_speedups.py` | 선택 실행 가속 패치(`generator.speedups`, 기본 전부 꺼짐). upstream 파일 수정 없이 프로세스 안에서 적용·기록·되돌리기 |
| `backends/qwen.py` | 로컬 Qwen의 새 대화별 단일/두 이미지 추론 |
| `backends/region.py`, `sam3.py` | v2 proposal worker: 공식 SAM 3 텍스트(entity, 부위 이름 part)·점(part) prompt를 한 `set_image`로 처리해 raw receipt 기록. processor 임계값은 ADVV 하한보다 0.01 낮게 주고 채택은 ADVV `≥` 필터가 정한다. Grounding DINO + SAM 2.1은 자리만 있고 미구현 오류 |
| `verifiers/parser.py`, `decision.py` | 엄격한 전체 JSON 파싱, 두 판정 AND |
| `storage.py` | atomic write, hash, 단일 writer lock, 소유 파일 삭제 |
| `visualization.py` | CPU 원본 overlay, 나란히 비교, 좌표·영역 metadata |
| `reporting.py` | 검증된 image-only export(사람 검수가 켜진 run은 pass만), 통계, 정적 report 링크 |
| `human_review.py` | run별로 켜고 끄는 선택 단계. Qwen 통과 이미지의 사람 최종 판정(← pass / → fail), 검수 화면 PNG, fail 삭제 |

가중치·CUDA 없는 검사는 `tests/test_*.py`, 실제 Qwen 검사는 `scripts/check_qwen.py`, 요청값 그림은 `scripts/preview_edits.py`, DragFlow 가속 측정은 `scripts/profile_dragflow.py`·`scripts/check_dragflow_equivalence.py`(GPU, 공용 harness `scripts/dragflow_harness.py`)를 사용한다. fake backend는 tests fixture에만 있으며 production export가 거부한다.

## 실행 순서와 GPU

```text
원본 ingest/snapshot → 로컬 Qwen source profile 고정
[object_region_v2일 때] proposals.json이 없는 원본마다: Qwen worker 종료 → SAM 3 proposal worker(첫 선택 GPU, profile subjects·subject별 parts(v4) 전달, entity마다 자기 subject 부위만 질의)
    → raw receipt → CPU build/write proposals.json (기술 오류가 재시도 후에도 남으면 region_proposals_missing 오류)
    → 모든 원본 처리 후 proposal worker 종료
    proposals/<source_id>/ 로드·검증·hash 고정 → 영역 없는 원본은 source_no_region 보류
while 저장된 unique accepted 수 < 전체 목표 N:
    원본을 round-robin 선택
    source별 attempt index로 무작위 계획·mask 생성 및 예약 저장
        (v2에서 유효 기하가 없는 attempt는 sampling_skipped로 state에 남기고 cursor만 전진)
    Qwen worker 종료 → DragFlow load/generate → generation receipt → worker 종료
    실제 입력 metadata와 비교 PNG 저장
    Qwen load → candidate-only physical VQA
    physical == YES이면 original-first/candidate-second semantic VQA
    판정 record 영속화 → 채택/격리/삭제 → 최종 PNG 갱신
export/report 저장 → completed 확정
[human_review.enabled일 때만] advv review: 사람이 accepted를 ←/→로 판정 → fail 삭제 → export(사람 pass만)/report 갱신
```

coordinator는 torch를 import하지 않는다. worker별 Python 실행 파일과 환경을 분리하고 시작 전에 `CUDA_VISIBLE_DEVICES`를 지정한다. 선택 목록 앞 두 GPU는 DragFlow의 논리 cuda:0/1에, 첫 GPU는 Qwen과 SAM 3 proposal worker의 cuda:0에 대응한다. 한 번에 한 종류의 model worker만 띄운다(`LocalBackend`가 종류를 바꿀 때 이전 worker를 종료). 나머지 선택 GPU의 병렬 활용은 후속 최적화다. 각 execution segment에 사용 GPU·Python·시작/종료를 기록한다.

프로파일과 두 VQA는 같은 Qwen checkpoint를 사용하지만 호출마다 새 메시지를 구성한다. 이전 응답을 대화 context에 누적하지 않는다. timeout 후에는 worker를 종료하고 제한된 기술 재시도에 새 worker를 사용한다. OOM·로드 실패는 즉시 partial failed로 종료한다. 연속 3개 후보의 기술 실패도 장애로 중단한다. 정상 NO/UNCERTAIN·중복에는 전체 시도 제한이 없다.

## 모델 revision과 연결

공식 DragFlow commit은 `b3a8fa7136df5131d07f98425c0d189fb6ca74b0`, FireFlow submodule은 `20ab81fb084c06a089bb8cf86726a28036c88d5b`다. 모델 revision은 `configs/upstream.lock.json`, 로컬 snapshot 경로는 `models/weights.lock.json`에 기록한다. `prepare`만 다운로드를 허용하며 worker는 offline/local-only로 실행한다.

`framework/bench_dragflow.py`는 import하지 않는다. 실제 `Dragger(conf, dtype=torch.float32)`, `load_pipeline()`, `dragger(raw_image, instruction, image_name)`를 호출한다. Transformer는 공식 qint8 경로이며 latent 최적화의 gradient를 유지한다. 공식 코드가 import 때 `./framework/config.yaml`을 읽으므로 worker는 upstream cwd를 사용하고, 모든 run 입출력은 절대 경로로 전달한다. upstream tracked 파일은 수정하지 않는다. 모델 경로를 고정하기 위한 ADVV wrapper 코드는 run manifest의 implementation hash에 포함한다.

| ADVV operation | upstream task | 입력 |
| --- | --- | --- |
| relocation | transformation | binary area, start/target centroid |
| deformation | deformation | binary area, start/target displacement; 독립 scale 인자 없음 |
| rotation | rotation | binary area, start/target, 별도 anchor |

원본과 mask, `instruction.json`은 후보별 `backend_input/`에 저장한다. sampler는 feature grid로 반올림했을 때 항등 이동/범위 이탈도 거부한다. 이미지 크기는 공식 코드에서 16의 배수로 낮춰 bicubic resize하며, 실제 시작점은 upstream이 feature 영역의 centroid로 교체한다. 그래서 `source_point`는 두 sampler 모두 mask의 반올림 centroid이고, 이동·회전·항등 검사도 이 점으로 계산한다. `object_region_v2`의 비볼록 mask는 centroid가 mask 밖일 수 있으므로 윤곽 선택용 mask 내부 점 `region_select_point`를 따로 저장해 `instruction.json`의 `centroids[0]`으로 넘긴다(`centroids[1]`은 `target_point`). upstream centroid는 원본 mask를 feature grid로 bilinear(align_corners=False, antialias 없음) 축소한 뒤 `> 0.5`인 영역에서 계산된다. v2 sampler는 이 축소를 재현해 grid 영역이 비거나 그 centroid가 원본 centroid의 grid 점과 한 칸 넘게 다른 얇은 proposal을 제외하므로, 남은 proposal에서만 두 시작점이 grid 한 칸 이내다. v1 계획에는 이 검사를 적용하지 않는다(v1 계획은 변경 전과 같다). upstream rotation 각도는 grid centroid 시작점·grid로 반올림한 목표점·anchor의 `atan2` 차이이므로, v2는 목표점을 실효 각도가 요청각과 같은 부호이고 `rotation_min_executed_degrees` 이상인 grid 칸 중 요청각에 가장 가까운 칸에 두고 허용 오차(`sampler.object_region.rotation_max_error_*`) 밖 각도는 다시 뽑는다(T015, T015-fix). v2의 grid 반올림은 rotation 목표 탐색의 anchor 점까지 upstream과 같은 float32 half-to-even이다(v1은 float64 유지). feature 영역의 bilinear soft mask를 NPY로 저장하고, nonzero 영역을 nearest로 원본 크기에 복원해 표시한다. 선택 영역과 확대 gradient mask를 혼동하지 않는다.

Qwen은 `Qwen3_5ForConditionalGeneration`과 `AutoProcessor`로 실제 이미지 tensor를 전달한다. profile은 원본 한 장, physical은 후보 한 장, semantic은 원본·후보 두 장 순서다. bf16, eval/inference mode, thinking off, greedy decoding을 사용한다. chat template의 비활성화 표기를 확인하고 새 토큰만 decode한다. prompt·ordered image hashes·profile/hint·모델·processor·전처리·decoding 조건이 검증 identity에 포함된다. semantic prompt에는 profile의 `summary`·`must_preserve`·`uncertain`과 hint만 채우며(`subjects`·`parts` 제외), identity의 `prompt_hash`는 채운 뒤의 prompt hash다.

환경은 DragFlow torch 2.5.1 / Transformers 4.48.0 / Diffusers 0.32.2, Qwen torch 2.6.0 / Transformers 5.17.0, SAM 3 torch 2.10.0+cu128(Python 3.12, `.venv-sam3`)으로 분리했다. SAM 3 공식 코드 commit·가중치 revision·파일 SHA256은 `configs/upstream.lock.json`의 `region_proposal.sam3`에 있다. 전체 설치 버전은 `environments/*.freeze.txt`와 run의 `config.json`에 남긴다.

## DragFlow 실행 가속 플래그 (T017)

`generator.speedups`의 다섯 플래그는 **기본 전부 `false`이고, 그때 wrapper는 upstream에 아무것도 덧씌우지 않는다**(공식 실행 경로, CPU 테스트로 고정). 켜면 `load_pipeline()` 직후 ADVV wrapper가 같은 프로세스 안에서 함수·속성을 바꾼다. upstream tracked 파일은 그대로이고 preflight의 dirty 검사도 그대로다. 켠 플래그는 run `config.json`(`load_config`가 다섯 키를 모두 채워 저장), 후보 `generation_info.speedups`(`enabled`, `numerically_equivalent_patches`, `precision_variants`, 실제 `tf32_allowed`), `report.json`의 `generator_speedups`와 `report.md` 첫머리, `exports/images.jsonl` 각 행의 `generator_speedups`(`official_execution_path`, `enabled`, `tf32`; 후보의 `generation_info.speedups`가 있으면 그것, 없으면 run 설정)에 남는다. 이 키가 없는 이전 run config는 전부 꺼짐으로 읽는다. **모두 GPU에서 아직 측정하지 않았다.** 아래 "결과 동일"은 upstream 코드에서 읽은 설계 근거이며, GPU 비결정성 범위 안의 일치는 `check_dragflow_equivalence.py`로 확인해야 한다.

| 플래그 | 하는 일 | 결과 동일 근거 (DragFlow b3a8fa7, `framework/`) | 비용·주의 |
| --- | --- | --- | --- |
| `skip_unused_single_blocks` | drag round의 F_drag forward에서만 single block 38개를 비운 `ModuleList`로 잠시 바꿔 실행. 그 호출의 noise prediction은 `None` | F_drag 호출(`dragger.py:353`)의 noise prediction은 바로 `del`(355). loss는 `target_block_feature_ids_flux`의 DOUBLE-17/18 feature만 쓴다(163-169, 403-422). single block은 double 19개 뒤에 실행된다(`overrider_DiT.py:346-420`). gradient(447-451)·z 갱신(452-467)·로그(438-439, 469)·다음 round 입력(z, instruction 영역 상태) 어디에도 single 출력이 들어가지 않는다. single block은 override되지 않아(`overrider_DiT.py:194, 222`) feature를 남기지 않고, 붙은 KV inject hook은 상태 없이 출력만 덮어쓴다(`hookhub.py:45-82`; `timeidx`는 `countdown_hooks`에서만 바뀜). F_orig 호출(320-326)은 no_grad이고 noise prediction을 쓰므로 그대로 둔다 | feature id가 DOUBLE-*가 아니거나 `use_optimizer: true`면 설치 거부 |
| `disable_gradient_checkpointing` | `load_pipeline`이 켠 checkpointing(`dragger.py:52`)을 끔 | checkpoint 분기와 일반 분기가 같은 block을 같은 입력으로 호출한다(`overrider_DiT.py:347-402, 421-472`). `use_reentrant=False` checkpoint는 backward에서 같은 forward를 다시 계산할 뿐이다. no_grad forward(inversion·sampling·F_orig)는 원래 checkpoint를 쓰지 않는다 | 재계산 1회 절약, activation 메모리 증가. single 생략 없이 끄면 single activation도 F_drag 동안 남는다 |
| `ip_adapter_on_transformer_device` | IP-adapter attention processor 57개를 cuda:1에서 transformer의 cuda:0으로 옮기고, 같은 코드에 장치 기본값만 바꾼 하위 클래스로 교체 | processor 기본 인자가 `device_1="cuda:1"`로 고정되어 있고 호출부가 기본값에 의존한다(`adapter/attn_processor.py:33-34, 46-50, 142-162`; 배치 `pipeline_flux.py:152-160`). 같은 연산을 다른 GPU(같은 기종)에서 할 뿐이며 장치 간 `.to()`는 값을 바꾸지 않는다. 함수 이름·인자 이름이 같아 diffusers의 signature 기반 kwargs 필터(`attention_processor.py:577-586`)도 같다 | cuda:0 약 +5.7 GB(fp32 약 1.43B param, 추정). image_proj·SigLIP·DINOv2는 cuda:1에 남아 GPU 두 장은 계속 필요 |
| `light_reclaim_memory` | `reclaim_memory`(4개 모듈의 전역 이름)를 현재 장치만 마지막 GPU로 두는 함수로 교체 | 원본은 호출마다 모든 GPU에 `set_device`·`synchronize`·`empty_cache`·`ipc_collect` 후 `gc.collect`(`dashboard_utils.py:66-85`). 어느 것도 값을 바꾸지 않는다. 남는 부작용인 현재 장치(마지막 GPU)는 유지한다 | 캐시 반환·GC가 늦어 reserved/피크 메모리가 오를 수 있음 |
| `tf32` | `torch.backends.cuda.matmul.allow_tf32`·`cudnn.allow_tf32`를 `True`로 | **결과가 달라지는 정밀도 변형.** tensor dtype은 fp32 그대로이고 matmul(quanto dequant 뒤 `torch.matmul`, attention 포함)이 TF32 tensor core를 쓴다. PyTorch 기본값은 matmul off·cuDNN on이라 공식 경로도 VAE conv는 이미 TF32다 | `precision_variants`로 기록. 품질은 전체 이미지 비교로 따로 판단 |

측정 도구(GPU 필요, 이번 작업에서는 실행하지 않음): `scripts/profile_dragflow.py`는 한 플래그 조합으로 짧은 inversion과 N개(기본 5) drag round를 실행해 no_grad/grad forward, backward, `reclaim_memory`, Python GC 시간과 `torch.profiler` kernel 분류(장치 간 복사 포함), 피크 메모리를 JSON으로 저장한다. `scripts/check_dragflow_equivalence.py`는 모델을 한 번 올려 기준(전부 꺼짐), 반복 기준(GPU 잡음 바닥), 누적 arm을 같은 seed·입력으로 N round 실행하고 round별 L1 loss, 갱신된 latent, inversion latent의 차이와 round 시간·피크 메모리를 비교한다. arm 상태는 `completed`·`oom`·`error`(설치 실패·비-OOM 예외 포함, 예외 문자열 기록)로 나누고, 완료된 arm만 비교한다. 반복 기준(`control_baseline`)은 판정 대상이 아니라 잡음 바닥이다: 기준과의 round별 최대 상대차(loss, latent·inversion latent)를 `noise_floor`로 기록하고, 결과 동일 arm의 허용오차를 지표별 `max(rtol, k × noise_floor)`(기본 rtol 1e-5, k 10)로 정한다. 반복 기준이 없거나 완료되지 않으면 rtol만 쓴다. `tf32` arm은 별도 진단 기준(loss 1e-2, latent 5e-2)만 기록하고 종료 코드에 쓰지 않는다. 종료 코드는 0(전부 완료·통과), 2(완료된 결과 동일 arm이 허용오차 밖), 3(수치 차이는 없고 OOM·오류 arm이 있음, `incomplete_arms`), 4(기준 미완료), 1(인자·입력 오류)이며, argparse 오류도 2가 아닌 1로 끝낸다. `equivalence.json`은 arm마다 다시 써서(`finished`, `pending_arms`) 중단·오류에도 그때까지 결과가 남는다. harness의 round는 drag K-loop의 operation 한 번(`dragger.py:332-334`)이고, `max_dragging_num`(공식 50)부터 INTENSIFY가 `conf["lr"]`을 바꾸므로(455-456) `--rounds`는 그 이하만 받는다. arm 사이에는 upstream `conf`를 원래 값으로 되돌린다. 두 스크립트 모두 GPU 번호를 인자로만 받고(기본값 없음), `CUDA_VISIBLE_DEVICES`가 명령에 같은 값으로 미리 지정되어 있어야 하며(스크립트가 설정하지 않음), 기존 run의 `candidates/<id>/backend_input`을 읽기만 한다. 결과는 입력 검증 뒤에 `_workspace/` 아래 새 디렉터리(프로젝트 밖은 `--allow-external-out`, 원본 run 안은 거부)에 만든다. bf16 계산·quanto 제거·round 수 축소처럼 DragFlow recipe를 바꾸는 가속은 이 플래그에 포함하지 않았다([조사 요약](REFERENCES.md#dragflow-가속-조사-요약-t016-2026-10-01)).

## 저장·복구

```text
runs/<run_id>/
├── config.json / manifest.json / sources.json / state.json
├── inputs/images/<source_id>.png
├── profiles/<source_id>.json
├── profiles/<source_id>_receipts/
├── proposals/<source_id>/         # object_region_v2만: proposals.json + entity/part mask PNG
│   └── receipt/                   # worker raw.json, raw_masks.npz, response.json
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
advv prepare --only sam3   # object_region_v2용 gated SAM 3 (선택)
advv preflight --config configs/advv.local.yaml --input-dir assets --target-count 10 --gpus 0,1
advv run 10 --config configs/advv.local.yaml --input-dir assets --gpus 0,1 --run-id pilot_001
advv run --run-dir runs/pilot_001 --resume --gpus 2,3
advv export --run-dir runs/pilot_001
advv report --run-dir runs/pilot_001
advv review --run-dir runs/pilot_001
```

숫자는 실행 예시다. `run N`과 기존 `--target-count N`은 동일한 전체 채택 수량으로 정규화하고, 양의 정수 여부·중복 지정·재개 시 수량 입력을 parser에서 검사한다. 수량을 생략한 새 실행은 YAML 설정을 사용한다. preflight는 가중치를 다운로드하거나 추론하지 않는다. CLI 경로는 현재 디렉터리, YAML 경로는 YAML 파일 위치, manifest 이미지 경로는 dataset root 기준이다. `--input-dir`은 manifest 입력을 해제한다. Python 실행 파일의 venv symlink는 resolve하지 않아 독립 환경이 유지된다.

외부 replay_plan import, sidecar 설명, 독립 GPU 상주/후보 병렬화, supervised export는 후속 확장이다. 저장된 run의 미완료 계획 재개는 현재 구현에 포함된다.
