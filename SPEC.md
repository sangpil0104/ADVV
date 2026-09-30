# ADVV 구현 명세

- 문서 버전: `0.4`
- 갱신일: `2026-09-30`
- 상태: MVP 코드 구현, CPU 79개·로컬 Qwen·실제 DragFlow N=1 통합 실행 확인; 확대 검증과 도메인 평가 대기
- 이름: **Augmentation using DragFlow with VQA Verification**

## 1. 확정된 목표와 범위

산업 현장의 고장·결함·사고, 멸종 위기종·희귀 생물처럼 확보하기 어려운 실제 세계 이미지를 증강한다. **1차 기여는 현실성과 원본 의미를 유지하는 증강 파이프라인**이다. 후속 모델의 성능 개선 검증은 2차 목표다.

사용자는 원본 이미지 폴더와 **입력 전체에 대한 새 증강 이미지 총 N장**을 지정한다. 영역·방향·변형을 무작위로 선택하고 DragFlow로 후보를 만든다. Qwen의 현실성·물리적 개연성 검증과 원본 의미 보존 검증을 모두 통과한 새로운 이미지가 N장 쌓일 때까지 반복한다. N에 원본 이미지는 포함하지 않는다.

사용자 확정 사항:

| 항목 | 결정 |
| --- | --- |
| 정답 라벨 | 현재 없음; image-only 파이프라인에 필수 아님 |
| 편집 계획 | 자동 무작위 생성; 수동 mask/점 입력 불필요 |
| 편집 설정 시각화 | start_point·end_point·area를 원본에 표시하고 생성 결과와 나란히 PNG 저장 |
| 허용 변화 | 위치·크기·자세·형태 등은 허용하되 핵심 의미·결함 유형·생물 정체성 유지 |
| 보존 기준 | 원본에서 자동 추정; 사용자 설명은 선택 입력 |
| 생성 수량 | 원본별 수량이 아닌 입력 전체에서 총 N장 |
| 총 실행 상한 | 없음; 목표 달성 또는 사용자 중단까지 계속 |
| 검증 모델 | 다운로드한 로컬 `Qwen/Qwen3.5-4B`, 추가 학습 없이 사용 |
| GPU | 사용자가 실행할 때 지정; 고정 번호·자동 전체 점유 금지 |
| 평가 순서 | 파이프라인·이미지 품질 검증 먼저, 후속 학습 실험은 이후 |

정상적인 NO·UNCERTAIN 반복으로 실행을 자동 중단하지 않는다. 처리 불가능한 입력, 가중치 로드 실패, 장치 부적합, 저장소 오류 같은 기술적 실패는 명시적으로 기록하고 중단할 수 있다. 총 실행 상한과 개별 작업 timeout은 다른 개념이다.

MVP에는 폴더 입력, 원본 의미 추출, 무작위 sampler, 실제 두 모델 연결, 두 VQA, 목표 수량 관리, NO 삭제, 사용자 중단·재개, image-only export와 report를 포함한다. Qwen fine-tuning, task-specific annotation 생성, 후속 모델 학습, 웹 UI는 필수 범위가 아니다.

## 2. 기법과 채택 조건

```text
h_i = Profile_Qwen(original_i, optional_user_hint_i)   # 원본마다 한 번 고정
edit_ij = RandomSampler(seed, source_id_i, attempt_index_j)
candidate_ij = DragFlow(original_i, edit_ij)
p_ij = PhysicalVQA_Qwen(candidate_ij)
s_ij = SemanticVQA_Qwen(original_i, candidate_ij, h_i)
accepted_ij = valid(p_ij) AND p_ij.answer == YES
              AND valid(s_ij) AND s_ij.answer == YES
quota_count = count(saved, unique, export-eligible accepted candidates)
stop_successfully_when(quota_count == N)
```

현실성·물리 검증을 먼저 수행하고 YES일 때 의미 보존 검증을 수행하는 순차 방식을 기본으로 한다. 앞 단계에서 탈락하면 다음 검증은 `not_run`으로 기록한다. 미실행을 NO로 바꾸지 않는다.

예: 차량 위치가 바뀌어 충돌 사고가 평범한 주차 장면이 되면 물리적으로 가능해도 의미 검증은 NO다. 결함이 사라지거나 다른 결함으로 바뀌는 것도 NO다. 반대로 위치·자세가 달라도 중요한 사건과 특징이 유지되면 의미 검증을 통과할 수 있다.

Qwen의 판정은 시각적 추정이다. 물리적 사실, 정확한 종 식별, 학습용 라벨의 정답을 증명하지 않는다. 결함이나 손상 자체를 물리적으로 불가능한 것으로 취급하지 않는다.

## 3. 입력과 원본 의미 기준

### 3.1 일반 사용자 입력

실행 수량은 `advv run N ...`의 위치 인자로 받는다. 기존 `advv run --target-count N ...`도 지원한다. 두 형식의 동시 지정, 0·음수·정수가 아닌 수량은 추론 전에 거부한다. CLI 수량을 생략하면 YAML의 `run.target_count`를 사용하며, 최종 수량이 없으면 실행하지 않는다. `--resume`에서는 수량 인자를 받지 않고 저장된 N을 유지한다.

필수 입력은 원본 폴더, 양의 정수 N, 실행할 GPU 선택이다. 모델·prompt·sampler 설정은 설정 파일로 제공한다. 선택적으로 이미지별 보존 의미 설명을 JSONL manifest로 전달할 수 있다. 원본마다 사람이 mask·좌표·GT 라벨을 만들 필요는 없다.

폴더에서 지원 이미지 파일을 안정적인 경로 순서로 읽고 manifest를 작성한다. 지원 확장자·디코딩·EXIF 방향 정규화·RGB 변환을 명시하고, 원본 파일은 수정하지 않는다. 같은 정규화 픽셀의 중복 입력은 중복 source로 집계하여 한 번만 처리한다. 이미지가 없거나 모든 입력을 사용할 수 없으면 원인을 알리고 종료한다.

### 3.2 내부 Source manifest: JSONL

| 필드 | 타입 | 규칙 |
| --- | --- | --- |
| `schema_version` | string | `1.1` |
| `source_id` | string | manifest 내 고유 ID |
| `image_path` | string | dataset root 기준 상대 경로 |
| `split` | enum | `source`, `train`, `val`, `test` |
| `group_id` | string | 같은 개체·시편·촬영 세션 등 독립성 단위 |
| `domain` | string | `industrial_defect`, `accident`, `rare_organism` 등 설명용 |
| `preserve_hint` | string 또는 null | 사용자 선택 설명; 학습 GT 아님 |
| `label` | object 또는 null | MVP는 null 가능 |

라벨 없는 구축 모드는 `source/train`을 처리하며 폴더 입력의 기본 split은 `source`다. val/test는 생성하지 않는다. 후속 학습 단계에서는 group 단위 split을 먼저 확정하고 train만 증강한다. 같은 group·동일 내용이 split 사이에 겹치면 검사 실패다. group 정보가 없으면 source별 임시 ID를 부여하고 실제 독립성까지 확인한 것은 아니라고 기록한다.

### 3.3 SourceProfile

[원본 의미 추출 prompt](prompts/source_profile_v2.txt)와 [schema](schemas/source_profile.schema.json)를 사용한다. `summary`, `must_preserve`, `uncertain`을 원본과 선택 설명에서 생성한다. 원본 전체와 이 기준을 의미 VQA에 함께 넣는다. 모델이 만든 문장만 비교하지 않는다.

추출 원문, 사용자 설명, 원본 hash, 모델·processor revision, prompt hash를 저장하고 **해당 원본의 후보를 생성하기 전에 고정**한다. 후보를 본 뒤 기준을 바꾸어 통과시키지 않는다. 명확한 종명이 보이지 않으면 종명을 발명하지 말고 관찰 가능한 구별 특징을 기록한다. 사용자 설명도 사진에서 확인할 수 없는 사실을 참으로 보장하지 않는다.

`uncertain=true` 또는 유효한 profile을 얻지 못한 source는 `source_uncertain/source_error`로 보류하고 다른 원본을 처리한다. 자동 생성 profile은 GT가 아니다. 모든 source가 보류되면 `no_eligible_sources`로 종료하여 사용자 설명/입력 수정을 요청할 수 있도록 report한다. 원본에 보이는 의미 자체가 미확정인 상태로 YES 수를 채우지 않는다.

## 4. 무작위 편집 계획

sampler는 사용 가능한 원본을 안정적인 round-robin 순서로 선택하고, 각 원본의 seed·attempt index로 새 영역·operation·방향·강도를 표본 추출한다. 총량 모드이므로 원본별 최종 채택 수는 같지 않을 수 있으며 report한다. 수락률이 높은 원본으로 자동 집중하는 학습 정책은 MVP에 없다.

MVP 영역은 이미지 내부의 연결된 기하 영역(타원 또는 사각형)을 무작위 생성한 binary mask다. 객체 segmentation 모델을 필수로 추가하지 않는다. 영역이 의미 있는 객체에 정확히 맞는다는 가정은 하지 않으며, 이 방식의 수락률과 다양성은 평가한다. 후속 객체 중심 proposal은 별도 sampler 버전이다.

`relocation/deformation/rotation` 중 backend에서 검증한 operation을 선택한다. 영역 면적, 이동 거리, 회전/변형 강도 분포는 설정으로 전달한다. 설정 예시의 수치는 시작점이며 도메인에 최적화된 값이 아니다. 의미를 보존하는지의 최종 판정은 VQA에서 수행한다.

새 후보는 항상 **입력 원본**에서 생성한다. 이미 생성한 이미지를 재편집하는 연쇄 증강은 기본 범위가 아니다. 동일한 NO 이미지의 판정을 바꾸려 하지 않고 새 계획을 만든다.

### 4.1 저장되는 EditRequest

| 필드 | 규칙 |
| --- | --- |
| `schema_version` | `1.1` |
| `source_id`, `edit_id`, `attempt_index` | source 참조, 고유 edit ID, 0부터 증가하는 원본별 index |
| `seed`, `sampler_version` | 실행 seed에서 안정적인 digest로 유도한 32-bit seed와 sampler 버전 |
| `operation`, `operation_params` | 검증된 operation과 추가 강도·회전 등 파라미터 |
| `region_mask_path` | 현재 run 기준 상대 경로; 생성된 0/255 PNG |
| `source_point`, `target_point` | 정규화 원본 크기 기준 `[x,y]` |
| `anchor_point` | operation별 필요성을 upstream 변환기가 검사 |
| `source_prompt`, `target_prompt` | 고정 profile과 편집 지시에서 작성; 결함/사고를 정상 장면으로 고치지 않도록 구성 |

좌상단 원점, `0 ≤ x < width`, `0 ≤ y < height`, source point는 mask 내부여야 한다. 빈 mask·항등 변환·범위 밖 좌표는 거부한다. 직사각형이 아닌 이미지에서도 x/y 축을 혼동하지 않는다. seed에 Python의 프로세스별 `hash()`를 사용하지 않는다.

후보 계획과 mask는 생성 전에 원자적으로 저장한다. upstream `instruction.json`과 동일한 schema라고 가정하지 않고 고정 commit에 맞게 변환한다. 리사이즈·패딩은 이미지/mask/좌표에 동일한 기하 변환을 적용하고 기록한다. ADVV의 mask 표시·역변환 보간은 nearest-neighbor다. 고정 upstream 내부의 feature mask는 bilinear 보간을 사용하므로 연속값 native mask와 그 nonzero 영역을 별도로 기록한다. sampler가 반복적으로 유효 계획을 만들 수 없으면 기술 오류로 처리하며 NO로 집계하지 않는다.

### 4.2 경로 규칙

- YAML 상대 경로: YAML 디렉터리 기준.
- source manifest의 이미지·label 경로: dataset root 기준.
- sampler가 저장한 mask와 모든 run artifact 경로: 해당 run 기준.
- 입력은 읽기 전용; `..`·symlink를 통한 허용 root 이탈과 원본 경로에 출력 쓰기를 거부한다.

### 4.3 편집 설정 시각화 — 기본 활성화

각 후보에서 **어디를, 어느 방향으로, 어느 범위에서 편집했는지** 이미지로 확인할 수 있어야 한다. 내부 `source_point`는 화면의 `start_point`, `target_point`는 `end_point`, `region_mask_path`의 binary mask는 `area`에 대응한다. 별도의 독립 좌표값을 만들어 관리하지 않는다.

기본 비교 이미지는 **왼쪽: 원본 + 편집 설정 표시 / 오른쪽: 표시 없는 생성 결과**의 1×2 배치다. 원본 위에 빨간 원형 시작점(S), 초록 십자형 도착점(E), 시작→도착 화살표, 반투명 파란 영역과 경계를 표시한다. 아래 여백에는 좌표·영역 면적/비율·operation·seed·후보 ID·두 VQA 결과를 적는다. rotation/deformation에 anchor가 있으면 별도 기호로 표시한다.

sampler의 요청값뿐 아니라 실제 backend에 전달된 좌표·mask와 전처리 변환을 저장한다. backend의 입력이 리사이즈/패딩/반올림으로 달라졌다면 이를 정규화 원본 좌표로 복원하여 **실제 전달한 편집 설정**을 표시한다. 화살표는 의도한 이동이며 생성 결과에서 측정한 이동이라고 주장하지 않는다.

`visualizations/<candidate_id>/drag_plan.png`, `comparison.png`, `metadata.json`을 저장한다. 시각화는 확인용 산출물이고 VQA 입력·증강 이미지·목표 수량과 분리한다. 기본 NO 삭제 정책은 생성 결과를 포함한 comparison/thumbnail에도 적용하며 원본과 설정만 포함한 drag_plan과 metadata는 남긴다. 상세 표시·좌표·삭제·재개 규칙은 [VISUALIZATION.md](docs/VISUALIZATION.md)를 따른다.

## 5. 두 VQA의 입출력

현실성·물리 검증은 [physical prompt](prompts/physical_plausibility_v1.txt)에 **후보 한 장**을 넣는다. 의미 보존은 [semantic prompt](prompts/semantic_preservation_v1.txt)에 **원본을 Image 1, 후보를 Image 2**로 넣고 고정 SourceProfile·사용자 설명을 데이터로 전달한다. 두 호출의 대화 상태를 공유하지 않는다.

각 호출은 [단일 VQA 응답 schema](schemas/vqa_response.schema.json)를 사용한다.

```json
{"answer":"YES","reason":"The visible evidence supports this check."}
```

결과 record에는 `physical`과 `semantic`을 별도 저장한다. 각 검증은 `status=completed/error/not_run`, raw completion, parsed response 또는 null, prompt hash, 모델/processor revision, 시도·시간을 포함한다. 최종 accept 여부는 coordinator가 계산하며 모델이 임의로 작성한 전체 verdict를 신뢰하지 않는다.

`YES/NO/UNCERTAIN`과 비어 있지 않은 reason만 허용한다. completion 전체에서 바깥 whitespace 제거 후 JSON 객체 하나를 파싱한다. 중복 key·추가 key·code fence·소문자 enum·잘린 JSON·JSON 밖 설명은 기술 오류다. reason의 YES/NO 단어는 판정에 영향을 주지 않는다. 사용자 prompt 안의 YES 예시가 응답으로 섞이지 않도록 새 token만 decoding한다.

로컬 Qwen3.5-4B를 평가 모드로 사용한다. baseline은 `enable_thinking=false`, `do_sample=false`, batch size 1이다. 모델 카드의 일반 sampling 권장값과 다른 ADVV 재현용 설정이며 실제 품질을 검증한다. 설정이 실제 template에 반영되는지 확인하고, thinking 문자열을 임의로 잘라 YES를 추출하지 않는다. 시각 전처리·token budget·dtype·양자화·출력 상한은 기록한다.

## 6. 최종 상태·삭제·중복

| 물리 검증 | 의미 검증 | 최종 상태 | 목표 수량 반영 |
| --- | --- | --- | --- |
| YES | YES | `accepted` | 저장·중복 검사·export 적합성 확인 후 +1 |
| NO | not_run | `rejected` | 0 |
| YES | NO | `rejected` | 0 |
| UNCERTAIN | not_run | `uncertain` | 0 |
| YES | UNCERTAIN | `uncertain` | 0 |
| 오류 | not_run | `verification_error` | 0 |
| YES | 오류 | `verification_error` | 0 |
| 생성 실패 | not_run | `generation_error` | 0 |

향후 두 검증을 모두 실행하는 실험에서는 **어느 한쪽 유효 NO이면 rejected, 둘 다 유효 YES일 때만 accepted**, NO 없이 기술 오류가 있으면 verification_error, 나머지 미확정은 uncertain/pending으로 처리한다. 실행하지 않은 판정은 성공으로 간주하지 않는다.

원본 및 이미 채택한 모든 source의 후보와 canonical RGB pixel hash가 같으면 `export_status=duplicate`로 두어 목표 수에 포함하지 않는다. 두 VQA가 YES인 사실은 보존한다. NO·UNCERTAIN·오류·중복은 다음 attempt의 새 편집을 유발한다. near-duplicate/편집량 threshold는 기본 채택 규칙에 추가하지 않고 다양성 지표로 먼저 평가한다.

NO 이미지는 record를 영속화한 후 현재 run이 소유한 생성 파일과 그 복사본만 삭제한다. 원본·이전 run·외부 symlink·accepted 파일은 삭제할 수 없다. 삭제 실패는 `cleanup_pending=true`로 기록하고 재개 시 정리한다. UNCERTAIN·기술 실패는 별도 quarantine에 둔다. Qwen 응답 원문·후보 hash·편집 계획은 이미지 삭제 후에도 보존한다.

NO의 생성 픽셀이 들어간 비교 PNG·썸네일도 같은 삭제 대상이다. 원본에 편집 계획만 표시한 `drag_plan.png`는 남긴다. 시각화를 만들기 위해 삭제된 후보를 다시 생성하지 않는다.

동일 후보의 의미 판정은 재질문하지 않는다. JSON 오류·일시적 기술 실패에 한해 동일 입력으로 최대 1회 추가 시도를 허용하고 모든 시도를 기록한다. OOM·모델 로드 실패는 자동 반복하지 않고 partial run을 저장한다. 총 시도 상한이 없다는 요구를 기술 실패를 숨기는 무한 재시도로 구현하지 않는다. 기본 `max_consecutive_technical_failures=3`은 generation_error/verification_error 후보가 연속 발생할 때 failed로 종료하는 장애 기준이다. 정상 NO/UNCERTAIN·중복은 이 횟수에 포함하지 않는다.

## 7. 수량·중단·재개

목표 N은 사용자 지정 양의 정수이며 기본 예시의 숫자를 실행 의도로 간주하지 않는다. `quota_scope=total`, `limit_policy=until_target_or_user_stop`, 총 attempt/time limit는 null이다. per-source quota는 현재 요구 범위가 아니다.

확정된 unique eligible artifact만 원자적으로 quota에 반영한다. `accepted_count == N`이 되면 새 후보 생성·검증을 시작하지 않는다. MVP는 candidate 한 건씩 처리하여 초과 채택을 피한다. future batch mode는 남은 quota와 이미 처리 중인 후보를 따로 관리해야 한다.

SIGINT/종료 요청을 받으면 새 작업 예약을 멈추고, 안전하게 확정된 파일·record·attempt cursor·count를 저장한다. `interrupted`로 종료하고 부분 export/report를 남긴다. 중단은 목표 달성으로 보고하지 않는다. 한 번도 목표에 도달하지 못하는 입력 조합도 있을 수 있으며 정상적인 낮은 수락률 때문에 임의 종료하거나 판정 기준을 낮추지 않는다.

재개 시 record와 파일 hash를 대조하여 count를 재계산한다. 완료된 판정은 다시 실행하지 않으며, 삭제된 NO 파일이 없다고 재생성하지 않는다. 예약되었으나 미완료인 같은 attempt를 복구한 후 다음 index로 이동한다. 사용자 설명·SourceProfile·prompt·sampler·모델·dtype 변경은 새 run을 요구한다. 실행 GPU ID 변경은 허용하되 새로운 실행 segment의 GPU·환경을 기록한다.

run status는 `running`, `completed`, `interrupted`, `failed`로 구분한다. `completed`는 목표 수량을 충족한 manifest와 활성화된 필수 시각화가 완전히 저장되었을 때만 설정한다. N장을 채운 뒤 시각화 저장만 실패했다면 추가 후보를 생성하지 않고 해당 렌더링만 복구한다. 진행 표시에는 `accepted/N`, attempts, 각 VQA 탈락 수, 중복 수, 경과 시간을 포함한다. 수락률에 따른 ETA는 추정임을 표시하고 관측 수락이 없으면 N/A다.

## 8. 모델 학습과 라벨 정책

Qwen 추가 학습은 **필요하지 않은 출발점**이다. 이미 학습된 멀티모달 checkpoint로 원본 이해와 VQA를 수행한다. 필요한 검증 데이터는 처음부터 전체 데이터의 GT가 아니라, 소규모 원본·후보 쌍을 사람이 판정한 평가 표본이다. 반복적인 도메인 오류가 확인되면 prompt·시각 해상도·보존 기준을 먼저 점검하고 별도 연구로 fine-tuning을 검토한다. 평가 표본을 그대로 학습에 섞지 않는다.

MVP는 라벨 없는 이미지와 계보 manifest를 출력한다. 후속 classification은 클래스 의미 보존 확인이, detection/segmentation은 새 이미지에 맞는 box/mask 작성·검수가 필요하다. VQA의 의미 YES는 GT annotation을 생성하거나 보장하지 않는다. 편집 mask도 segmentation GT가 아니다.

## 9. 재현성과 완료 기준

run에는 입력 snapshot·파일/픽셀 hash, 고정 SourceProfile, source별 attempt index, seed, sampler 설정, 실제 EditRequest·mask, 모델·processor·가중치 revision, upstream commit·patch hash, 두 VQA prompt/raw response, 저장 상태, 실행 GPU segment와 시간을 보존한다. Qwen cache key에는 check 종류와 **각 이미지의 순서·hash**, profile/user hint hash, 모델·prompt·전처리·decoding 설정을 모두 포함한다.

가중치·소스·환경을 고정해도 GPU·kernel 차이로 bitwise 재현성이 달라질 수 있다. GPU 변경 기록 없이 동일 환경 재현이라고 주장하지 않는다.

MVP 완료는 실제 DragFlow와 실제 로컬 Qwen으로 N장의 고유한 두 검증 통과 이미지를 만들고 사용자 중단·재개·삭제를 검증한 상태다. 장비/가중치가 준비되지 않아 실행하지 못했다면 구현 완료와 통합 검증 대기를 구분한다. 후속 모델의 성능 개선은 MVP 완료 조건이 아니다.
