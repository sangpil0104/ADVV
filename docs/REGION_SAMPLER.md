# 객체 중심 영역 sampler v2 설계

상태: **SAM 3 worker 연결 구현(CPU 테스트). 부위 이름 텍스트 part adapter는 T012에서 GPU로 `assets/` 5장 실행(부위 4–7개/장), T011 `v2_pilot_001`에서 Qwen v3 profile `parts`와 SAM 3 proposal을 9장에 실행했으나 v2 `advv run`은 첫 후보 생성 timeout으로 판정 0건. v4 profile은 GPU 미실행**. 작성 2026-09-30, 같은 날 SAM 3 기본으로 갱신, T009b에서 worker 연결, 2026-10-01 T012에서 부위 이름 텍스트 prompt를 점 prompt와 병행, T015에서 부위 이름을 subject별로 나눈 `source_profile_v4`와 rotation 목표 칸 선택(실효 각도 허용 오차)을 추가(CPU 테스트만, GPU 미실행).

- 구현됨(CPU, fixture로 테스트): proposal 저장 계약과 검증 로더(`src/advv/proposals.py`), mask 정리·필터·중복 제거, `object_region_v2` sampler(`src/advv/sampler.py`), 설정 검증, `source_profile_v4`의 `subjects`·subject별 `parts`(T015), pipeline의 proposal 생성·고정·`source_no_region` 보류, `sampling_skipped`, 시각화 footer, SAM 3 proposal worker(`src/advv/backends/region.py`, `sam3.py`)와 raw receipt, `advv prepare --only sam3`, preflight 검사, feature grid에서 사라지는 얇은 mask 제외(§3.3), 부위 이름 텍스트 part와 출처 기록(§3.1, §3.2), rotation 실효 각도 허용 오차(§3.3).
- 가짜 SAM 3 패키지(`tests/fixtures/fake_sam3/`)로 별도 프로세스 계약·receipt·오류 경로를 확인했다. T009d(GPU9, 2026-10-01)에서 T012 이전 adapter를 `assets/` 사진 5장에 실행했다: entity는 5장 모두 잡혔지만 점 prompt part는 거의 나오지 않았다(고양이 0개, 굴착기는 track만, 설표는 entity의 67% 덩어리). SAM 3가 점마다 entity 전체를 가장 확신하고(점수 0.8 이상 mask는 거의 entity 전체) 작은 mask(multimask 0)의 예측 IoU 중앙값이 0.1–0.4이기 때문이다. 그래서 T012에서 부위 이름 텍스트 prompt를 추가했다. **T012 adapter의 GPU 실행은 아직 하지 않았다**(`tests/test_sam3_integration.py`, `-m integration`).
- 미구현: Grounding DINO + SAM 2.1 비교 backend(설정은 받지만 worker·preflight가 "not implemented" 오류로 멈춘다).
- 기본 sampler는 여전히 `random_geometry_v1`이다.

## 1. 배경

`random_geometry_v1`은 이미지 안의 무작위 타원·사각형을 편집 영역으로 쓴다([SPEC §4](../SPEC.md#4-무작위-편집-계획)). DragFlow는 영역 단위 affine supervision을 사용하므로, 객체와 무관한 사각형을 옮기면 경계가 결과에 이음매로 남거나 객체가 잘린 채 이동한다. DragFlow 자체는 임의 모양 mask를 받는다. 따라서 편집 영역을 **실제 객체(entity)와 그 부위(part)의 mask**로 바꾼다.

공식 DragFlow는 사람이 칠한 `operation.png`를 받아 외곽 윤곽선별로 영역을 나누고, 드래그 시작점을 포함하는 윤곽선을 그 operation의 영역으로 쓴다(`framework/dashboard_utils.py`의 `_get_independent_regions`). 그 뒤 upstream은 드래그 시작점을 선택된 영역의 centroid로 **교체**한다. 그러므로 v2의 mask는 **구멍 없는 하나의 연결 영역**(8-연결)이어야 하고, 윤곽 선택에 쓰는 점은 mask 안에 있어야 하며, 이동량·회전·범위 검사는 mask centroid를 시작점으로 계산해야 계획과 실행이 일치한다(§3.3).

## 2. 결정 요약

| 항목 | 결정 |
| --- | --- |
| sampler 버전 | `object_region_v2` (v1과 공존, `sampler.version`으로 선택. 기본은 v1) |
| 기본 proposal 모델 | **SAM 3**: 텍스트 명사구 → entity 인스턴스 mask. part는 두 출처를 병행한다: 부위 이름 텍스트 prompt(`"<subject> <part>"`, `"<part>"`) → entity 안에 들어간 인스턴스, entity 내부 점 → multimask 3개 |
| 비교용 | Grounding DINO(텍스트 → box) + SAM 2.1(box/point → mask). 설정 `region_proposal.backend`로 선택 |
| granularity | entity와 part를 모두 저장하고 operation별로 선택 |
| 실패 시 | 기하 영역으로 fallback하지 않는다. 원본을 `source_no_region`으로 보류 |
| proposal 파일이 없을 때 | worker로 생성한다. 생성이 기술 오류로 끝나면(재시도 후) 오류(`region_proposals_missing`)로 멈춘다. "영역 없음"으로 보지 않는다 |
| 실행 시점 | 원본마다 한 번, profile 고정 후·첫 후보 생성 전에 proposal을 만들어 고정 |

SAM 3를 기본으로 둔 이유: 한 모델이 텍스트 → 인스턴스 mask와 점 → 3단계 mask를 모두 처리하므로 entity와 part를 같은 이미지 특징에서 얻는다. 두 모델을 잇는 box 단계가 없다. 대신 요구 환경(Python ≥3.12, torch ≥2.7)이 DragFlow·Qwen과 달라 별도 venv가 필요하고, HF 수동 승인이 필요하다(승인 완료). 논문은 학습 분야 밖의 세밀한 개념에 약하다고 밝히므로 결함 자체보다 **결함을 가진 객체**의 명사구로 grounding한다.

part 전용 모델(사람 파싱·동물 부위 분할 등)은 산업 결함·사고·희귀 생물 도메인에서 일반화되지 않으므로 쓰지 않는다. SAM 계열의 point prompt는 도메인과 무관하게 whole/part/subpart 계층을 준다.

## 3. 파이프라인

```text
원본 ingest → Qwen SourceProfile 고정 (source_profile_v4: subjects·subject별 parts 포함)
→ RegionProposal worker (proposals.json이 없는 원본마다 1회, 결과 고정):
    SAM 3 (기본)
      1. set_image 1회 → subjects 명사구마다 텍스트 prompt → entity 인스턴스 mask + 점수
      2a. entity마다 그 entity 명사구(subject)의 parts 이름으로만 텍스트 prompt("<entity 명사구> <part>", "<part>") → 인스턴스마다
          entity 포함 비율 기록 → text part 후보 (text_parts가 켜져 있을 때)
      2b. entity mask 내부 seeded 점(part_prompt_points) → 점 prompt multimask 3개 + 예측 IoU → point part 후보
    Grounding DINO + SAM 2.1 (비교)
      1. 명사구 → entity box → SAM 2.1 box prompt → entity mask
      2. 위 2와 같은 점 prompt → SAM 2.1 multimask → part 후보
    raw mask·점수·점 → proposals/<source_id>/receipt/ (raw_masks.npz, raw.json, response.json)
→ coordinator(CPU):
    3. build_proposals: 최대 연결 성분·구멍 채움·필터·중복 제거
    4. write_proposals → proposals/<source_id>/ (mask 먼저, proposals.json 마지막)
    worker 종료(DragFlow 로드 전)
→ pipeline: proposals 로드·검증·hash 고정, 사용할 (proposal, operation)이 없으면 source_no_region
→ while quota < N:
    sampler v2(CPU): seed·source·attempt로 operation·level·proposal·방향·강도 선택
    → DragFlow → 두 VQA (기존과 동일)
```

GPU를 쓰는 곳은 1–2의 모델 호출뿐이다. 후처리(3–4), 검증, sampler는 CPU 코드이며 v1과 같이 seed·source·attempt에 대해 결정론적이고 가중치 없이 테스트한다. worker(`advv.backends.worker --kind proposal`)는 `advv.proposals.collect_raw`로 모델을 호출하고 raw mask와 점수만 receipt에 쓴다. 정리·필터·저장은 coordinator가 `build_proposals`/`write_proposals`로 한다. worker venv에도 advv가 import되어야 한다(`.venv-sam3/bin/python -m pip install --no-deps -e .`).

worker 세부(`src/advv/backends/sam3.py`, 공식 commit `2345a4a`):

- `build_sam3_image_model(bpe_path=<패키지 내장 bpe>, device="cuda", checkpoint_path=<model_path>/sam3.pt, load_from_HF=False, enable_inst_interactivity=True)`. 점 prompt에는 `enable_inst_interactivity=True`가 필요하다. 빌더 stdout에 upstream의 missing key 출력이 있으면 치명 오류로 멈춘다.
- TF32를 켜고 원본마다 `torch.inference_mode()` + bf16 autocast 안에서 `set_image`를 한 번 부른 뒤 명사구마다 `reset_all_prompts` → `set_text_prompt`, 이어서 같은 state로 `predict_inst(point, multimask_output=True)`를 부른다(T009c에서 두 호출이 서로 간섭하지 않음을 확인).
- `Sam3Processor(confidence_threshold)`는 `scores > confidence_threshold`인 인스턴스만 남기며, 이 비교에서 Python float는 점수 텐서의 dtype(float32/bf16)으로 바뀐다. 그래서 하한 바로 아래 부동소수 값(`nextafter`)을 넘겨도 다시 하한과 같은 값이 되어, 점수가 정확히 하한인 인스턴스를 SAM 3가 버린다(T009b QA L1). T012부터 processor에는 `min(min_entity_score, min_text_part_score) − 0.01`(0 미만이면 0, `text_parts`가 꺼져 있으면 `min_entity_score`만)을 넘기고, 채택 여부는 ADVV의 `≥` 필터(`collect_raw`, `build_proposals`, 로더)가 정한다. 0.01은 1 근처 bf16 간격(2⁻⁸ ≈ 0.0039)보다 크다. 그 폭만큼 하한 미만 인스턴스가 receipt에 더 남을 수 있고 build에서 `entity_score`·`text_part_score`로 거부된다.
- 부위 이름 텍스트 prompt는 entity 텍스트 prompt와 같은 state에서 `reset_all_prompts` → `set_text_prompt`로 부른다. entity는 자기 명사구(subject)에 profile이 적은 부위만 묻는다(T015). 명사구마다 한 번만 묻고(두 subject가 같은 `"<part>"`를 적었으면 공유) 각 인스턴스를 그 명사구를 물은, 점수 하한을 넘은 entity마다 포함 비율과 함께 기록한다.
- text part의 포함 비율 = 인스턴스 mask 중 `clean_mask`한 entity mask 안에 든 픽셀의 비율. 한 명사구가 인스턴스를 여러 개 내면 각각 후보다. `min_text_part_score` 미만이거나 포함 비율이 `text_part_containment` 미만인 인스턴스는 receipt에 점수·box·포함 비율만 남기고 mask는 저장하지 않는다.
- part 점은 entity를 `clean_mask`한 뒤 `part_prompt_points`로 고른다. seed는 run seed·source ID·원본 hash·worker 출력의 entity 순번으로 정한다(`part_seed`). multimask 3개를 모두 part 후보로 넘기며, `min_part_score` 미만 mask는 receipt에 점수만 남기고 mask는 저장하지 않는다(`build_proposals`가 점수로 먼저 거른다).
- raw receipt(`raw.json`, schema `1.2`, T015에서 `parts`가 subject별 목록이 되어 올림)에는 backend, `models`(`facebook/sam3` revision, `facebookresearch/sam3` commit), 명사구, subject별 part 이름(`parts`), entity 점수·box, part마다 출처(`source: text|point`), text part의 명사구·인스턴스 순번·점수·box·포함 비율, point part의 점 좌표·multimask 순번·예측 IoU, `raw_masks.npz`의 SHA256, torch 버전·정밀도·피크 VRAM·추론 시간을 쓴다. coordinator가 proposals.json을 쓰기 전에 멈췄다면 재개 때 같은 명사구·part 이름의 완료 receipt를 GPU 없이 다시 읽는다. schema `1.0`·`1.1` receipt는 읽지 않는다.
- worker는 point part를 억제 설정(`suppress_point_parts_with_text`, §3.2)과 상관없이 항상 묻고 receipt에 남긴다. 억제는 build 단계 규칙이라 같은 receipt를 설정만 바꿔 다시 build할 수 있다(T013에서 T012 receipt 5장을 CPU로 다시 build해 확인).
- 테스트용 가짜 `sam3` 패키지는 `__fixture__`로 자신을 표시하고, adapter는 그 출력을 `backend=fixture`로 기록한다. 가짜 processor는 `reset_all_prompts` 없이 두 번째 텍스트 prompt가 오면 오류를 내므로(upstream은 prompt를 state에 누적한다) adapter의 reset 누락을 테스트가 잡는다. fixture proposal은 fake 생성 backend run에서만 로드된다.
- 오류: worker 로드 실패·CUDA 없음·예상 밖 예외는 치명(`FatalBackendError`)으로 run을 멈춘다. timeout은 이미지 단위 기술 오류라 `max_technical_retries`만큼 새 worker로 다시 시도하고, 그래도 실패하면 `region_proposals_missing`으로 멈춘다. 시도 기록은 `state.json`의 `region_proposal_generation[source_id]`에 남는다.

### 3.1 대상 명사구와 부위 이름

`source_profile_v3`([prompt](../prompts/source_profile_v3.txt), [schema](../schemas/source_profile_v3.schema.json))는 v2에 `subjects`(핵심 의미를 가진 객체의 짧은 영어 명사구 1–3개, 단어 5개 이하)를 추가한다. prompt와 schema는 같은 형식을 요구한다: 소문자·숫자 단어(숫자로 시작해도 된다, 예: `3d printer`)를 공백 하나로 잇고, 단어 안의 `-`·`'`만 허용한다(schema 패턴 `^[a-z0-9]+(['-][a-z0-9]+)*( [a-z0-9]+(['-][a-z0-9]+)*){0,4}$`). 대문자나 다른 구두점은 schema 위반이라 profile이 기술 오류가 된다. `object_region_v2`를 선택하면 config 검증이 profile schema에 `subjects`가 필수인지 확인한다(v3·v4 모두 만족). v1은 v2 profile을 그대로 쓴다. 현재 v2의 기본 profile은 `source_profile_v4`([prompt](../prompts/source_profile_v4.txt), [schema](../schemas/source_profile_v4.schema.json))이며 `subjects` 규칙은 v3와 같고 `parts` 형식만 다르다. worker는 `subjects`를 그대로(같은 순서) grounding 명사구로 쓰며, Grounding DINO 비교 경로에서는 마침표로 이어 붙인다(공식 형식 `"cat . car ."`). 사용자 `preserve_hint`는 계속 선택 입력이다.

`parts`는 보이는 subject의 부위 중 subject의 정체를 바꾸지 않고 따로 움직일 수 있는 것의 짧은 영어 명사구다(예: `tail`, `front leg`, `head`, `bucket`, `boom arm`, `door`). T012의 v3는 이미지 전체에 하나의 목록(0–5개)이었고 worker가 모든 entity에 같은 목록을 조합했다. 그래서 `v2_pilot_001`(T011)의 `accident_car_tree`에서 subject `tree`에도 `tree front bumper` 같은 자동차 부위 prompt가 나가 39개 인스턴스가 포함 비율로 거부됐다(결과에는 무해하지만 질의 비용과 receipt 혼란). T015의 **v4는 `parts`를 subject별 목록**으로 바꾼다:

```json
{
  "subjects": ["damaged car", "tree"],
  "parts": [{"subject": "damaged car", "parts": ["front wheel", "hood"]}, {"subject": "tree", "parts": []}]
}
```

- schema: `parts`는 0–3개의 `{"subject", "parts"}` 객체(다른 키 금지), `subject`와 각 부위 이름은 `subjects`와 같은 패턴, subject마다 부위 0–5개·중복 금지. prompt는 subject마다 하나씩 `subjects` 순서대로 쓰라고 하고, 다른 subject의 부위를 적지 말라고(나무에는 범퍼가 없다) 지시한다.
- schema로 표현할 수 없는 교차 규칙은 profile 응답을 읽을 때 코드가 검사한다(`profile_parts_check`): 각 항목의 `subject`가 profile `subjects`에 있고 서로 달라야 한다. 위반은 `ResponseError`라 profile 기술 오류(재시도 후 `source_error` 보류)다. 항목이 없는 subject는 부위 이름이 없는 것으로 본다(순서와 누락은 허용).
- worker는 entity마다 그 entity 명사구와 같은 `subject` 항목의 부위만 조합한다. 다른 subject의 부위는 묻지 않는다.
- subject 단어를 반복하지 않는다(`kitten tail`이 아니라 `tail`). prompt는 부위가 "고칠" 대상이 아니라 움직이거나 자세를 바꿀 수 있는 대상임을 밝히고, 균열·찌그러짐·파손·화상·누출·상처 등 결함·사고 증거와 증거를 **담은** 부위(찌그러진 문, 깨진 범퍼 등, T013)를 part로 적지 말라고 지시한다(그 증거는 그대로 보존해야 한다, [SPEC §3.3](../SPEC.md#33-sourceprofile)).
- v3를 제자리 수정하지 않은 이유: v3는 `runs/v2_pilot_001`(T011)에서 실제로 쓰여 고정된 profile이 있다. v3 prompt·schema 파일은 그대로 두며, v3 profile은 `text_parts: false`인 v2(이름 없이 `subjects`만 사용)와 v1에서만 쓸 수 있다. `text_parts: true`인 v2는 config 검증이 v4 형식(`parts` 항목이 `subject`·`parts`를 가진 객체)을 요구한다. v3 flat `parts`를 subject별로 해석하는 호환 경로는 두지 않았다: T015에서 `src/advv`가 바뀌어 implementation hash가 달라지므로 v3 run은 어차피 재개할 수 없고, proposals.json·receipt schema도 `1.2`로 올라 v3 시절 파일은 읽지 않는다.
- 문구 형태(`text_part_forms`): `subject_part`는 `"<entity 명사구> <part>"`(예: `excavator bucket`), `part`는 `"<part>"` 단독이다. 기본은 둘 다다. 어느 쪽이 SAM 3에서 더 잘 잡히는지 아직 실측하지 않았으므로 둘 다 묻고, 같은 부위의 두 결과는 중복 IoU로 하나만 남긴다. `part` 단독은 다른 객체의 같은 부위도 찾을 수 있지만 포함 비율 검사로 entity 밖 인스턴스를 거른다. 명사구 하나당 text 호출 1회(같은 state, 이미지 인코딩 재사용)라 비용은 subject별 part 수 × 2의 합 이하다(T015 전에는 part 수 × 2 × entity 명사구 수).
- 이력: T012·T013은 v3를 제자리 수정했다(그때까지 v3를 쓴 run이 없었다). T011이 v3로 `v2_pilot_001`을 실행했으므로 T015부터는 새 버전(v4)을 만든다.
- `subjects`·`parts`는 영역 proposal용 이름이며 보존 기준이 아니다(v4에서도 같다). 의미 보존 VQA의 `{{preservation_context}}`에는 profile의 `summary`·`must_preserve`·`uncertain`(과 사용자 `preserve_hint`)만 넣는다([SPEC §5](../SPEC.md#5-두-vqa의-입출력)).
- `text_parts: true`이고 `object_region_v2`이면 config 검증이 profile schema의 `parts`가 필수이고 subject별 형식(v4)인지 확인한다. `text_parts: false`이면 worker에 part 이름을 넘기지 않고(point part만) proposals.json의 `parts`는 빈 목록이다.

### 3.2 proposal 필터 (초기값, 미검증)

| 조건 | 설정 키 | 기본값 | 의미 |
| --- | --- | --- | --- |
| entity 점수 하한 | `min_entity_score` | 0.50 | SAM 3 텍스트 prompt 점수(공식 `confidence_threshold` 기본값과 같음) |
| part 점수 하한 | `min_part_score` | 0.80 | 점 prompt의 예측 IoU |
| 부위 이름 텍스트 part | `text_parts` | true | profile v4 `parts`(subject별)로 텍스트 prompt를 점 prompt와 병행 |
| 텍스트 문구 형태 | `text_part_forms` | `[subject_part, part]` | `"<subject> <part>"`, `"<part>"` 중 사용할 것 |
| 텍스트 part 점수 하한 | `min_text_part_score` | 0.50 | SAM 3 텍스트 인스턴스 점수. 공식 `confidence_threshold` 기본값 0.5를 따른 미검증 초기값 |
| 텍스트 part의 entity 포함 비율 하한 | `text_part_containment` | 0.85 | 인스턴스 mask 중 entity 안에 든 비율. 미만이면 그 entity의 part가 아님. T014에서 0.90 → 0.85(사용자 결정): 굴착기 crawler track이 entity mask 경계 때문에 0.865/0.888로 탈락했고, entity 밖 다른 차 바퀴는 0.0이라 0.85에서도 걸러짐 |
| entity 면적 / 이미지 면적 | `entity_area_fraction` | 0.02 – 0.60 | 너무 작거나 배경 수준인 mask 제거 |
| part 면적 / entity 면적 | `part_area_fraction_of_entity` | 0.02 – 0.70 | entity와 거의 같은 mask는 part가 아님(상한 < 1 강제). 하한은 T013에서 0.05 → 0.02(꼬리·팔 같은 작은 부위 유지, 미검증) |
| 중복 판정 IoU | `dedupe_iou` | 0.85 | 정렬 순서에서 앞선 것 하나만 유지 |
| part 포함 중복: 포함 비율 | `part_dedupe_containment` | 0.95 | 작은 part 중 큰 part 안에 든 비율. 아래 면적비와 **둘 다** 넘으면 같은 부위(미검증) |
| part 포함 중복: 면적비 | `part_dedupe_area_ratio` | 0.80 | 작은 part 면적 / 큰 part 면적. 붐 vs 붐+암처럼 작은 비율의 계층은 유지(미검증) |
| 텍스트 part가 있으면 점 part 생략 | `suppress_point_parts_with_text` | true | entity에 필터를 통과한 text part가 하나라도 남으면 그 entity의 point part를 쓰지 않음 |
| Grounding DINO 임계값 | `grounding_dino.box_threshold` / `text_threshold` | 0.35 / 0.25 | 비교 경로 전용. 공식 예제값 |

- 경계값은 포함한다(`lo ≤ x ≤ hi`, 점수 `≥` 하한). 중복은 IoU `≥ dedupe_iou`이면 버린다. part는 여기에 더해 **포함 중복**도 버린다: 작은 mask의 `포함 비율(교집합 / 작은 면적) ≥ part_dedupe_containment`이고 `면적비(작은 / 큰) ≥ part_dedupe_area_ratio`인 쌍(`nested_duplicate`, 대칭). 완전히 포함된 쌍은 면적비 = IoU이므로 기본값에서는 IoU 0.80–0.85의 "거의 같은" 쌍만 새로 합쳐진다. entity 중복은 IoU만 본다.
- 남는 쪽: 중복 판정은 아래 정렬 순서대로 하나씩 채택하면서 이미 채택된 part와 비교하므로, 두 중복 중 **정렬에서 앞선 것**(text가 point보다, 같은 출처면 점수가 높은 것, 같으면 worker 출력 순서)이 남는다. 포함 중복도 같다(큰 쪽이 아니라 앞선 쪽).
- point part 억제(`suppress_point_parts_with_text: true`, 기본): text part를 먼저 모두 거른 뒤 그 entity에 남은 text part가 1개 이상이면 그 entity의 point part를 점수 검사 전에 모두 `point_suppressed_by_text`로 거부한다. `text_parts`가 꺼져 있거나 profile `parts`가 빈 목록이거나 text part가 모두 걸러진 entity에서는 기존처럼 point part를 쓴다. entity별 판단이다. 이유: T012 GPU 실측에서 사고 차량의 point part가 손상 증거(들린 보닛·깨진 앞유리)를 편집 후보로 남겼다.
- mask 정리: 8-연결 최대 성분만 남기고, 테두리와 4-연결되지 않은 배경(구멍)을 채운다. part는 먼저 entity와 교집합을 취한다. text part와 point part는 같은 정리·면적 비율·중복 IoU를 거치고 sampler에서 같은 grid 검사(§3.3)를 받는다. 차이는 점수 하한(`min_text_part_score` / `min_part_score`)과 text part의 포함 비율 검사뿐이다.
- 순서: entity는 점수 내림차순, 같으면 worker 출력 순서. part는 **text part 먼저, 그다음 point part**이고 각 묶음 안에서 점수 내림차순, 같으면 worker 출력 순서다. 두 점수는 척도가 다르므로(검출 신뢰도 vs 예측 IoU) 서로 비교하지 않으며, 같은 mask가 두 출처에서 나오면 이름이 있는 text part를 남긴다. ID는 이 순서로 `entity_000`, `part_000_003`처럼 붙는다.
- 거부 수(`rejected`)는 출처별로 센다: point part는 `point_suppressed_by_text`·`part_score`·`part_area`·`part_duplicate`·`part_nested_duplicate`, text part는 `text_part_score`·`text_part_containment`·`text_part_area`·`text_part_duplicate`·`text_part_nested_duplicate`. 억제된 point part는 `point_suppressed_by_text`에만 센다.
- raw 입력의 entity에 `phrase`·`score`·`mask`가, text part에 `phrase`·`score`·`containment`·`mask`가, point part에 `point`·`score`·`mask`가 없으면 `DataError`다(값이 `null`인 mask는 허용, 점수 하한을 넘었는데 null이면 `DataError`).
- part 점은 entity bbox를 √n×√n 칸으로 나눈 칸마다 entity 내부 픽셀 하나를 seed로 고르고, 넘치면 n개로 줄인다(`part_points_per_entity`, 초기값 16).

### 3.3 operation과 granularity

| operation | 영역 | 점·anchor |
| --- | --- | --- |
| relocation | entity | 이동량은 entity bbox가 이미지 안에 남는 정수 shift 상자 `dx ∈ [−x0, W−1−x1]`, `dy ∈ [−y0, H−1−y1]`와 거리 구간의 교집합에서 뽑는다. 교집합이 비는 entity는 제외 |
| rotation | part | anchor = part를 k px 팽창한 영역 ∩ (entity − part)의 centroid(관절 위치). 교집합이 비면 그 part는 rotation에 쓰지 않는다 |
| deformation | part 또는 entity | 끝점이 이미지 안에 남는 shift 상자와 displacement 범위의 교집합에서 뽑는다 |

- 점 두 개를 구분한다. **`source_point` = mask의 반올림 centroid**(원본 해상도, v1과 같은 계산)이며 DragFlow가 실제로 드래그를 시작하는 점이다. 이동량·`entity_bbox_after`·회전·grid 항등 검사는 모두 이 점으로 계산한다. **`region_select_point`** 는 upstream이 `pointPolygonTest`로 윤곽을 고를 때만 쓰는 mask 내부 점으로, centroid가 mask 안이면 centroid, 아니면 centroid에 가장 가까운 mask 픽셀이다. 오목한 proposal(C자·L자·꼬리 등)도 버리지 않는다. `instruction.json`에는 `centroids = [region_select_point, target_point]`로 넘긴다.
- upstream은 mask를 feature grid 크기로 `F.interpolate(bilinear, align_corners=False)`(antialias 없음) 축소하고 `> 0.5`로 자른 영역의 반올림 centroid에서 드래그를 시작한다(`dragger.py:362-370`, `dragger_utils.py:83-99`). 축소 폭이 scale(2–4 px)과 비슷한 얇은 구조는 격자 위상에 따라 사라지거나 일부만 남는다. 사라지면 upstream은 경고만 내고 grid (0, 0)에서 드래그한다. 그래서 sampler는 같은 축소를 numpy float32로 재현해(`upstream_grid_region`, torch 대조 결과 0.5 동점의 부동소수 차이 외 일치) **grid 영역이 비었거나(`grid_empty`) grid centroid가 계획 `source_point`의 grid 점과 한 칸(각 축 차이 1) 넘게 다른(`grid_centroid_shift`) proposal을 정적으로 제외**하고 `state.json`의 `region_proposals[source_id].excluded`에 사유별 수를 남긴다. 남은 proposal에서만 upstream 시작점이 원본 centroid와 grid 한 칸 이내다. 계획의 `operation_params.upstream_grid_start`에 이 grid 시작점을 기록한다.
- 선택 순서: 사용 가능한 operation 균등 → 그 operation의 level 균등 → proposal 균등 → 기하. 각 단계는 attempt seed의 `random.Random`으로 뽑는다.
- relocation/deformation의 shift는 위 상자 ∩ 거리 구간(`displacement_diagonal_fraction × 대각선`) ∩ feature grid에서 시작점과 다른 칸·grid 범위 안(`grid_ok`와 같은 계산)을 만족하는 정수 shift 전체에서 `1/거리` 가중으로 한 번 뽑는다. 이 가중은 v1의 "거리 균등·방향 균등"을 허용 집합 안으로 조건부 근사한 것이며, 이 집합에서 뽑으므로 재시도가 없다. 집합이 빈 (proposal, operation)은 정적으로 제외한다.
- relocation/deformation의 목표 칸은 계획 시작점의 칸뿐 아니라 `upstream_grid_start` 칸도 피한다(upstream 이동량 0 방지).
- rotation 각도(요청각)는 v1처럼 설정 범위에서 연속 균등으로 뽑는다(|각도| < 2°는 다시 뽑음). upstream은 시작점을 grid centroid(`upstream_grid_start`)로 바꾸고 목표점·anchor를 grid로 반올림한 뒤 그 세 점의 `atan2` 차이로 회전한다(`dragger_utils.py:109-121`, `dashboard_utils.py:214-228`의 `scale_coordinates`, `dragger.py:371`). 그래서 실효 각도(`grid_rotation_degrees`)는 목표점이 놓인 **grid 칸**으로만 정해진다.
- 목표점 선택(T015): anchor의 grid 점을 중심으로 `upstream_grid_start`를 요청각만큼 돌린 실수 grid 점을 구하고, 그 반올림 칸과 주변 8칸(3×3) 중 끝점 범위·`grid_ok`·끝점 칸 ≠ `upstream_grid_start`를 만족하고 **실효 각도가 요청각과 같은 부호이며 `|실효각| ≥ rotation_min_executed_degrees`**(T015-fix, 초기값 2.0°)인 칸에서 **실효 각도가 요청각에 가장 가까운 칸**을 고른다(동률은 실수 grid 점까지 거리, 그다음 y·x 순). 목표 픽셀은 그 칸으로 반올림되는 픽셀 중 원본 해상도에서 시작점(centroid)을 anchor 중심으로 요청각만큼 돌린 실수 점에 가장 가까운 것이다. 이전에는 원본 픽셀에서 시작점을 돌린 뒤 반올림했으므로 그 픽셀이 grid에서 한 칸 옆에 떨어질 수 있었다.
- 방향·최소 실효각(T015-fix, 사용자 결정 2026-10-01 "같은 방향 + 최소 2°"): 허용 오차(3.0°)가 |요청각|(2–3°)보다 크면 반경 방향 칸(실효 0°, upstream이 회전하지 않음)이나 반대로 도는 칸이 오차 최소가 될 수 있었다(T015 QA M1: `src_cfe95f7139096ee7`의 `front bumper`, 반지름 11칸, 요청 −2.2° → 칸 실효 0.0°, 끝점 (577, 553)). 그래서 이 조건을 만족하지 않는 칸은 후보에서 빼고, 남은 칸에 아래 허용 오차를 적용한다. 같은 계획은 이제 끝점 (578, 557), 실효 −4.76°(오차 2.56° ≤ 3.0°)다. 남은 칸이 없거나 모두 허용 오차 밖이면 기존처럼 그 각도를 다시 뽑고, 범위 전체에서 그런 part는 정적으로 제외한다. `rotation_min_executed_degrees`는 요청각 하한(|요청각| < 2°는 다시 뽑음)과 **별개 키**다. 요청각 하한은 v1과 공유하는 코드 상수라 설정으로 바꾸면 v1 계획이 바뀌기 때문이고, 실효각 하한은 v2 grid 칸 선택에만 쓰인다. 초기값은 요청각 하한과 같은 2.0°이며 미검증이다.
- grid 반올림(T015-fix, QA L1): upstream `scale_coordinates`는 `torch.round(torch.tensor([x / W × g, …]))`로, Python float64로 나눈 값을 float32 텐서로 만든 뒤 half-to-even으로 반올림한다. v2 sampler의 grid 계산(`grid_point(..., float32=True)`, `valid_shifts`, `_cell_pixel`, `grid_ok`, 실효각, rotation 목표 탐색의 anchor 점(T017 QA L6), 계획 검증)은 같은 순서로 `np.round(np.float32(x / W × g))`를 쓴다(numpy도 half-to-even). float64 반올림과는 `x / W × g`가 k + 0.5에 float32 오차 이내로 가까울 때만 다르다(예: 1000×750 → grid 330×245에서 x = 350은 float64 115.49999999999999 → 115, float32 115.5 → 116). v1(`random_geometry_v1`)은 기존 float64 반올림을 유지한다(v1 계획·golden 불변). 960×720 등 자주 쓰는 크기에서는 두 방식이 같아 v2 golden도 바뀌지 않았다. torch로 직접 대조하지는 않았다(근거: upstream 원문과 `torch.tensor` 기본 dtype float32, `torch.round` half-to-even 문서).
- 허용 오차: `|grid_rotation_degrees − requested_rotation_degrees| ≤ max(rotation_max_error_degrees, rotation_max_error_fraction × |요청각|)`(설정 `sampler.object_region`, 초기값 3.0°·0.30, 미검증, 경계 포함). 가장 가까운 칸도 이 밖이면 그 각도를 버리고 operation부터 다시 뽑는다(attempt 기하 예산 512회를 소모). 설정 범위를 0.5° 간격으로 훑어 허용 오차 안의 각도가 하나도 없는 part는 정적으로 rotation에서 제외한다. 모두 seed·source·attempt로 결정론적이다.
- 반지름 하한도 유지한다: `upstream_grid_start`와 anchor의 grid 점 거리 `grid_radius_cells`가 `MIN_ROTATION_RADIUS_CELLS`(4칸, 코드 상수) 미만인 part는 rotation에서 제외한다(QA2: 반지름 2칸에서 오차 중앙값 약 8°). 계획의 `operation_params`에는 요청각(`requested_rotation_degrees`), 실효 각도(`grid_rotation_degrees`), 적용한 허용 오차(`rotation_tolerance_degrees`, T015), `grid_radius_cells`를 기록한다.
- T015 원인 분석(`v2_pilot_001` 첫 후보, `accident_car_tree`의 `front wheel`, 960×720 → grid 320×240, scale 3): 요청 −10.51°. 원본 픽셀 회전·반올림으로 끝점 (440, 542)(픽셀 각도 −11.63°). upstream 기준으로 시작점은 grid centroid (145, 180)(실수 145.33, 180.0), anchor (447, 521) → (149, 174)(실수 173.67), 끝점 → (147, 181)(실수 146.67, 180.67)이 되어 실효 −17.74°였다. 단계별로 시작점 교체만 반영하면 −13.84°, anchor 반올림까지 −14.40°, 끝점 반올림까지 −17.74°로 끝점 반올림이 가장 크다. 반지름 7.2칸(≥ 4칸 규칙 통과)에서 한 칸은 약 8°이고 요청 호 길이는 약 1.3칸이라, 칸 하나 차이가 요청각의 70%를 넘는 오차가 됐다. 새 규칙은 칸 (146, 181), 끝점 (439, 542), 실효 −10.49°를 고른다(`tests/test_rotation_accuracy.py` 회귀 테스트).
- 같은 run의 rotation 가능 part 30개(9장 proposal, 읽기만)에 각도를 무작위로 넣어 비교한 CPU 측정: 이전 방식 오차 중앙값 1.48°·90% 4.27°·최대 9.58°, 허용 오차 밖 937/10705(8.8%). 새 방식 중앙값 0.34°·90% 1.17°·최대 3.37°, 허용 오차 밖 4/11188(0.04%), 유효 목표 칸 없음 0건. 실효 각도는 계획 수치이며 DragFlow 결과 이미지에서 측정한 회전각이 아니다.
- T015-fix 재측정(같은 30개 part, 요청각 −30..30°를 0.05° 간격, |요청각| ≥ 2°): 수정 전 규칙(CPU 재현)은 채택 33638건 중 실효 0° 16건, 0 < |실효| < 1° 165건, |실효| < 2° 434건, 반대 부호 0건. 수정 후 채택 33506건, 실효 0°·반대 부호·|실효| < 2° 모두 0건, 오차 중앙값 0.34°·90% 1.16°·최대 3.43°. 무작위 비교(part마다 `Random(0)` 400회)에서는 가장 가까운 허용 칸이 허용 오차 밖이라 다시 뽑게 되는 비율이 0.04%에서 0.38%(42/11100)로 늘었고, 30개 part 모두 여전히 rotation 가능하다.
- 접촉 팽창 kernel은 한 변 `2k+1`의 정사각형(8-이웃, PIL `MaxFilter`)이다. 즉 part에서 Chebyshev 거리 k 이내의 픽셀을 포함한다.
- 한 attempt의 기하 예산은 512회다(각도 거부만 소모). 소진하면 run을 멈추지 않고 `SamplingSkipped`로 그 attempt를 `state.json`의 `sampling_skipped[source_id].attempts`에 기록하고 cursor를 전진한다. skip은 후보·이미지가 없으므로 목표 수량·NO·연속 기술 실패 카운트에 들어가지 않는다. 같은 원본의 연속 skip이 `max_consecutive_sampling_skips`에 닿으면 그 원본을 `source_no_region`(`held_reason: sampling_exhausted`)으로 보류하고, 재개 때도 보류를 유지한다. 계획이 만들어지면 연속 카운트는 0으로 돌아간다. report.json에 `sampling_skipped`를 남긴다.
- v1의 rotation anchor는 사각형 좌상단이라 물리적 의미가 없었다. v2는 `operation_params`에 `anchor_method`, `contact_dilation_px`, 좌표계(`normalized_original_px`)를 기록한다.

한 원본에서 선택 가능한 (proposal, operation) 조합이 없으면 `source_no_region`으로 보류하고 경고를 남긴다. 모든 원본이 보류되면 기존 `no_eligible_sources`로 종료한다. 기하 영역으로 대체하지 않는다.

## 4. 저장과 재현성

```text
runs/<run_id>/proposals/<source_id>/
├── proposals.json      # 계약 아래. 마지막에 원자적으로 기록(commit 표시)
├── entity_000.png      # 0/255 L-mode, 원본 해상도, 구멍 없는 단일 연결 영역
├── part_000_003.png    # entity 000의 part 003
└── receipt/            # worker raw 출력: raw.json, raw_masks.npz, response.json
```

`proposals.json` (schema `1.2`. T012에서 `parts`와 part 출처를 추가해 `1.1`, T015에서 `parts`가 profile v4와 같은 subject별 목록이 되어 `1.2`. `1.0`·`1.1`은 읽지 않는다):

```json
{
  "schema_version": "1.2",
  "source_id": "src_53b5fe1e6b038249",
  "source_pixel_sha256": "<원본 정규화 픽셀 hash>",
  "size": [120, 80],
  "backend": "sam3",
  "models": {"facebook/sam3": "<HF revision>", "facebookresearch/sam3": "<git commit>"},
  "settings": {"backend": "sam3", "min_entity_score": 0.5, "...": "run의 region_proposal 섹션 전체"},
  "source_profile_sha256": "<고정된 SourceProfile의 frozen_sha256>",
  "subjects": ["kitten"],
  "phrases": ["kitten"],
  "parts": [{"subject": "kitten", "parts": ["tail", "front leg"]}],
  "status": "ready",
  "entities": [
    {
      "proposal_id": "entity_000", "phrase": "kitten", "score": 0.9,
      "bbox": [20, 20, 59, 59], "area_fraction": 0.1667,
      "mask_path": "entity_000.png", "mask_sha256": "<file sha256>",
      "parts": [
        {"proposal_id": "part_000_000", "source": "text", "phrase": "kitten tail", "score": 0.81,
         "bbox": [20, 40, 35, 59], "area_fraction_of_entity": 0.2,
         "mask_path": "part_000_000.png", "mask_sha256": "<file sha256>"},
        {"proposal_id": "part_000_001", "source": "point", "point": [50, 30], "score": 0.95,
         "bbox": [40, 20, 59, 39], "area_fraction_of_entity": 0.25,
         "mask_path": "part_000_001.png", "mask_sha256": "<file sha256>"}
      ]
    }
  ],
  "rejected": {"entity_score": 1, "part_area": 2, "text_part_containment": 1},
  "created_at": "2026-09-30T00:00:00+00:00"
}
```

로더(`load_proposals`)가 거부하는 경우: 깨진 JSON이나 객체가 아닌 최상위·항목(`DataError`), schema·source ID·원본 픽셀 hash·해상도 불일치, 허용되지 않은 backend(`fixture`는 fake 생성 backend의 run에서만 허용), 비어 있는 모델 revision, run의 `region_proposal` 설정과 다른 `settings`, 고정 profile과 다른 `source_profile_sha256`·`subjects`·`parts`(`text_parts`가 꺼져 있으면 빈 목록이어야 한다), subject별 형식이 아니거나 profile `subjects`에 없는·중복된 subject를 가진 `parts`, `subjects`와 다른(순서 포함) `phrases`, part의 `source`가 `text`/`point`가 아님, text part의 `phrase`가 그 entity 명사구와 **그 subject의** `parts`·`text_part_forms`로 만들 수 있는 명사구가 아님(다른 subject의 부위 이름이면 거부), point part의 `point`가 정수 `[x, y]`가 아니거나 이미지·그 entity mask 밖, text part가 point part 뒤에 옴, `suppress_point_parts_with_text`가 켜져 있는데 한 entity에 text part와 point part가 함께 있음, `status`와 entity 목록 불일치, ID·파일명 규칙 위반, mask hash 변경, binary 0/255·L-mode·원본 크기 위반, 여러 연결 성분이나 구멍, bbox 불일치, part가 entity 밖으로 나가거나 entity와 같음, 목록에 없는 phrase, 설정 하한 미만 점수(text part는 `min_text_part_score`, point part는 `min_part_score`), mask와 다르거나 설정 범위 밖인 `area_fraction`/`area_fraction_of_entity`, `dedupe_iou` 이상 겹치는 entity(또는 같은 entity의 part), 같은 entity 안의 포함 중복 part. T013의 새 키는 `settings`에만 들어가므로 T012에 쓴 파일은 설정 불일치로 거부된다. T015에서 schema를 `1.2`로 올렸으므로 `v2_pilot_001`의 proposals.json(`1.1`)은 새 코드에서 읽지 않는다(그 run은 implementation hash 때문에 어차피 재개할 수 없다).

- revision 규칙: 설정 템플릿의 `region_proposal.model_revision: null`은 허용한다(다른 모델 revision과 같이 실행 전에 채운다). 실제 backend(`sam3`, `grounding_dino_sam2.1`)의 proposal을 읽을 때는 `model_revision`이 비어 있지 않은 문자열이고 `models[model_id]`와 같아야 한다. `fixture`는 테스트 전용이라 이 대조를 하지 않는다.
- proposal은 profile 고정 **후에** 만든다. `write_proposals`는 고정된 profile record를 받아 `phrases`가 `subjects`와, `parts`가 profile의 subject별 `parts`(`text_parts`가 꺼져 있으면 빈 목록)와 다르면 거부한다. worker도 요청의 `parts`가 subject별 형식이 아니거나 `subjects`에 없는 subject를 담으면 `DataError`로 거부한다.
- 계획의 `region_phrase`는 지금처럼 entity 명사구다. part의 출처와 부위 명사구는 계획의 `region_proposal_id`로 proposals.json에서 찾는다(EditRequest schema는 바꾸지 않았다).

- proposal hash는 `created_at`을 뺀 `proposals.json` 내용의 digest이며 mask 파일 hash를 포함한다. pipeline은 첫 로드 때 `state.json`의 `region_proposals`에 hash·상태·조합 수를 고정하고, 후보 record에도 `region_proposals_sha256`을 남긴다. 재개 때 둘 중 하나라도 다르면 "Region proposals changed; start a new run"으로 멈춘다.
- 모델·임계값·명사구·prompt가 바뀌면 새 run이다.
- `EditRequest` schema `1.2`에 `region_proposal_id`, `region_level`(`entity`/`part`), `region_phrase`를, `1.3`에 `region_select_point`를 추가했다. v1 계획은 네 값이 `null`이고, 저장된 schema `1.1`·`1.2` v1 계획도 그대로 읽힌다. `region_select_point`가 없는 `1.2` v2 계획(수정 전 코드 산출물)은 `validate_plan`이 거부한다. 그런 run은 implementation hash가 달라 어차피 재개할 수 없다. `region_mask_path`는 선택한 proposal mask의 바이트 사본이라 `mask_sha256`이 proposal의 hash와 같다.
- `validate_plan`은 v2 계획의 mask가 단일 연결 영역인지, `region_select_point`가 mask 안인지, `source_point`가 mask centroid와 같은지 다시 확인한다.
- proposal mask는 **편집 영역**이며 segmentation GT가 아니다. export에 라벨로 승계하지 않는다.
- 시각화 footer에 `region=<level> | phrase=<phrase> | proposal=<id>` 줄과 `S = mask centroid ... | contour select=(x, y)` 줄을 추가한다(v2 계획일 때만). S는 실제 드래그 시작점(centroid)이며 오목한 mask에서는 파란 영역 밖일 수 있다. area는 기존처럼 mask의 실제 모양을 그린다. report.json에는 `region_proposals`(원본별 상태·hash·조합 수)를 넣는다.

## 5. 환경과 GPU

- **SAM 3 (기본)**: venv `.venv-sam3`(Python 3.12, torch 2.10.0+cu128, 공식 `facebookresearch/sam3` commit `2345a4a` editable, `numpy<2`, `setuptools<82`, sam3가 선언하지 않았지만 import하는 `einops`·`pycocotools`·`psutil`). 설치 절차는 [README](../README.md#sam-3-proposal-환경-object_region_v2)와 `environments/sam3-requirements.txt`, 실제 버전은 `environments/sam3.freeze.txt`. T009c 실측: 드라이버 535.288.01(CUDA 12.2)에서 cu128 wheel이 동작했고, RTX A6000 한 장에서 960px 이미지 기준 피크 약 6.2 GiB(`nvidia-smi`), 모델 로드 8.4 s였다.
- **Grounding DINO + SAM 2.1 (비교)**: 별도 `.venv-region`(Python 3.10, torch 2.5.1, 공식 `sam2` commit 고정, Grounding DINO는 Transformers 포트).
- 어느 쪽도 DragFlow·Qwen 환경과 합치지 않는다. worker Python은 `execution.proposal_python`으로 전달한다.
- proposal worker는 선택된 GPU 목록의 첫 GPU에서 순차 실행하고, 끝나면 종료한 뒤 DragFlow를 올린다. 다른 사용자가 점유한 GPU를 쓰지 않는 기존 규칙을 따른다.
- 가중치는 `advv prepare --only sam3`가 `configs/upstream.lock.json`의 `region_proposal.sam3`(HF revision, 파일별 크기·SHA256)대로 `models/facebook__sam3/<revision>/`에 받는다. 이미 있는 파일은 다시 받지 않고 SHA256만 검증하며, 불일치 파일은 지우지 않고 오류로 멈춘다. 결과는 기존 모델과 같이 `models/weights.lock.json`의 `models["facebook/sam3"]`(revision, path, files)에 기록한다. gated 모델이라 `--only all`(기본)에는 포함하지 않는다. 공식 코드는 `checkpoint_path`를 주지 않으면 revision 없이 내려받으므로 반드시 로컬 경로를 넘긴다.
- preflight(v2일 때): backend가 `sam3`인지, weights lock의 `facebook/sam3` 항목과 `model_path`·`model_revision`이 같고 파일 크기가 맞는지, `repo_path` checkout이 `code_revision`이고 tracked 파일이 수정되지 않았는지, `proposal_python`이 첫 선택 GPU만 보이는 상태로 `torch`·`sam3`·`advv`를 import하고 그 `sam3`가 `repo_path`의 것인지 확인하고 pip freeze를 run 설정에 고정한다.

## 6. 설정

`configs/advv.example.yaml`에 아래 키가 있고 `config.py`가 검증한다. 예시 파일의 `sampler.version`은 `random_geometry_v1`이며, v1에서는 `object_region`·`region_proposal`을 검증만 하고 쓰지 않는다. 두 섹션이 없는 예전 설정도 v1에서는 유효하고, v1에서는 나중에 추가된 `max_consecutive_sampling_skips`와 `rotation_max_error_degrees`·`rotation_max_error_fraction`·`rotation_min_executed_degrees` 묶음(T015, 최소 실효각은 T015-fix)과 `text_parts`·`text_part_forms`·`min_text_part_score`·`text_part_containment` 묶음과 `part_dedupe_containment`·`part_dedupe_area_ratio`·`suppress_point_parts_with_text` 묶음이 없어도 된다(묶음 안에서 하나라도 있으면 그 묶음을 모두 검사한다). v2에서는 모두 필수다.

```yaml
source_profile:
  prompt_path: ../prompts/source_profile_v4.txt          # v2 + text_parts에 필요 (subject별 parts)
  response_schema: ../schemas/source_profile_v4.schema.json

sampler:
  version: object_region_v2
  operations: [relocation, deformation, rotation]
  operation_distribution: uniform
  displacement_diagonal_fraction: [0.02, 0.20]
  rotation_degrees: [-30.0, 30.0]
  object_region:
    granularity:              # 허용: relocation ⊆ [entity], rotation ⊆ [part], deformation ⊆ [part, entity]
      relocation: [entity]
      rotation: [part]
      deformation: [part, entity]
    rotation_anchor: part_entity_contact
    contact_dilation_px: 3    # 1–64
    on_no_region: hold_source # 다른 값은 거부(fallback 없음)
    max_consecutive_sampling_skips: 8  # 1–1000. 연속 sampling_skipped가 닿으면 source_no_region 보류
    rotation_max_error_degrees: 3.0    # (0, 180]. 실효 각도 허용 오차 = max(이 값,
    rotation_max_error_fraction: 0.30  # [0, 1]   이 비율 × |요청각|) (T015)
    rotation_min_executed_degrees: 2.0 # (0, 180]. 목표 칸의 실효각은 요청각과 같은 부호, |실효각| ≥ 이 값 (T015-fix)

execution:
  proposal_python: null       # ../.venv-sam3/bin/python

region_proposal:
  backend: sam3               # 비교: grounding_dino_sam2.1 (미구현, 실행 시 오류)
  model_id: facebook/sam3
  model_path: null            # sam3.pt가 있는 폴더 (models/facebook__sam3/<revision>)
  model_revision: null        # 3c879f39826c281e95690f02c7821c4de09afae7
  repo_path: ../third_party/sam3
  code_revision: null         # 2345a4ad109ac29c569da749c91d84f10dc08c40
  min_entity_score: 0.50
  min_part_score: 0.80
  entity_area_fraction: [0.02, 0.60]
  part_area_fraction_of_entity: [0.02, 0.70]
  dedupe_iou: 0.85
  part_dedupe_containment: 0.95  # (0, 1]
  part_dedupe_area_ratio: 0.80   # (0, 1]
  part_points_per_entity: 16  # 1–256
  text_parts: true            # profile parts로 텍스트 prompt를 점 prompt와 병행
  text_part_forms: [subject_part, part]  # 비어 있지 않고 중복 없는 부분집합
  min_text_part_score: 0.50   # [0, 1]
  text_part_containment: 0.85 # (0, 1]
  suppress_point_parts_with_text: true  # bool
  grounding_dino:             # grounding_dino_sam2.1일 때 필수
    box_threshold: 0.35
    text_threshold: 0.25
```

## 7. 검증

CPU 테스트(`tests/test_region_sampler.py`, `tests/test_proposal_worker.py`, `tests/test_text_parts.py`, `tests/test_part_filters.py`, `tests/test_subject_parts.py`, `tests/test_rotation_accuracy.py`, `tests/test_prepare.py`, fixture mask·가짜 SAM 3, 가중치 없음) — 구현됨:

- 같은 seed·source·attempt에서 같은 proposal·operation·점(다른 run 디렉터리에서도 동일). 16개 attempt의 v2 golden(`tests/fixtures/sampler_v2_golden.json`, 세 operation 포함)과 run seed를 바꾸면 계획이 달라지는지.
- relocation은 entity만, rotation은 part만, deformation은 둘 다. 경계가 없는 part는 rotation 제외.
- anchor가 part–entity 접촉 영역 근처(k px 이내), `source_point` = mask centroid, `region_select_point`는 mask 내부. relocation 후 entity bbox가 이미지 안이고 target − centroid = 기록한 shift.
- C자형 entity(centroid가 mask 밖): 시작점은 centroid, `instruction.json`의 `centroids[0]`은 선택점.
- 가로 전체 entity(QA 재현): relocation 200개 attempt와 N=40 run이 멈추지 않음. 강제 skip의 cursor 전진·재개, 연속 skip 보류와 재개 후 유지.
- profile hash·subjects·phrases 불일치 거부, 실제 backend의 revision null/불일치 거부, 점수·면적·중복·항목 타입 재검사, 깨진 JSON, v3 subjects 패턴.
- 필터 경계값(점수·entity 면적·part 면적·중복 IoU), 최대 성분·구멍 채움.
- 저장 계약: 왕복, fixture backend 거부, 설정·원본 hash 불일치, 다중 성분, part가 entity 밖, mask 변경.
- proposal이 모두 걸러지면 `source_no_region` 보류 → `no_eligible_sources`, 생성·edit_regions 없음(fallback 없음). proposal 파일이 없으면 `region_proposals_missing` 오류.
- proposal 내용을 바꾼 뒤 재개하면 거부. v1 계획은 변경 전 코드로 만든 golden 값과 같음.
- grid 축소: 2 px 막대가 격자 위상에 따라 남거나 사라짐, QA2의 3 px 막대(`grid_empty`)·얇은 꼬리 달린 blob(`grid_centroid_shift`) 제외, rotation 반지름 하한과 `grid_rotation_degrees` 기록, 목표 칸이 `upstream_grid_start`를 피함. v2 golden은 T009b에서 이 기록 필드와 fixture part(entity 왼쪽 절반, 반지름 5칸)로 다시 만들었다.
- worker: 가짜 SAM 3로 별도 프로세스 → raw receipt → proposals.json → 계획까지(`backend=fixture`), 원본당 1회 생성과 생성 후 worker 종료, 완료 receipt 재사용, 로드 실패·CUDA 없음·예외(치명)·timeout(재시도 가능)·비교 backend 미구현·`proposal_python` 누락, 생성 실패는 `region_proposals_missing`이고 재개 시 생성, 모두 걸러지면 `source_no_region`.
- prepare: 없는 파일만 받기, 있는 파일은 SHA256 검증, 불일치 파일 보존, weights lock·설정 대조.
- text part(`tests/test_text_parts.py`, 가짜 SAM 3의 `kitten tail`·`tail`·`kitten head` 응답 포함): 명사구 형태·순서, 명사구당 1회 질의와 여러 entity 귀속, 포함 비율·점수 경계값, text 우선 정렬과 출처별 거부 수, `text_parts` 꺼짐(질의 없음, 파이프라인이 빈 목록 전달), 저장·로드 왕복과 출처·명사구·점·순서·profile parts 재검사, 설정 검증, profile schema의 `parts` 형식, processor 임계값 여유(float32·bf16 간격).
- T015 subject별 parts(`tests/test_subject_parts.py`): entity가 자기 subject의 부위만 묻고 다른 subject 부위 prompt(`tree front bumper`)가 나가지 않음, 항목 없는 subject는 부위 없음, 두 subject가 공유한 부위 이름은 한 번만 질의, v4 schema 경계(flat 목록·키 누락/추가·대문자·중복·개수 상한)와 prompt 예시, profile의 미지·중복 subject는 `source_error`, 파이프라인의 subject별 전달·고정, 로더가 다른 subject의 부위 명사구를 거부, worker 요청 검사. 가짜 SAM 3 worker 테스트에서도 box entity에 kitten 부위를 묻지 않는다.
- T015 rotation(`tests/test_rotation_accuracy.py`): `v2_pilot_001` 계획 수치로 이전 실효 −17.74° 재현과 새 목표 (439, 542)·−10.49°, 선택 칸이 3×3 후보 중 오차 최소, 허용 오차 경계(같으면 통과, 바로 아래면 실패, 비율 쪽), 표본 계획이 허용 오차 안이고 요청·실효 각도를 모두 기록, 허용 오차를 아주 작게 하면 rotation 정적 제외, 설정 검증. v2 golden은 rotation 계획 4개에 `rotation_tolerance_degrees`가 더해지고 그중 2개(10, 14번)의 목표점이 바뀌었다(실효 오차 1.66°→1.07°, 3.27°→1.50°). 다른 operation 계획과 v1 golden은 그대로다.
- T015-fix(`tests/test_rotation_accuracy.py`, `tests/test_subject_parts.py`, `tests/test_text_parts.py`): QA M1 `front bumper` 요청 −2.2°가 실효 0° 칸 (577, 553) 대신 −4.76°를 고름, 최소 실효각을 6°로 올리면 이 각도는 불가, 두 계획 수치에서 ±2–30° 0.05° 스윕의 채택 실효각이 모두 같은 부호·≥ 2°, `rotation_min_executed_degrees` 설정 경계와 v1 묶음 규칙, 1000×750의 x = 350이 float32로 칸 116(v1은 115)이고 `_cell_pixel`·`valid_shifts`가 같은 반올림을 씀, 해시 불가 parts subject가 `TypeError` 대신 오류 문구, v2 profile(subjects 없음)의 config 문구. v1·v2 golden은 그대로다.
- T013 part 규칙(`tests/test_part_filters.py`): 포함 중복의 포함 비율·면적비 경계값과 대칭성, 계층(면적비 0.5) 유지, 정렬에서 앞선 쪽이 남음(text·point 모두), 면적 하한 0.02 경계, point 억제(text 채택 시 억제, text가 모두 걸러짐·`text_parts` 꺼짐·빈 `parts`·설정 꺼짐이면 사용, entity별), 로더의 포함 중복·억제 위반·점 좌표 범위 거부, 깨진 raw 입력의 `DataError`, 설정 검증, 가짜 processor의 reset 강제. 의미 VQA context(`tests/test_semantic_context.py`).

GPU 통합 확인 (`tests/test_sam3_integration.py`는 `-m integration`; T009d에서 T012 이전 adapter로, T012에서 텍스트 part adapter로 5장 실행; `assets/` 샘플 사진). part 이름은 `ADVV_SAM3_PARTS`(쉼표 구분, 빈 값이면 point part만)로 넘긴다:

- 원본별 entity·part 수(출처별), 실패 원본, proposal 시간과 VRAM.
- text part: `subject_part`와 `part` 형태별 점수·포함 비율 분포, `min_text_part_score` 0.5와 `text_part_containment` 0.85가 적절한지.
- 균열·사고 차량처럼 명사구 grounding이 약할 수 있는 도메인의 실패 사례 기록.
- v1과 v2의 소규모 N 비교: 물리/의미 VQA 통과율, 사람 검수로 본 이음매·잘린 객체 비율.
- grounding 비교: SAM 3 vs Grounding DINO + SAM 2.1 (선택: Qwen3.5 box, 출력 좌표 형식을 먼저 실측).

## 8. 남은 결정

- 결함 도메인에서 편집 대상이 결함 자체인지, 결함을 가진 객체인지. 초기에는 profile `subjects`에 따르고 평가 후 정한다.
- 한 원본에 entity가 여러 개일 때 선택 확률(균등 vs 면적 가중). 초기값은 균등.
- 필터 초기값(특히 `min_entity_score`, part 면적 범위, `min_text_part_score`, `text_part_containment`, `part_dedupe_containment`·`part_dedupe_area_ratio`)은 실제 proposal 분포를 본 뒤 조정한다. 조정은 새 run에서만 한다. T012 receipt 재build(T013) 결과, 포함 중복 0.95/0.80에서는 실측 5장의 포함 쌍(면적비 0.24–0.78)이 하나도 합쳐지지 않았다: 사람 "바지 다리" vs "바지+신발 다리"(0.73/0.68), 굴착기 붐 vs 붐+암(0.78)은 모두 남는다. T014 사용자 결정: 포함 중복 0.95/0.80은 그대로 둔다(계층 부위를 별도 편집 대상으로 유지). 한계: "바지 다리" vs "바지+신발 다리"처럼 사실상 같은 부위의 쌍도 남아 비슷한 영역이 두 번 뽑힐 수 있다.
- T014: `text_part_containment` 0.90 → 0.85(사용자 결정). T012 receipt 기준 굴착기 crawler track(0.865/0.888)과 눈표범 `front leg` 한 인스턴스(0.878)가 새로 통과 대상이 되고, entity 밖 바퀴(0.0)와 `front leg` 0.754는 여전히 거부된다. T014 GPU9 재실행(5장)에서 crawler track은 text part로 채택(rotation 가능)됐고 눈표범 `front leg`는 `snow leopard front leg`에 포함된 계층 part로 함께 남았다(part 28→30). 0.85–0.90 구간 표본은 3개뿐이다.
- `text_part_forms`의 두 형태 중 하나만 남길지는 T012 이후 GPU 실측으로 정한다.
- T015: Qwen이 v4 subject별 `parts`를 지시대로 내는지(특히 subject 철자 복사, 다른 subject 부위 누락)는 아직 GPU로 확인하지 않았다. rotation 허용 오차 3.0°·0.30은 미검증 초기값이며, 실효 각도는 계획 값이지 생성 이미지에서 측정한 각도가 아니다.
- T013: text part가 남은 entity에서는 point part를 쓰지 않는다(사용자 결정, `suppress_point_parts_with_text`). text part가 없는 entity의 point part `min_part_score`(0.80)는 그대로다.
