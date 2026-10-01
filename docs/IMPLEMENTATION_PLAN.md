# 구현 단계와 완료 기준

확정 요구사항은 SPEC v0.4 기준이다. 코드 및 CPU·로컬 Qwen 검증을 마쳤으며, 실제 DragFlow와 Qwen으로 N=1 생성·검증·저장을 완료했다. 확대된 실제 통합 검증과 연구 평가는 아래 미완료 항목으로 남긴다. [검증 범위](RUN_VERIFICATION.md)를 참고한다. 목표는 **총 N장의 현실적이고 의미가 보존된 새 이미지를 자동 생성하는 파이프라인**이다.

## P0. 외부 모델 연결과 입력 계약

- [x] 사용자 예제 `assets/original_image.png` (528×598)를 연결하고 정규화·Qwen 입력·요청값 그림을 확인했다. 산업/사고/희귀 생물 대표 표본의 품질 평가는 별도로 필요하다.
- [x] DragFlow·submodule commit, FLUX/adapter/encoder revision을 고정한다.
- [x] 공식 demo와 instruction 구조를 확인하여 relocation/deformation/rotation에 필요한 mask·점·anchor·강도 매핑을 문서화한다.
- [x] Qwen3.5-4B와 processor를 로컬에 준비하고 별도 환경 버전을 고정한다. 추가 학습은 하지 않는다.
- [x] `--gpus`로 선택한 장치만 worker에 노출하는 preflight를 구현한다. 단일 GPU 지원 여부는 가정하지 않는다.
- [ ] 실행 범위가 주어졌을 때 공식 DragFlow demo와 Qwen의 한 장/두 장 이미지 입력을 각각 검증한다.

완료 기준: 실제 연결 지점, operation별 지원, 로컬 가중치·환경·장치 요구가 확인되었다. 실행하지 않은 추론은 검증 대기로 표시한다.

## P1. CPU 기반 상태·수량·sampler

- [x] 이미지 폴더를 manifest로 변환하고 input 중복·경로·split을 검사한다.
- [x] SourceProfile/EditRequest/CheckResult/Decision 계약과 엄격한 JSON 파서를 구현한다.
- [x] seed·source ID·attempt index에 따른 연결 영역·방향·operation sampler를 구현한다.
- [x] 총 N장 counter, 원본 round-robin, 무제한 반복, 사용자 중단·재개를 구현한다.
- [x] 원자적 저장, 단일 writer lock, NO 삭제, 중복 제외를 구현한다.
- [x] [시각화 명세](VISUALIZATION.md)의 start/end/area overlay와 원본·생성 결과 1×2 PNG 렌더러를 CPU에서 구현한다.
- [x] fake backend는 CPU fixture 전용으로 표시하고 실제 dataset export와 분리한다.

필수 CPU 검증:

| 사례 | 기대 결과 |
| --- | --- |
| physical=YES, semantic=YES | 두 결과가 유효할 때만 accepted |
| physical=YES, semantic=NO | rejected; 사고/결함 의미 변경을 통과시키지 않음 |
| physical=NO/UNCERTAIN | semantic=not_run; 각각 rejected/uncertain |
| physical=YES, semantic=UNCERTAIN/오류 | uncertain/verification_error, quota +0 |
| 소문자·추가/중복 key·잘린 JSON·빈 reason·code fence | verification_error |
| reason 또는 입력 prompt에만 YES가 있음 | answer를 대체하지 않음 |
| 동일 seed/source/attempt, 다른 worker 실행 순서 | 동일 계획·mask |
| 범위 밖 좌표·빈 mask·항등 변환·비지원 operation | backend 호출 전 실패 |
| 두 이미지의 순서나 원본/profile hash 변경 | 기존 semantic cache 재사용 금지 |
| 원본 10장, 목표 N=3 | 총 3장 저장; 원본별 3장으로 처리하지 않음 |
| 두 VQA YES이지만 원본/기존 후보와 동일 픽셀 | 중복 제외, quota 증가 없음 |
| N-1장에서 NO·오류·중복 후 YES | N장에서 성공 종료; 다음 생성 호출 없음 |
| NO가 긴 시퀀스로 반복됨 | 숨겨진 총 attempt cap 없이 유지; 테스트는 취소 이벤트로 종료 |
| 사용자 취소 후 같은 run 재개 | 확정 count/attempt 보존, 중복 채택·재추론 없음 |
| GPU ID만 바꿔 재개 | 허용, 새 execution segment 기록 |
| 모델·prompt·sampler·profile 변경 후 재개 | 기존 run에 섞기 금지 |
| NO record 저장 후 삭제 전에 중단 | 재개 시 삭제만 수행 |
| accepted 파일 저장 후 record 전에 중단 | hash/transaction 확인 후 복구; count 중복 없음 |
| 원본·다른 run·외부 symlink를 삭제 대상으로 지정 | 삭제 차단 |
| 원본 profile 모두 uncertain/error | no_eligible_sources, 성공 아님 |
| val/test 입력 또는 split 간 group/내용 중복 | 생성 제외 또는 preflight 실패 |
| 직사각형 원본의 리사이즈·패딩·EXIF 회전 | start/end/area에 같은 좌표 변환 적용 |
| backend가 점·mask를 반올림/변환함 | 요청값과 실제 전달값을 구분하여 표시 |
| 가까운 점·가장자리 점·작은 영역·anchor | 기호·좌표·범례가 가려지지 않음 |
| 시각화 전후 실제 이미지·VQA 입력 | 파일/픽셀 hash 동일; overlay가 검증 입력에 섞이지 않음 |
| NO 후보의 비교 PNG·썸네일이 존재 | 생성 픽셀 포함 복사본 삭제, 원본 설정 PNG 보존 |
| N장 달성 후 시각화만 실패하여 재개 | 렌더링만 복구; 추가 생성·VQA·quota 증가 없음 |

완료 기준: 실제 가중치·네트워크·CUDA 없이 반복과 상태를 검증한다. 무제한 반복 테스트에 실제 무한 실행을 사용하지 않는다.

## P2. 원본 의미 추출과 실제 DragFlow 생성

- [x] Qwen으로 원본의 summary/must_preserve/uncertain을 추출하고 생성 전에 고정한다.
- [x] 사용자 선택 설명을 context로 연결하되 자동 추정 내용과 출처를 구분한다.
- [x] sampler 계획을 upstream instruction으로 변환하고 실제 편집을 수행한다. N=1 rotation 사례 확인.
- [x] 원본 정규화·mask·좌표·prompt·seed·계획·출력 hash를 추적한다.
- [x] 산업 손상/사고를 정상화하는 target prompt가 자동 삽입되지 않는지 확인한다.
- [ ] supported operation의 실제 출력과 편집 적용 여부를 원본과 비교한다.
- [x] 실제 backend에 전달된 점·mask·전처리 정보를 저장하고 이를 사용한 시각화의 좌표 정합성을 확인한다. N=1 사례 범위.

완료 기준: 수동 mask·점·GT 라벨 없이 입력 이미지에서 자동 계획과 실제 DragFlow 후보를 만든다. 수동 replay는 디버깅 수단이며 MVP 자동화의 완료 증거를 대체하지 않는다.

## P2b. 객체 중심 영역 sampler v2 (SAM 3 worker 연결, GPU 통합 미검증)

[설계 문서](REGION_SAMPLER.md) 기준. 기본 proposal 모델은 SAM 3, 비교용은 Grounding DINO + SAM 2.1.

- [x] `.venv-sam3`(Python 3.12, torch 2.10.0+cu128, 공식 sam3 commit `2345a4a`)을 만들고(T009a) commit·HF revision·파일 SHA256을 `upstream.lock.json`에 추가했다(T009b). `advv prepare --only sam3`, preflight 검사. GPU7 스모크(T009c): 드라이버 535에서 cu128 동작, 피크 약 6.2 GiB.
- [ ] 비교용 `.venv-region`(Python 3.10, torch 2.5.1, 공식 sam2 commit, Transformers Grounding DINO).
- [x] `source_profile_v3`에 `subjects` 명사구를 추가한다. v2 선택 시 config가 요구한다. 실제 Qwen 출력 형식은 미확인.
- [x] T012: `source_profile_v3`에 `parts`(부위 이름 0–5개)를 추가하고, SAM 3 worker가 부위 이름 텍스트 prompt(`"<subject> <part>"`, `"<part>"`)를 점 prompt와 병행한다. entity 포함 비율로 귀속, part 출처(`text`/`point`) 기록, proposals.json·raw receipt schema `1.1`, 설정 `text_parts`·`text_part_forms`·`min_text_part_score`·`text_part_containment`. SAM 3 processor 임계값을 ADVV 하한 −0.01로 바꿔 동점 처리 불일치(T009b QA L1)를 없앴다. 실제 Qwen의 `parts` 출력과 GPU 결과는 미확인.
- [x] proposal 저장 계약·검증 로더, mask 정리·필터·중복 제거·part 점 선택(CPU, `proposals.py`).
- [x] RegionProposal worker: SAM 3 텍스트 → entity mask, 점 → part multimask → raw receipt → coordinator의 `build_proposals`/`write_proposals`, pipeline에서 profile 고정 후 원본마다 1회 호출하고 DragFlow 전에 종료(T009b). 가짜 SAM 3 패키지로 프로세스 계약·오류 경로 CPU 테스트.
- [x] T013: part 포함 중복(`part_dedupe_containment` 0.95·`part_dedupe_area_ratio` 0.80, 미검증), part 면적 하한 0.05 → 0.02, text part가 남은 entity의 point part 억제(`suppress_point_parts_with_text`, 기본 on), 의미 VQA context를 profile `summary`·`must_preserve`·`uncertain`으로 축소, v3 prompt에 "증거를 담은 부위 제외" 추가, 로더 점 좌표 범위·raw 필드 누락 `DataError`·가짜 SAM 3 reset 강제(T012 QA M1, L1–L4).
- [x] T014(사용자 결정): `text_part_containment` 0.90 → 0.85(T012 실측 굴착기 crawler track 0.865/0.888, entity 밖 바퀴 0.0). 포함 중복 0.95/0.80은 유지하며 "바지 다리" vs "바지+신발 다리" 같은 거의 같은 쌍이 남는 한계를 [설계 §8](REGION_SAMPLER.md#8-남은-결정)에 기록. T014 GPU9 재실행(5장)에서 crawler track이 text part로 채택(containment 0.865, rotation 가능)됐고 part 28→30, rotation 가능 24→26. 0.85–0.90 구간 표본은 3개뿐이라 entity 밖 객체 유입 가능성은 더 큰 표본으로 확인해야 한다.
- [ ] ADVV adapter를 통한 실제 SAM 3 GPU 실행(`tests/test_sam3_integration.py`, `-m integration`)과 v2 `advv run`(DragFlow/Qwen 환경 필요, T005). T009d(GPU9)에서 T012 이전 adapter로, T012에서 텍스트 part adapter로 `assets/` 5장을 실행했다. T013 규칙은 T012 receipt를 CPU로 다시 build해 확인했다(adapter 변경 없음). v2 `advv run`과 실제 Qwen profile `parts`는 미실행.
- [x] QA2 N1–N4: feature grid에서 사라지거나 시작점이 한 칸 넘게 달라지는 얇은 proposal 제외, rotation grid 반지름 하한(4칸)과 실효 각도 기록, v1 설정의 `max_consecutive_sampling_skips` 선택화, subjects 숫자 시작 단어 허용.
- [x] `object_region_v2` sampler: operation별 granularity, part–entity 접촉 anchor, `source_no_region` 보류, EditRequest schema `1.3`(centroid 시작점 + 윤곽 선택점), 허용 이동 집합 직접 추출, `sampling_skipped`, 시각화 footer.
- [x] CPU 테스트: 결정론, granularity 매핑, anchor 위치, 필터 경계값, fallback 금지, proposal hash 변경 재개 거부, v1 golden 회귀, v2 golden·seed 민감도, 오목 mask centroid, 가로 전체 entity relocation, skip 재개·보류, profile 연결, revision 고정, 로더 재검사.
- [ ] `assets/` 샘플로 proposal 품질·실패 사례를 확인하고 v1과 소규모 N의 VQA 통과율·이음매 비율을 비교한다.
- [ ] grounding 비교: SAM 3 vs Grounding DINO + SAM 2.1 (선택: Qwen3.5 box, 좌표 형식 실측).

## P3. 두 VQA와 목표 수량 반복

- [x] candidate-only physical, source+candidate semantic 호출을 별도로 연결한다.
- [x] non-thinking template, completion 추출, parser, raw response 기록을 구현한다.
- [x] 각 판정의 completed/error/not_run 상태와 최종 AND 결합을 구현한다.
- [ ] 생성/검증 GPU 생명주기를 반복 루프와 연결하고 사용자 선택 밖의 장치를 쓰지 않는지 확인한다.
- [ ] 실제 소규모 N을 지정하여 목표 수량 달성·NO 제거·중단·재개를 검증한다.

완료 기준: 실제 두 모델로 unique accepted N장을 만들고 계보와 두 판정을 추적한다. 통합 테스트에 시간 제한을 둘 수 있으나 production 총 실행 정책과 구분한다. 모델이 반드시 NO를 출력해야 하는 테스트는 두지 않고 결정론적 fixture로 분기를 검증한다.

## P4. 이미지 데이터셋과 품질 평가

- [x] 확정/부분 export manifest를 만들고 성공/중단/실패를 구분한다.
- [x] accepted/N, attempts, 두 check별 탈락, 중복, 원본별 기여, 시간/VRAM을 보고한다.
- [x] report에서 후보별 drag_plan/comparison 파일을 연결하고 삭제된 비교 이미지는 링크에서 제외한다.
- [ ] [평가 계획](EVALUATION.md)에 따라 소규모 독립 검수 표본으로 현실성·의미 보존·다양성을 확인한다.
- [x] 실제 설치·실행·GPU 선택·재개 예시와 한계를 README에 갱신한다.

완료 기준: 사용자가 이미지와 N을 넣고 GT 없이 파이프라인을 사용할 수 있다. 품질 검증용 사람 판정과 모델 학습용 GT를 구분한다.

## P4b. 사람 최종 검수

- [x] run별 on/off: 설정 `human_review.enabled`(기본 false) 또는 `advv run --human-review`. 꺼지면 기존 Qwen 전용 파이프라인과 동일.
- [x] `advv review`: 두 VQA 통과 이미지를 ← pass / → fail로 판정, fail은 판정 저장 후 삭제하고 재생성하지 않는다.
- [x] `review/current.png` 한 파일을 갱신하는 원본·후보 비교 화면(VS Code 자동 갱신).
- [x] export는 사람 pass만 포함, report에 pass/fail/pending 집계, 검수 시작 후 resume 거부.
- [x] CPU 테스트: 판정·삭제·export 필터·중단 후 이어하기·resume 차단·hash 변경·삭제 실패 재시도·화살표 키 파싱.
- [ ] 실제 터미널(VS Code Remote + tmux)에서 실제 생성 이미지로 키 입력과 화면 갱신 확인.

## P5. 이후 연구: 후속 성능과 선택적 fine-tuning

- [ ] 필요해진 과제의 GT와 task-specific exporter를 준비한다.
- [ ] 증강 전 group split을 고정하고 공정한 비교군으로 후속 모델을 평가한다.
- [ ] Qwen의 반복적인 도메인 오류가 확인될 때만 별도 fine-tuning 실험을 설계한다.

P5는 1차 파이프라인의 완료 조건이 아니다. Qwen 추가 학습이 필요하다고 미리 가정하지 않는다.

## 초기 구현의 검증 기록 (2026-09-29)

- CPU: **65개 테스트 통과**. strict parser/AND 판정, round-robin 총량, source·accepted 중복, NO/UNCERTAIN 분기, bounded 기술 재시도, OOM, 중단·receipt 복구, 삭제 안전성, profile 보류, 시각화 복구, EXIF/좌표, venv 경로 검사.
- 실제 Qwen: `integration_checks/qwen_worker/results.json`에 profile 및 단일/두 이미지 결과와 VRAM·토큰·버전 기록. 같은 원본 두 장 비교는 의미 입력 경로 검사이며 증강 성공 사례가 아니다.
- 실제 DragFlow: core import와 adapter 구현, FLUX 다운로드 및 GPU 0–7 사전 검사까지 확인. 생성·effective geometry·성능·GPU 메모리 검증은 미실행.
- 요청값 그림: `integration_checks/edit_previews/visualizations/`, 생성 미실행 표기.
- 외부 replay 파일 import/sidecar/다중 후보 병렬화/GT annotation export는 현재 구현하지 않았다.

- 설치 후 검사: `integration_checks/flux_installation.json`의 FLUX 24개 파일 크기·공개 LFS SHA256 통과. `integration_checks/preflight_after_flux.json`의 모델/환경/예제 입력 및 GPU 0–7 사전 검사 ready.

## 2026-09-30 CLI 수량 입력

- `advv run N ...` 위치 인자를 추가하고 기존 `--target-count N`과 YAML 수량 입력을 유지했다.
- 잘못된 수량, 두 형식의 동시 지정, 재개 중 수량 입력은 모델·run 접근 전에 거부한다.
- CLI에서 설정 로드·실제 수량 반복·저장·재개까지 CPU fixture로 검사했다. NO와 중복을 제외하고 요청한 총량에서 종료하는지 확인했다. 전체 CPU 테스트 **79개 통과**, lint 통과. 이번 변경 검증에서 GPU 추론은 실행하지 않았다.

## 2026-09-30 사람 최종 검수

- 선택 단계인 `advv review`와 `human_review.py`를 추가했다(기본 꺼짐). 전체 CPU 테스트 **101개 통과**, lint 통과. 키 입력은 pipe로 넣은 escape sequence로 검사했고 실제 터미널과 실제 생성 이미지로는 아직 확인하지 않았다.
- 코드가 바뀌어 implementation hash가 달라졌으므로 이 변경 전에 만든 run은 `--resume`할 수 없다(기존 규칙).

## 2026-09-30 sampler v2 CPU 부분

- `proposals.py`(저장 계약·검증·후처리), `object_region_v2` sampler, `source_profile_v3`, 설정 검증, pipeline의 proposal 고정·`source_no_region` 보류, 시각화 footer를 추가했다. 기본 sampler는 v1 그대로다.
- proposal 파일이 없으면 `region_proposals_missing` 오류로 멈춘다. 생성 worker(SAM 3)는 T009에서 연결한다.
- 전체 CPU 테스트 **127개 통과**(신규 26), lint 통과. GPU·모델 추론은 실행하지 않았다.
- 코드가 바뀌어 implementation hash가 달라졌으므로 이전 run은 `--resume`할 수 없다.

## 2026-09-30 sampler v2 QA 수정 (T002-fix)

- 시작점을 upstream과 같은 mask centroid로 계산하고 윤곽 선택용 `region_select_point`를 분리했다(EditRequest `1.3`). relocation/deformation 이동량은 허용 정수 이동 집합에서 직접 뽑고, 유효 기하가 없는 조합은 정적으로 제외한다. 그래도 예산을 소진한 attempt는 `sampling_skipped`로 넘기고 연속 skip 상한에서 `source_no_region` 보류한다.
- proposals.json에 frozen profile hash와 `subjects`를 기록하고 phrases 일치를 검사한다. 실제 proposal backend는 고정된 `model_revision`을 요구한다. 로더가 점수·면적·중복·항목 타입을 다시 검사하고 깨진 JSON을 `DataError`로 거부한다. v3 schema의 subjects 패턴을 prompt(소문자)와 맞췄다.
- 전체 CPU 테스트 **136개 통과**(신규 9), lint 통과. GPU·모델 추론은 실행하지 않았다.
- `src/advv/*.py`가 바뀌어 implementation hash가 달라졌으므로 이전 run은 `--resume`할 수 없다.

## 2026-09-30 SAM 3 proposal worker 연결 (T009b)

- `backends/region.py`·`sam3.py`(worker `--kind proposal`), `proposals.py`의 `collect_raw`·raw receipt, pipeline의 proposal 생성(profile 고정 후 원본마다 1회, 기술 오류는 재시도 후 `region_proposals_missing`, 영역 없음은 `source_no_region`), `LocalBackend`의 단일 worker 생명주기, `prepare --only sam3`, preflight의 SAM 3 검사, `upstream.lock.json`의 `region_proposal.sam3`, 설정 `region_proposal.repo_path`·`code_revision`을 추가했다.
- QA2 N1–N4를 수정했다. v2 계획의 `operation_params`에 `upstream_grid_start`(모든 operation), `grid_rotation_degrees`·`grid_radius_cells`(rotation)가 추가되어 v2 golden을 다시 만들었다. v1 golden은 그대로다.
- 전체 CPU 테스트 **155개 통과**(신규 19, 통합 1개는 기본 제외), lint 통과. GPU·모델 추론·다운로드는 실행하지 않았다.
- `src/advv/*.py`가 바뀌어 implementation hash가 달라졌으므로 이전 run은 `--resume`할 수 없다.

## 2026-10-01 부위 이름 텍스트 part (T012)

- T009d GPU 실측에서 점 prompt part가 거의 나오지 않아(SAM 3가 점마다 entity 전체를 가장 확신) 사용자 결정(부위 이름 텍스트 + 점 병행)에 따라 profile `parts`와 텍스트 part를 추가했다. v3 prompt·schema는 실제 run에 쓰인 적이 없어 제자리 수정했다.
- text part와 point part는 같은 정리·면적·중복·grid 규칙을 거치고, 정렬은 text 먼저(척도가 다른 두 점수를 비교하지 않음)다. 로더는 출처·부위 명사구·점 좌표·순서·profile `parts`를 다시 검사한다. v1·v2 golden은 바뀌지 않았다.
- 전체 CPU 테스트 **173개 통과**(신규 18, 통합 1개는 기본 제외), lint 통과. GPU·모델 추론·다운로드는 실행하지 않았다.
- `src/advv/*.py`가 바뀌어 implementation hash가 달라졌으므로 이전 run은 `--resume`할 수 없다. proposals.json·raw receipt `1.0`도 읽지 않는다.

## 2026-10-01 part 규칙 조정과 의미 VQA context 축소 (T013)

- T012 GPU 실측(포함 관계 중복, 면적 하한 탈락, 사고 차량 point part의 손상 증거)과 사용자 결정에 따라 part 포함 중복, 면적 하한 0.02, text part가 남은 entity의 point part 억제를 추가했다. 새 설정 3개는 v2 필수, v1은 묶음 단위 선택.
- T012 QA M1: 의미 VQA의 `{{preservation_context}}`에 profile `summary`·`must_preserve`·`uncertain`만 넣는다. prompt 파일은 그대로이며, v2 profile(sampler v1) run의 semantic 입력은 바이트 단위로 같다(테스트). v3 profile run은 `subjects`·`parts`가 빠진다.
- T012 receipt 5장을 CPU로 다시 build: part 수 27 → 28, rotation 가능 22 → 24(이미지별 표는 `_workspace/T013_implementer_report.md`). 포함 중복 0.95/0.80은 실측 쌍을 합치지 않았다.
- 전체 CPU 테스트 **199개 통과**(신규 26, 통합 1개는 기본 제외), lint 통과. v1·v2 golden 불변. GPU·모델 추론·다운로드는 실행하지 않았다.
- `src/advv/*.py`가 바뀌어 implementation hash가 달라졌으므로 이전 run은 `--resume`할 수 없다. T012에 쓴 proposals.json은 settings 불일치로 읽지 않으며, raw receipt(`1.1`)는 그대로 다시 build할 수 있다.

## 2026-09-30 실제 N=1 실행

- `first_trial_001`에서 실제 DragFlow 생성과 Qwen 두 YES 판정, 고유 이미지 한 장의 저장·export·시각화를 확인했다.
- 위 초기 기록의 생성 미실행 상태 이후 진행된 결과다. 전체 operation의 품질, 실제 모델 중단·재개와 도메인 평가는 완료로 간주하지 않는다.
- 실행 파일은 로컬에 보관하며 Git에 포함하지 않는다. 수치와 범위는 [검증 기록](RUN_VERIFICATION.md)에 정리했다.
