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

## 2026-09-30 실제 N=1 실행

- `first_trial_001`에서 실제 DragFlow 생성과 Qwen 두 YES 판정, 고유 이미지 한 장의 저장·export·시각화를 확인했다.
- 위 초기 기록의 생성 미실행 상태 이후 진행된 결과다. 전체 operation의 품질, 실제 모델 중단·재개와 도메인 평가는 완료로 간주하지 않는다.
- 실행 파일은 로컬에 보관하며 Git에 포함하지 않는다. 수치와 범위는 [검증 기록](RUN_VERIFICATION.md)에 정리했다.
