# 공식 근거와 연결 기록

확인일: **2026-09-29**. 링크의 `main` 내용은 바뀔 수 있다. `configs/upstream.lock.json`에 확인한 commit·가중치 revision을 고정했다. 이 문서는 설치·GPU 실행의 검증 보고서가 아니다.

## DragFlow

- 논문: [DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](https://arxiv.org/abs/2510.02253), arXiv:2510.02253. 확인한 문서 버전은 v3이다.
- [논문 본문](https://arxiv.org/html/2510.02253v3): FLUX 기반 영역 단위 편집, affine supervision, 배경 보존, subject consistency를 다룬다. ADVV의 생성 후 VQA 채택 정책은 이 프로젝트에서 추가하는 설계다. 논문의 intent 해석용 MLLM과 혼동하지 않는다.
- [공식 저장소](https://github.com/Edennnnnnnnnn/DragFlow): 설치·demo·benchmark 지침. dual 24 GB GPU 사용 설명이 있으며 ADVV 전체 장비 요구량을 뜻하지 않는다.
- [벤치마크 runner](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/bench_dragflow.py): Dragger 호출 흐름과 CLI 전역 동작 확인.
- [Dragger 구현](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/dragger.py): core class와 두 CUDA 장치 참조 확인.
- [데이터 로더·저장 함수](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/dashboard_utils.py): 입력 파일명, `region_operations`, 출력 저장·좌표 처리 확인.
- [모델 설정](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/config.yaml): `black-forest-labs/FLUX.1-dev`, InstantCharacter adapter 경로, SigLIP·DINOv2 encoder 설정 확인.
- [환경 파일](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/dependencies/dragflow.yaml): Python 3.10.16, torch 2.5.1, diffusers 0.32.2, transformers 4.48.0이 기재되어 있다. ADVV에서 검증한 lock은 아니다.

## Qwen

- [Qwen3.5-4B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-4B): 이미지·텍스트 입력을 지원하는 post-trained checkpoint와 로컬 실행 자료, thinking 비활성화 방법을 확인했다. 이 프로젝트의 기본 모델이다.
- [Qwen 공식 모델 목록](https://huggingface.co/Qwen/models): 2026-09-29 확인한 공식 범용 4B급 이미지 VQA 모델 중 Qwen3.5-4B를 선택했다. 모델 계열 번호가 더 높다는 이유만으로 4B 공개 checkpoint가 있다고 가정하지 않는다.
- [공식 저장소의 릴리스 이력](https://github.com/QwenLM/Qwen3.8): Qwen3.5의 소형 모델 공개일은 2026-03-02로 안내한다. 기존 `QwenLM/Qwen3.5` 주소가 새 계열 저장소로 redirect되어 구체적인 사용법은 선택 모델 카드와 revision을 기준으로 확인한다.

모델 카드와 라이브러리 버전별 예제 API가 다를 수 있으므로 구현에서는 선택한 버전에 맞는 하나의 경로를 검증하고 고정한다. 이미지 입력을 지원하는 다른 Qwen으로 바꾸면 동일한 prompt·parser 계약을 유지하되 별도 실험 revision을 부여한다.

추가 fine-tuning 없이 checkpoint로 VQA를 시작하는 것은 ADVV의 초기 설계 선택이다. 그것만으로 산업 결함 판정이나 희귀 생물 식별의 정확도가 입증되는 것은 아니다. Qwen3.5 지원 버전을 실제 검증하기 전 예전 Qwen3-VL용 환경 하한을 재사용하지 않는다.

## 확인된 사실과 ADVV의 설계 선택

| 구분 | 내용 |
| --- | --- |
| 공식 자료로 확인 | DragFlow의 영역 편집 방식, 실제 runner/core class, 모델 설정, 각 환경의 안내 버전 |
| 사용자 확정 | 전체 입력 총 N장, 자동 무작위 편집, 현실성·물리적 개연성과 원본 의미 보존 검증, 선택 설명, 총 상한 없는 반복, 로컬 4B급 Qwen, 실행 시 GPU 선택 |
| ADVV 설계 선택 | physical/semantic 별도 호출, 자동 원본 profile, geometry sampler, 엄격한 JSON, UNCERTAIN 분리, 기본 NO 삭제, 독립 worker |
| 실행 연결 확인 | 실제 DragFlow + Qwen N=1 생성·채택 및 해당 실행의 시간/메모리; [범위](RUN_VERIFICATION.md) 참조 |
| 아직 검증하지 않음 | 확대된 통합 처리량, 도메인별 판정 성능, 후속 성능 개선 |
| 실행 전 확정 | 사용자 목표 N, 도메인별 sampler 초기값 검증 |

## 남겨야 할 연결 기록

구현 시 `environments/`와 실행의 `environment.json`에 ADVV 코드 버전, DragFlow 및 submodule commit, patch hash, 모델·processor·adapter·encoder revision 또는 로컬 파일 hash, Python/PyTorch/CUDA/드라이버 및 패키지 버전을 남긴다. 링크만 존재하거나 다운로드에 성공한 것은 재현 완료의 근거가 아니다.

## 로컬 구현 기록 (2026-09-29)

- `configs/upstream.lock.json`: DragFlow/FireFlow commit 및 모델별 revision.
- `models/weights.lock.json`: FLUX.1-dev를 포함해 다운로드 완료한 로컬 snapshot. 서버 브라우저 인증 후 gated 모델 접근을 확인했다.
- `environments/*.freeze.txt`: 설치한 coordinator/DragFlow/Qwen 환경.
- `integration_checks/qwen_worker/results.json`: 실제 Qwen3.5-4B의 단일/두 이미지 추론, torch 2.6.0 / Transformers 5.17.0, 약 9.5 GB 이하의 관측 allocated VRAM. 다른 이미지 크기의 요구량을 보장하지 않는다.
- run `config.json`에 prompt/model/environment, `manifest.json`에 implementation hash, `state.json`에 실행 segment를 기록한다.
