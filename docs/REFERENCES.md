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

## DragFlow 가속 조사 요약 (T016, 2026-10-01)

조사 원문은 `_workspace/T016_researcher_dit_speedup.md`(로컬 작업 기록)이다. 아래 시간 비중은 T011 로그와 코드 루프에서 셈한 추정이며 profiler 측정 전이다.

- 시간 구성(960×720, T011): no_grad forward 1회 약 16.4 s, drag round 약 29 s × 70(TRANSPORT 50 + INTENSIFY 20) ≈ 34분, inversion 약 5.5분, sampling 약 11분. round가 약 2/3다.
- 병목 후보(코드로 확인, 지배 요인은 미측정): fp32 activation + quanto qint8 dequant 뒤 `torch.matmul`(fused kernel 없음), fp32라 FlashAttention 불가, IP-adapter processor의 attention 층마다 cuda:0↔cuda:1 왕복, KV hook의 pageable CPU↔GPU 복사, 호출마다 전 GPU를 동기화하는 `reclaim_memory`.
- 최적화 없는 drag 논문은 모두 다른 생성기라 DragFlow를 대체할 수 없다: [LazyDrag](https://arxiv.org/abs/2509.12203)(MM-DiT, TTO 없음, runtime 수치 없음), [FastDrag](https://arxiv.org/abs/2405.15769)(SD1.5, 3.12 s/point vs DragDiffusion 21.54 s), [RegionDrag](https://arxiv.org/abs/2407.18247), [InstantDrag](https://arxiv.org/abs/2409.08857), [LightningDrag](https://arxiv.org/abs/2405.13722), [Inpaint4Drag](https://arxiv.org/abs/2509.04582), [DragNoise](https://arxiv.org/abs/2404.01050), [GoodDrag](https://arxiv.org/abs/2404.07206), [StableDrag](https://arxiv.org/abs/2403.04437), [FlowOpt](https://arxiv.org/abs/2510.22010). DragFlow 논문([arXiv 2510.02253](https://arxiv.org/abs/2510.02253))에는 runtime 표가 없다. 논문은 INTENSIFY lr 1200, 코드는 1000이며 ADVV는 코드를 따른다.
- 추론 캐싱([TeaCache](https://arxiv.org/abs/2411.19108), [ToCa](https://arxiv.org/abs/2410.05317), [FORA](https://arxiv.org/abs/2407.01425), [Δ-DiT](https://arxiv.org/abs/2406.01125), [FasterCache](https://arxiv.org/abs/2410.19355))은 inversion KV capture와 충돌하거나 sampling 비중이 작아 후순위다. [FireFlow](https://arxiv.org/abs/2412.07517)는 이미 쓰고 있다.
- 정밀도: [PyTorch 2.5 TF32 설명](https://docs.pytorch.org/docs/2.5/notes/cuda.html)(matmul 기본 off, cuDNN 기본 on), [SDPA](https://docs.pytorch.org/docs/2.5/generated/torch.nn.functional.scaled_dot_product_attention.html)와 [FlashAttention](https://github.com/Dao-AILab/flash-attention)(fp16/bf16 전용), [quanto + diffusers](https://huggingface.co/blog/quanto-diffusers)(목적은 메모리, 지연은 비슷하거나 증가).
- 사용자 결정(2026-10-01): 0단계 profiler, 1단계 결과 동일 패치 묶음, 2단계 TF32를 진행한다. bf16·quanto 제거·round 축소는 보류. 구현은 [아키텍처의 가속 플래그](ARCHITECTURE.md#dragflow-실행-가속-플래그-t017), 측정은 아직이다.

## Qwen

- [Qwen3.5-4B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-4B): 이미지·텍스트 입력을 지원하는 post-trained checkpoint와 로컬 실행 자료, thinking 비활성화 방법을 확인했다. 이 프로젝트의 기본 모델이다.
- [Qwen 공식 모델 목록](https://huggingface.co/Qwen/models): 2026-09-29 확인한 공식 범용 4B급 이미지 VQA 모델 중 Qwen3.5-4B를 선택했다. 모델 계열 번호가 더 높다는 이유만으로 4B 공개 checkpoint가 있다고 가정하지 않는다.
- [공식 저장소의 릴리스 이력](https://github.com/QwenLM/Qwen3.8): Qwen3.5의 소형 모델 공개일은 2026-03-02로 안내한다. 기존 `QwenLM/Qwen3.5` 주소가 새 계열 저장소로 redirect되어 구체적인 사용법은 선택 모델 카드와 revision을 기준으로 확인한다.

모델 카드와 라이브러리 버전별 예제 API가 다를 수 있으므로 구현에서는 선택한 버전에 맞는 하나의 경로를 검증하고 고정한다. 이미지 입력을 지원하는 다른 Qwen으로 바꾸면 동일한 prompt·parser 계약을 유지하되 별도 실험 revision을 부여한다.

추가 fine-tuning 없이 checkpoint로 VQA를 시작하는 것은 ADVV의 초기 설계 선택이다. 그것만으로 산업 결함 판정이나 희귀 생물 식별의 정확도가 입증되는 것은 아니다. Qwen3.5 지원 버전을 실제 검증하기 전 예전 Qwen3-VL용 환경 하한을 재사용하지 않는다.

## 영역 proposal 모델 (sampler v2 후보)

확인일: **2026-09-30**. [sampler v2 설계](REGION_SAMPLER.md)에서 사용한다. 기본은 SAM 3이고 Grounding DINO + SAM 2.1은 비교용이다(2026-09-30 사용자 결정). 아래 commit·revision은 확인 시점의 값이다. SAM 3는 `configs/upstream.lock.json`의 `region_proposal.sam3`에 고정했다(아래 "SAM 3 고정과 설치"). 비교용 모델은 worker를 구현할 때 고정한다.

| 모델 | 공식 자료 | 확인 시점 commit / revision | 라이선스·접근 | 요구 환경 |
| --- | --- | --- | --- | --- |
| SAM 2.1 (비교용) | [facebookresearch/sam2](https://github.com/facebookresearch/sam2), [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` / `665f8e2ad61cf5f53d65644ff27c8ee525124610` | Apache-2.0, 승인 불필요 | Python ≥3.10, torch ≥2.5.1 |
| Grounding DINO (비교용) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | `856dde20aee659246248e20734ef9ba5214f5e44` / `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` | Apache-2.0, 승인 불필요 | Transformers 포트는 4.48에 포함 |
| SAM 3 (기본) | [facebookresearch/sam3](https://github.com/facebookresearch/sam3), [facebook/sam3](https://huggingface.co/facebook/sam3), [논문](https://arxiv.org/abs/2511.16719) | `2345a4ad109ac29c569da749c91d84f10dc08c40` / `3c879f39826c281e95690f02c7821c4de09afae7` | 자체 SAM License, HF 수동 승인 | Python ≥3.12, torch ≥2.7, CUDA ≥12.6 |
| Grounded-SAM-2 (참고) | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` | Apache-2.0 / BSD-3 | 조합 예제 참고용. SAM 3 미통합 |

- SAM 2.1은 point prompt 하나에 점수가 매겨진 mask 3개를 반환한다(`multimask_output`). 텍스트 prompt는 지원하지 않는다.
- Grounding DINO는 box만 반환한다. 텍스트는 소문자·마침표 구분 명사구이며 `box_threshold`·`text_threshold` 조정이 필요하다.
- SAM 3는 텍스트 → 인스턴스 mask와 point → 3단계 mask를 모두 지원한다. 논문은 학습 분야 밖의 세밀한 개념(의료·열화상 등)에 약하다고 밝히며, 산업 결함으로는 평가되지 않았다. 후속 SAM 3.1(2026-03-27)은 동영상 다중 객체 추적 개선이다.
- SAM 3 공식 API(commit `2345a4a`): `build_sam3_image_model(..., checkpoint_path, load_from_HF=False, enable_inst_interactivity=True)`, `Sam3Processor.set_image` → `set_text_prompt`(점수 = sigmoid(logit)×presence, `confidence_threshold`보다 큰 것만 반환, 기본 0.5), `model.predict_inst(state, point_coords, point_labels, multimask_output=True)` → 원본 해상도 mask 3개·예측 IoU(정렬 안 됨)·low-res logits (3, 288, 288)(T009c 실측, T008 문서의 256은 정정). 전처리는 1008×1008 종횡비 무시 resize.
- SAM 3 논문(arXiv:2511.16719) Appendix B는 학습 분야 밖의 세밀한 개념에 약하다고 밝히고, Roboflow100-VL zero-shot AP는 Industrial 9.0이다. 그래서 결함 자체가 아니라 결함을 가진 객체의 명사구로 grounding한다.
- Qwen3.5-4B 모델 카드는 RefCOCO 88.1을 보고하지만 box 출력 형식은 문서화하지 않았다. Qwen3-VL의 상대 좌표(0–1000) 형식을 이어받는다고 가정하지 않고 실측한다.

### SAM 3 고정과 설치 (2026-09-30)

| 항목 | 값 |
| --- | --- |
| 코드 | `facebookresearch/sam3` `2345a4ad109ac29c569da749c91d84f10dc08c40` → `third_party/sam3`(editable 설치, 수정 금지) |
| 가중치 | `facebook/sam3` `3c879f39826c281e95690f02c7821c4de09afae7`의 `sam3.pt`(3,450,062,241 B), `config.json`, `LICENSE`. 크기·SHA256은 `configs/upstream.lock.json`, 로컬 경로는 `models/weights.lock.json` |
| 사용하지 않음 | `model.safetensors`(Transformers 포트용), `facebook/sam3.1`(동영상 multiplex) |
| 환경 | `.venv-sam3`: Python 3.12, torch 2.10.0+cu128, `numpy<2`, `setuptools<82`, 추가 `einops`·`pycocotools`·`psutil`(sam3가 선언하지 않고 import). `environments/sam3-requirements.txt`, `environments/sam3.freeze.txt` |
| GPU 실측(T009c, 스모크 스크립트) | 드라이버 535.288.01에서 cu128 동작, missing key 0, 로드 8.4 s, RTX A6000 960px 이미지 피크 약 6.2 GiB, 텍스트 호출 뒤 같은 state의 점 호출 결과 동일 |
| 라이선스 | 자체 SAM License(2025-11-19). 재배포 시 사본 동봉(1.b.i), 논문 사용 표기(1.b.ii), 금지 용도(1.b.v). "research only"·"non-commercial" 문구는 없고 output의 소유·재배포를 직접 정한 조항도 없다. 저장소 pyproject classifier의 "MIT"와 충돌하며 효력 판단은 하지 않는다 |

Transformers 포트는 5.10.1에서 SAM 3 텍스트 인코더 가중치가 로드되지 않는 회귀가 있었고(5.10.2 수정), 점 prompt에는 별도 `Sam3TrackerModel`이 필요해 공식 패키지를 택했다. ADVV adapter를 통한 GPU 실행은 `tests/test_sam3_integration.py`(`-m integration`)로 따로 확인한다.

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

- `configs/upstream.lock.json`: DragFlow/FireFlow commit 및 모델별 revision. `region_proposal.sam3`에 SAM 3 코드 commit·HF revision·파일 SHA256(2026-09-30 추가).
- `models/weights.lock.json`: FLUX.1-dev를 포함해 다운로드 완료한 로컬 snapshot. 서버 브라우저 인증 후 gated 모델 접근을 확인했다.
- `environments/*.freeze.txt`: 설치한 coordinator/DragFlow/Qwen 환경.
- `integration_checks/qwen_worker/results.json`: 실제 Qwen3.5-4B의 단일/두 이미지 추론, torch 2.6.0 / Transformers 5.17.0, 약 9.5 GB 이하의 관측 allocated VRAM. 다른 이미지 크기의 요구량을 보장하지 않는다.
- run `config.json`에 prompt/model/environment, `manifest.json`에 implementation hash, `state.json`에 실행 segment를 기록한다.
