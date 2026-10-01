결과: done
변경 파일: _workspace/T016_researcher_dit_speedup.md (신규). 저장소 파일 수정 없음
실행한 검사: upstream 코드 정독(third_party/DragFlow b3a8fa7: dragger.py, overrider_DiT.py, pipeline_flux.py, hookhub.py, adapter/attn_processor.py, dashboard_utils.py), .venv-dragflow의 optimum-quanto 0.2.6 qbytes_mm 소스, runs/v2_pilot_001/logs/generator.log 타이밍, `nvidia-smi topo -m`(읽기 전용 조회, GPU 작업 없음), 공식 논문/저장소 웹 확인(2026-10-01). GPU 실행·벤치마크·pip·다운로드: 미실행
남은 것 / 막힌 이유: 아래 속도 이득 수치는 모두 [I] 추정. 실측(§5 실험) 전에는 확정 불가
사용자 결정 필요: (1) §5 실험 1(수치 동일 패치)을 implementer+gpu-runner에 맡길지 (2) 정밀도 변경(TF32/bf16)을 "DragFlow recipe 변형"으로 허용할지 (3) round 수 축소 A/B를 할지

---

# T016 DragFlow latent 최적화 가속 조사

표기: [V] 공식 출처/고정 코드에서 확인, [I] 추론, [U] 확인 불가. 확인일 2026-10-01.

## 0. 핵심 결론

- 시간 구성(960×720, T011 로그 [V]): forward 1회(no_grad) ≈ **16.4 s**, drag round ≈ 29 s × 70 ≈ 34 min, inversion 20 forward ≈ 5.5 min, sampling 약 40 forward ≈ 11 min [I: dragger.py 루프에서 셈]. 즉 rounds 약 2/3, **no_grad forward 약 1/3**.
- 16.4 s/forward는 FLUX 연산량(3,212 토큰에서 약 60 TFLOP [I])에 비해 A6000 FP32 성능의 약 10%만 쓰는 수준이다 [I]. 원인 후보는 공식 경로가 쓰는 **fp32 activation + quanto qint8 dequant(층마다 fp32 weight를 만든 뒤 matmul)** [V: quanto `qbytes_mm`], fp32라서 FlashAttention을 못 씀 [V], IP-adapter processor가 attention 57층마다 cuda:0↔cuda:1 왕복 [V], KV hook의 pageable CPU↔GPU 복사 [V]이다. **어떤 것이 지배적인지는 profiler로 확인해야 한다 [U]**.
- 논문 가속(최적화 없는 drag)은 모두 **다른 생성기**라서 AGENTS 규칙상 DragFlow를 대체할 수 없다. 별도 backend로 이름을 붙여야만 쓸 수 있다.
- 추천 1순위: **수치가 같은 실행 패치**. 드래그 round에서 쓰지 않는 single block 38개 forward 생략, gradient checkpointing 끄기(메모리가 허용하면), IP processor 같은 GPU 배치, `reclaim_memory` 무력화를 묶어 먼저 측정한다.

## 1. upstream 병목과 조정 손잡이 (commit b3a8fa7)

| 손잡이 | 위치·현재값 [V] | 바꾸면 | 공식 설정 이탈 |
|---|---|---|---|
| 계산 dtype | `bench_dragflow.py`와 ADVV wrapper 모두 `Dragger(conf, dtype=torch.float32)` | TF32 플래그 / bf16 | 정밀도 이탈. 기록 필요 |
| Transformer 양자화 | `dragger.load_pipeline`: `quantize(transformer, weights=qint8)`, `freeze`, cuda:0. T5도 qint8 (`HFEmbedder`) | bf16 weight 비양자화 | 이탈(코드 경로) |
| quanto 연산 | CUDA·fp32 activation이면 `qbytes_mm` → `weights = scales * int8` 후 `torch.matmul` (fused kernel 없음). backward도 `matmul(gO, qweight)` | bf16이어도 같은 dequant 경로. 비양자화 bf16이면 dequant 없음 | — |
| Gradient checkpointing | `transformer.enable_gradient_checkpointing()`. 블록마다 `checkpoint(use_reentrant=False)` | 끄면 doubles 재계산 1회분 절약 | 수치 동일 |
| Loss 대상 feature | `DOUBLE-17-0/-2, DOUBLE-18-0/-2` (framework/config.yaml을 import 시점에 읽음. ADVV conf로 못 바꿈) | — | 바꾸면 알고리즘 이탈 |
| Round forward 범위 | `_extract_latent_features`가 double 19개 + **single 38개 전부** 실행. noise_prediction(`np_drag`)은 바로 `del` | single 생략은 loss·grad가 수학적으로 같음 | 수치 동일(코드 경로만 다름) |
| Backprop 범위 | loss → DOUBLE-18 → … → DOUBLE-0 → `x_embedder` → z (19 double 전부) | 일부만 backprop하면 grad가 달라짐 | 알고리즘 이탈 |
| Rounds | `max_dragging_num 50` + `max_intensify_num 20`, `max_operation_num 1`(첫 sampling step 한 곳에서만 최적화) | 선형 시간 감소 | 이탈(ADVV `generator.parameters`로 이미 허용) |
| Inversion/sampling | `inversion_step_num 25`, `sampling_step_num 25`, `skip_step_num 6` → 각 19 step. FireFlow는 step당 1 NFE(첫 step 2) | 줄이면 inversion 충실도 저하 | 이탈 |
| 두 GPU 사용 | cuda:1에는 SigLIP·DINOv2(fp32), image_proj_model, **IP attn processor 57개(fp32, 약 1.43B param ≈ 5.7 GB [I])**, loss 텐서. `_get_ip_hidden_states`가 층마다 query를 cuda:1로 보내고 ip_q/k/v를 cuda:0으로 되돌림(기본 인자 `device_1="cuda:1"` 하드코딩). GPU8–9는 같은 PCIe 스위치(PIX), NVLink 없음 [V topo] | 같은 GPU에 두면 복사·동기화 제거 | 수치 동일(장치만 다름) |
| `reclaim_memory()` | 호출마다 **모든 GPU**에 `set_device`+`synchronize`+`empty_cache`+`ipc_collect`, 그리고 `gc.collect()`. round마다 여러 번 호출 | no-op로 바꾸면 sync·GC 비용 제거, 단편화로 피크가 오를 수 있음 | 수치 동일 |
| KV hook | inversion에서 13개 블록의 k/v를 `.cpu()`로 저장(pageable), round·sampling forward마다 `.to(device)`로 주입. checkpoint 재계산 때 한 번 더 | pinned memory·GPU 상주 | 수치 동일 |
| 논문과 코드 차이 | 논문: INTENSIFY 20회 lr **1200** [V arXiv HTML]. 코드: `self.conf["lr"]=1000.0` [V] | — | 참고. ADVV는 코드를 따름 |

메모리 추정 [I] (현재 피크 GPU8 17.6 GiB / GPU9 19.3 GiB, T011 [V]): FLUX transformer 11.9B param 기준 bf16 ≈ 24 GB, qint8 ≈ 12 GB, fp32 ≈ 48 GB(48GB 카드에 안 들어감). (a) bf16 비양자화: cuda:0 ≈ 24 + T5 qint8 ≈ 5 + activation 수 GB ≈ 32–38 GB, 들어감. (b) fp32+qint8에서 checkpointing 끄기: double 19블록 activation이 약 1 GB/블록이면 +~20 GB → 약 38 GB, 빠듯하지만 들어갈 수 있음. bf16이면 절반. 반드시 `max_memory_allocated`로 측정해야 한다.

## 2. 드래그 편집 논문 (최적화 비용 축소)

| 방법 | 출처 | 방식 | 백본 | 보고 속도 | ADVV 적용 |
|---|---|---|---|---|---|
| DragFlow | ICLR 2026, arXiv 2510.02253 [V] | region affine supervision, 7번째 step에서 70 iter 최적화 | FLUX.1-dev | 논문 본문·부록에 runtime 표 없음 [V] | 현재 기준 |
| LazyDrag | arXiv 2509.12203 [V] | 명시적 correspondence로 attention 제어. TTO 없음 | FLUX.1 Krea-dev(MM-DiT), H800, 50 step | runtime 수치 없음 [V] | 다른 생성기. 대체 불가. 별도 backend 후보로 가장 가까움 |
| FastDrag | NeurIPS 2024, arXiv 2405.15769 [V] | latent warpage 1 step, 최적화 없음 | SD1.5(+LCM) | 3.12 s/point vs DragDiffusion 21.54 s (RTX 3090) [V] | UNet. 대체 불가 |
| RegionDrag | ECCV 2024, arXiv 2407.18247 [V] | region 기반, attention swap, 1 iteration | SD 계열 | 512² < 2 s, DragDiffusion 대비 100배 이상 [V] | 대체 불가. region 개념은 DragFlow와 같은 축 |
| InstantDrag | SIGGRAPH Asia 2024, arXiv 2409.08857 [V] | 학습된 FlowGen+FlowDiffusion, 최적화 없음 | 자체 학습 모델 | "interactive" (수치 미확인 [U]) | 대체 불가 |
| LightningDrag | arXiv 2405.13722 [V] | 비디오로 학습한 조건부 생성 | SD 계열 [I] | 약 1 s [V] | 대체 불가 |
| Inpaint4Drag | ICCV 2025, arXiv 2509.04582 [V] | 픽셀 warp + inpainting | 임의 inpainting 모델 | warp 0.01 s, inpaint 0.3 s (512²) [V] | 대체 불가 |
| DragNoise | CVPR 2024, arXiv 2404.01050 [V] | U-Net bottleneck 예측 noise 편집 | SD U-Net | DragDiffusion 대비 최적화 시간 50% 이상 감소 [V] | DiT로 옮기면 "일부 block만 최적화" 아이디어. 알고리즘 이탈 |
| GoodDrag | arXiv 2404.07206 [V] | denoise·drag 교대(AlDD) | SD 계열 | 속도 수치 미확인 [U] | 품질 중심. 속도용 아님 |
| StableDrag | arXiv 2403.04437 [V] | point tracking + confidence latent | GAN/SD | 수치 없음 [V] | 속도용 아님 |
| FlowOpt | arXiv 2510.22010 [V] | 전체 flow를 블랙박스로 보는 zero-order 최적화. backprop 없음 | flow/diffusion 공통 | 기존과 비슷한 NFE [V] | 목적함수가 다름. 연구 아이디어 수준 |

정리: DiT/FLUX에서 동작이 확인된 것은 DragFlow와 LazyDrag뿐이다. 나머지 가속 논문은 UNet 기반이거나 별도로 학습한 모델이다. 쓸 수 있는 아이디어는 "적은 iteration", "일부 블록만 최적화" 정도이며, 둘 다 DragFlow 레시피를 바꾸는 것이다.

## 3. 역전파 비용 축소 기법

| 기법 | 근거 | 이 코드에 적용 | 예상 [I] |
|---|---|---|---|
| Round 중 single block 38개 생략 | loss는 DOUBLE-17/18 feature만 사용 [V] | wrapper에서 forward 조기 종료(monkeypatch). grad는 같음 | round마다 forward 약 45%(single 몫) 절약 → round 약 15–20% 단축 |
| Checkpointing 끄기 | diffusers checkpointing은 double을 재계산 [V] | `disable_gradient_checkpointing()` | round 약 15–20% 단축, 메모리 +10–20 GB |
| TF32 | PyTorch 1.12 이상에서 `allow_tf32` 기본 False, A100에서 matmul 약 7배 [V docs.pytorch.org] | wrapper에서 플래그만 켬. fp32 dtype 유지 | GEMM 지배 구간 1.5–3배. 정밀도 약간 하락 |
| bf16 + FlashAttention | FA2는 fp16/bf16 전용 [V], SDPA는 FA2/mem-eff/math [V] | dtype=bf16 또는 autocast. latent·lr=1000 업데이트는 fp32 유지 권장 | forward 2–4배. 수치 이탈이 가장 큼 |
| quanto 제거(bf16 weight) | quanto 블로그: 목적은 메모리이고 지연은 비슷하거나 약간 늘어남 [V hf.co/blog/quanto-diffusers] | load_pipeline 재현(upstream 수정 없이 wrapper에서) | dequant 제거 |
| torch.compile | 미검증. quanto tensor subclass·hook·checkpoint와 호환성 [U] | 후순위 | [U] |
| 일부 층만 backprop | 알고리즘상 grad가 z까지 가야 해서 19 double 전부 필요 [V] | 불가(이탈) | — |

## 4. 추론 가속(inversion·sampling 구간)

| 방법 | 출처 | 보고 수치 | 적용성 |
|---|---|---|---|
| TeaCache | CVPR 2025, arXiv 2411.19108. TeaCache4FLUX README [V] | FLUX-dev A800 기준 1.5배(thresh 0.25) ~ 2.25배(0.8) | 낮음. inversion을 캐시하면 KV capture·재구성이 틀어짐 [I]. sampling에만 써도 전체 약 7% [I] |
| FireFlow | arXiv 2412.07517 [V] | 8 step 편집, ReFlow inversion 대비 3배 | 이미 사용 중 |
| ToCa | ICLR 2025, arXiv 2410.05317 [V] | PixArt-α 1.93배, OpenSora 2.36배 | FLUX 수치 없음. 낮음 |
| FORA | arXiv 2407.01425 [V] | "several times" (DiT) | 낮음 |
| Δ-DiT | arXiv 2406.01125 [V] | PixArt-α 20 step에서 1.6배 | 낮음 |
| FasterCache | arXiv 2410.19355 [V] | 비디오 1.67배. CFG-cache 포함 | FLUX-dev는 distilled guidance라 CFG 분기가 없음 [I]. 해당 없음 |

결론: 캐싱은 첫 우선순위가 아니다. per-forward 비용(dtype·quanto·장치 왕복)을 줄이면 inversion·sampling도 함께 빨라진다.

## 5. 추천 실행 계획 (품질 위험 낮은 순 = 우선순위)

공통 측정 하네스: `_workspace/` 스크립트로 wrapper `DragFlow` 1회 로드, accident_car_tree 960×720, T011 계획(seed 고정). 측정 항목: no_grad forward s/step(inversion 3 step), round s/round(warmup 1 + 5 round, `max_dragging_num 5`, `max_intensify_num 0`; 모든 arm 같은 설정), `max_memory_allocated`(cuda:0/1), round별 loss, 5 round 뒤 z_drag의 기준 대비 max|Δ|. GPU 2장은 advv-gpu-safety 절차로 배정한다.

**0단계 (필수, 약 15분):** baseline에서 `torch.profiler`로 forward 1회 + round 1회. aten::mm·dequant mul·Memcpy PtoP/HtoD/DtoH·SDPA·cudaDeviceSynchronize·Python GC 시간 비율을 본다. 16.4 s의 원인을 정한 뒤 아래 순서를 조정한다.

1. **수치 동일 패치 묶음** (품질 위험 거의 없음 / 구현 작음–중간 / 예상 round −25~40%, forward는 장치 왕복·reclaim 비중에 따라 [U])
   - arm 1a single 생략, 1b +checkpointing off, 1c +IP processor·image_proj를 cuda:0으로, 1d +`reclaim_memory` no-op. 누적 비교.
   - 합격 기준: loss 궤적 상대오차 ≤1e-4, z max|Δ| ≤1e-3(GPU 비결정성 허용), 피크 ≤44 GB.
   - 기록: generation receipt에 `execution_patches` 목록 추가. src 변경으로 implementation hash가 바뀜.
2. **TF32** (위험 낮음–중간 / 구현 2줄 / 예상 forward 1.5–3배 [I])
   - arm: 1의 최선 + `torch.backends.cuda.matmul.allow_tf32=True`. 5 round 지표 + 전체 생성 3장(소스 3개)의 baseline 대비 PSNR/LPIPS, Qwen 판정 비교(증명 아님).
   - 기록: `dragflow_precision: fp32+tf32`. recipe 변형으로 표시.
3. **bf16 compute** (위험 중간 / 구현 중간 / 예상 forward 2–4배 [I])
   - arm 3a dtype=bf16 + qint8 유지, 3b bf16 비양자화 weight. latent·SGD 업데이트는 fp32 유지(autocast)를 우선 시험. IP processor dtype 혼합 동작은 [U]라서 smoke로 확인.
   - 같은 품질 비교. 열화가 보이면 채택하지 않는다.

보류: rounds 50+20 → 30+10 같은 축소는 시간에 비례해 줄지만(round 구간 −43%) 논문 설정에서 벗어나 드래그 도달도가 떨어질 위험이 있다. 1–3 뒤에도 느리면 별도 A/B로 진행한다. LazyDrag 등 다른 방법은 대체가 아니라 별도 backend 결정 사항이다.

## 출처 (확인 2026-10-01)
- DragFlow https://arxiv.org/abs/2510.02253 , https://arxiv.org/html/2510.02253v1 ; 코드 third_party/DragFlow @ b3a8fa7136df5131d07f98425c0d189fb6ca74b0
- LazyDrag https://arxiv.org/abs/2509.12203 · FastDrag https://arxiv.org/abs/2405.15769 · RegionDrag https://arxiv.org/abs/2407.18247 · InstantDrag https://arxiv.org/abs/2409.08857 · LightningDrag https://arxiv.org/abs/2405.13722 · Inpaint4Drag https://arxiv.org/abs/2509.04582 · DragNoise https://arxiv.org/abs/2404.01050 · GoodDrag https://arxiv.org/abs/2404.07206 · StableDrag https://arxiv.org/abs/2403.04437 · FlowOpt https://arxiv.org/abs/2510.22010
- TeaCache https://arxiv.org/abs/2411.19108 , https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX · FireFlow https://arxiv.org/abs/2412.07517 · ToCa https://arxiv.org/abs/2410.05317 · FORA https://arxiv.org/abs/2407.01425 · Δ-DiT https://arxiv.org/abs/2406.01125 · FasterCache https://arxiv.org/abs/2410.19355
- PyTorch TF32 https://docs.pytorch.org/docs/2.5/notes/cuda.html · SDPA https://docs.pytorch.org/docs/2.5/generated/torch.nn.functional.scaled_dot_product_attention.html · FlashAttention https://github.com/Dao-AILab/flash-attention · quanto+diffusers https://huggingface.co/blog/quanto-diffusers · RTX A6000 https://www.nvidia.com/en-us/design-visualization/rtx-a6000/
- optimum-quanto 0.2.6 소스(.venv-dragflow): optimum/quanto/library/qbytes_mm.py, tensor/weights/qbytes.py, tensor/function.py
