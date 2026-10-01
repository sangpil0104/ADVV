결과: done
변경 파일: _workspace/T020_researcher_regiondrag_lazydrag.md (신규). 저장소 파일 수정 없음
실행한 검사: 이 문서는 코드를 읽고 공식 자료를 확인한 결과다. 무엇을 실행했고 무엇을 실행하지 않았는지는 아래와 같다.
- 읽은 코드: third_party/DragFlow @ b3a8fa7(dragger.py, dragger_utils.py, dashboard_utils.py, masker_utils.py, hookhub.py, config.yaml), src/advv/backends/dragflow.py, src/advv/contracts.py.
- 공식 자료(확인일 2026-10-01):
  - arXiv HTML 원문: RegionDrag 2407.18247, LazyDrag 2509.12203v3, DragFlow 2510.02253v1, LightningDrag 2405.13722v2, FlowOpt 2510.22010.
  - RegionDrag 코드: scratchpad에 clone해서 읽기만 했다.
  - GitHub API와 HF API: 메타데이터만 조회했다.
- 실행하지 않음: GPU, 모델 다운로드, pip, 그리고 실험.
남은 것 / 막힌 이유:
- (B)의 속도·품질 효과는 실측 전이라 [I] 추정이다.
- (C) LazyDrag는 공식 코드가 아직 공개되지 않았다(아래 §C1).
- 배경 LPIPS에 필요한 lpips 패키지가 .venv-dragflow, .venv-qwen, advv 어디에도 없다.
사용자 결정 필요:
- (1) §B5 실험을 "DragFlow 설정 변형"으로 진행할지
- (2) LPIPS용 lpips 설치와 AlexNet 가중치 다운로드를 허용할지. 허용하지 않으면 PSNR/SSIM과 DINOv2 거리만 쓴다.
- (3) C는 공식 코드가 공개될 때까지 보류할지

표기: [V] 공식 원문 또는 고정 코드에서 확인, [I] 추론, [U] 확인 불가.

---

## B. RegionDrag식 영역 latent 초기화 + DragFlow round 축소

### B1. RegionDrag 원문 절차 (ECCV 2024, 코드 Visual-AI/RegionDrag @ 5cf4922, 2024-10-10, LICENSE 파일 없음 [V])

**1:1 매핑** [V 논문 Alg.1]
- 영역이 삼각형이나 사각형이면 affine 또는 perspective 행렬로 매핑한다.
- 임의 모양이면 Region-to-Point Mapping을 쓴다. target 픽셀의 x를 handle의 x 범위로 선형 스케일하고, 같은 열 안에서 y를 handle 열의 y 범위로 스케일한 뒤 floor한다.
- 좌표는 이미지 좌표를 `//8`로 나눠 latent 좌표로 바꾼다 [V drag.py:189].

**latent copy** [V 논문 Eq.3·5, Alg.2, drag.py:100–147]
- CP(z₁,z₂,h,t): z₂[t] ← z₁[h]. 정수 인덱싱(nearest)이며 보간하지 않는다.
- 초기 handle 영역은 r_α(z,H) = (1−H)z + H(√(1−α²)z + αε)로 노이즈와 섞는다.
  - 평가 스크립트 기본값은 α=1, UI 기본값은 0.6이다.
- 매 denoising step t ≥ t″에서 **inversion 때 캐시한 z_t의 handle 값**을 편집 latent의 target에 붙인다.
- 마스크 밖은 `torch.where(mask==1, latent, hook_latent)`로 원본 궤적 값으로 되돌린다.
- 사용 설정: SD1.5, DDPM 20 step, t′=500(10 step inversion), t″=200, 512² 이미지, V100 GPU [V 4.1절].

**attention swap** [V 논문 3.3절, drag.py:159–178]
- MasaCtrl 방식으로 UNet 모든 self-attn(attn1)의 to_k/to_v 출력 **전체**를 inversion 때 값으로 바꾼다.
- 영역별로 매핑해서 넣지 않는다. 모든 timestep에 적용한다.

**성능과 한계** [V]
- 512²에서 약 1.5 s 걸린다.
- DragBench-S(R)에서 MD 6.4, DragBench-D(R)에서 MD 6.6이다.
- 부록 7: α=1이어도 handle 위치의 원래 물체가 지워진다는 보장은 없다.
- DragFlow 논문 Tab.2(ReD Bench)에서 RegionDrag는 MD₁ 33.69, DragFlow는 19.46이다. DragFlow 논문은 "handcrafted 매핑이 영역 내부 구조를 보존하지 못한다"고 평가한다.

### B2. DragFlow에서 초기화를 넣을 지점과 상호작용 (b3a8fa7)

**이미 비슷한 초기화가 있는가:** 없다 [V].
- `z_orig = z_drag.detach().clone()`(dragger.py:575) 다음에 z_drag를 바로 `dragger_step`(629)에 넘긴다.
- 논문 Eq.1도 z_t^(0) ≜ z_t, 즉 inversion latent를 그대로 시작점으로 둔다.
- 논문은 RegionDrag와 FastDrag를 "copy-paste 계열"로 따로 분류한다.

**삽입 지점 (upstream 무수정):** wrapper가 `Dragger.dragger_step`을 감싸서 첫 호출에서 받은 `z_drag`만 warp한다 [I].
- `z_orig`와 그로부터 만드는 `F_orig`(319–329)는 원본 그대로 둔다.
- `process_inversion`을 감싸면 575에서 z_orig까지 warp되므로 쓰면 안 된다.

**affine supervision과의 충돌 (핵심):** 원래 흐름은 다음과 같다 [V]. 손실(390–420)은 `F_drag·M^(k)`와 `F_orig`를 같은 부분 affine(k/K)로 grid_sample한 값의 L1이다. `progressive_weight = k/max_dragging_num`이고(dragger_utils.py:72), 목표 영역은 50 round에 걸쳐 조금씩 이동한다.
- latent를 미리 **전부** 옮겨 두면 초반 round의 목표(부분 이동 위치)와 어긋난다. 그러면 gradient가 물체를 중간 위치로 되돌리게 된다 [I].
- 해결: `max_dragging_num: 1`로 둔다(둘 다 `ALLOWED_PARAMETERS`라 코드 수정이 필요 없다).
  - 그러면 첫 round부터 weight=1이 된다(72행: (0·1+1)/(1·1)).
  - 같은 round에서 `is_last_operationidx`가 참이 되어 `full_grid`가 저장된다(152–153행).
  - 남은 N−1회는 INTENSIFY로 full_grid를 반복한다.
- 결과적으로 초기화 + N round는 `max_dragging_num: 1, max_intensify_num: N−1`이 된다 [V 코드 경로, 효과 I].

**배경 보존:** `_combine_latents`(458행, 700행)가 매 GD와 매 sampling step에서 `mask_fit` 밖을 z_orig_/z_orig로 덮어쓴다 [V].
- mask_fit은 source 영역과 target 영역을 합친 minAreaRect다(masker_utils.py, `create_adaptive_mask`) [V].
- 따라서 warp 결과를 **mask_fit 안으로 잘라야** 한다. 밖으로 나간 부분은 첫 GD 뒤에 사라진다.

**KV 주입:** 13개 블록(double 0,7,8,9,10,18 / single 6,9,18,23,26,31,37)의 to_k/to_v 출력 **전체**를 inversion 값으로 교체한다(eta=0, 모든 timestep) [V hookhub.py:45–82, 183–187].
- 이것은 RegionDrag의 attention swap과 같은 종류다. B를 위해 KV를 따로 추가할 필요는 없다 [I].
- 캡처되는 K는 RoPE를 적용하기 전 값이라, 위치 x에는 원본 x의 내용이 주입된다. 그래서 비워진 source 자리의 query가 원래 물체의 key와 강하게 매칭되어 **물체가 원래 자리에 다시 생길(ghost) 위험**이 있다. 공식 경로에도 있는 위험이다 [I].
- 매핑된 KV 주입(target x에 M(x)의 K/V를 넣는 방식)은 LazyDrag 쪽 기법이고, 알고리즘 이탈이 크다. B에는 넣지 않는다.

**IP-adapter:** 영향 없음 [I].

### B3. FLUX latent에서 1:1 매핑할 때 주의점

| 항목 | 권장 |
|---|---|
| 작업 격자 | 패킹된 1/16 토큰(64 dim)이 아니라, `decode_for_calculation`으로 푼 **16ch × H/8 × W/8**에서 매핑한 뒤 `encode_for_calculation`으로 다시 패킹한다. upstream의 combine도 이 격자에서 한다 [V dashboard_utils 471–515] |
| affine 행렬 | upstream `_process_rotation`/`_process_transformation`의 θ를 weight=1로 재사용한다. 정규화 좌표(align_corners=False)라 cut_shape(예: 320×240)와 latent 격자(120×90)의 종횡비가 같으면 그대로 맞는다. floor-16 resize 때문에 생기는 1칸 이내 오차는 기록한다 [I] |
| 보간 | **nearest**를 쓴다. bilinear는 서로 독립인 노이즈 성분을 평균해 분산을 줄이고(blur latent), 결과가 흐려진다 [I]. RegionDrag(정수 인덱스)와 LazyDrag(Π 격자 투영)도 nearest다 [V]. rotation에서 생기는 구멍과 충돌은 역매핑(target→source) grid_sample(nearest)로 피한다 |
| 경계 | target 마스크는 `region_init`를 latent 격자로 nearest 변환한 뒤 같은 θ로 warp한다. 이미지 밖은 버린다(zeros padding). mask_fit과 교집합을 취한다. 1칸 feather는 선택 arm으로 둔다 |
| 빈 자리(source − target) | f1: 원본을 유지한다(z_orig). ghost 위험이 있다. f2: ε ~ N(0,I)로 교체한다(LazyDrag 식(2), RegionDrag α=1). 첫 조작 시점이 t≈0.888(960×720, shift 반영 [I 계산])이라 z_t ≈ 0.11·x₀ + 0.89·ε이므로, 순수 ε와 분포 차이가 작다 [I]. f2를 1순위로 둔다 |
| deformation | upstream도 deformation을 평행이동으로 처리한다(dragger_utils.py:129) [V]. 같은 θ를 쓴다 |
| 패치 정렬 | 1/8 기준으로 홀수 칸을 이동하면 2×2 패치 경계를 넘는다. 패킹 전에 다루므로 오류는 아니다. 패치 단위 정렬은 선택 사항 [I] |

### B4. 유사 시도 (존재와 결과만)

- **FastDrag** (2405.15769): 한 번의 latent warpage로 처리하고 최적화는 없다. 3.12 s/point [V T016].
- **LazyDrag "Latent Init"**: 명시적 대응으로 z_T를 만들고 inpainting 영역은 ε로 채운다. TTO는 없다.
  - ablation에서 WTA와 Latent Init을 빼면 MD가 21.49에서 23.69로 나빠진다 [V Tab.3].
  - FastDrag에 Latent Init을 더하면 MD가 31.84에서 28.97로 좋아진다 [V Tab.5].
- **LightningDrag** (2405.13722): 학습 모델에서 noise prior를 비교했는데, copy-paste prior보다 "noised source latents"가 가장 좋았다 [V 4.3.1].
- **SDE-Drag** (2311.01410): copy-paste를 하고 denoise/invert를 반복한다 [V RegionDrag 원문 Eq.4].
- **DragNoise**: 최적화 시간을 50% 이상 줄였다 [V T016]. **GoodDrag**: 속도 수치는 확인하지 못했다 [U]. **FlowOpt**: gradient 없는 zero-order 방식이다 [V].
- copy 초기화 **뒤에** gradient latent 최적화를 줄여서 붙인 DiT 논문은 찾지 못했다 [U]. 따라서 B는 ADVV 자체 변형이다.

### B5. 실험 설계 (GPU 2장, advv-gpu-safety 절차를 따른다. 모든 arm에 같은 speedups `exact_all`)

- 후보: v2_pilot_001의 relocation 2장, rotation 2장, deformation 2장. 각 후보의 seed, mask, plan을 고정한다.
- arm:
  - A0: 공식 50+20
  - A1: 대조군. 초기화 없이 1+(N−1). 효과가 초기화 때문인지 full-target supervision 때문인지 분리한다.
  - A2: 단순 축소. 초기화 없이 round(5N/7)+round(2N/7)
  - B1: 초기화 f2 + 1+(N−1)
  - B2: 초기화 f1 + 1+(N−1)
  - N은 {10,20,30}이다. B1은 N=1도 넣는다(복사만 하고 DragFlow sampling, 하한 확인용).
- 지표:
  - (a) 시간: drag 구간과 전체 wall time
  - (b) drag 도달:
    - 마지막 round의 DragFlow loss(같은 t에서만 비교 가능)
    - DINOv2-giant(이미 cuda:1에 로드됨)로 원본 source 영역 패치 feature와 생성 이미지 target 영역 feature 사이의 masked cosine 거리
    - source 점 64개의 NN 매칭으로 구한 MD(목표점까지 거리)
  - (c) ghost: 생성 이미지의 source−target 영역과 원본 같은 영역의 DINOv2 유사도. 높으면 물체가 남아 있다는 뜻이다.
  - (d) 배경: mask_fit을 16px 팽창한 영역 밖에서 PSNR과 SSIM을 잰다. 원본 대비, 그리고 x0_orig(재구성) 대비 두 가지로 잰다. LPIPS는 설치 승인 뒤에 추가한다.
  - (e) Qwen 두 판정(현재 고정 prompt). 판정은 참고 지표이고, 품질의 증명이 아니다.
- 합격 기준 제안: B1이 A0 대비 (b)를 90% 이상 유지하고, (d) 차이가 0.5 dB 이내이며, Qwen 판정 YES율이 같거나 높은 가장 작은 N을 고른다.
- 예상 시간 [I]: exact_all 3.1 s/round, no_grad forward 3.4 s 기준.
  - 후보 1장: inversion 77 s + sampling 약 40 forward ≈ 136 s + drag.
  - A0는 약 430 s, N=20은 약 275 s다(전체 1.6배, drag 구간만 3.5배).
- 기록 방식:
  - `generator.parameters`(max_dragging_num, max_intensify_num)는 기존 `ALLOWED_PARAMETERS`로 기록된다.
  - 초기화는 새 플래그로 둔다. 예: `generator.recipe_variants.region_latent_init: {fill: noise, interp: nearest}`. 이 플래그는 `speedups`가 아니라 **recipe variant**로 분리한다.
  - `official_execution_path=false`와 `recipe_variant=["region_latent_init"]`을 run config, generation_info, export manifest에 남긴다.
  - src가 바뀌므로 implementation hash도 바뀐다.
  - 문서에는 "DragFlow b3a8fa7 + ADVV 초기화 변형"으로 적고, 공식 DragFlow 결과라고 보고하지 않는다.

## C. LazyDrag를 별도 backend로

### C1. 공개 상태

| 항목 | 내용 |
|---|---|
| 논문 | arXiv 2509.12203 v1 2025-09-15, v2 09-25, v3 2026-02-03. 논문 라이선스 CC BY 4.0. StepFun. 프로젝트 페이지에 ICLR 2026 표기 [V] |
| 코드 | **공개되지 않음.** 프로젝트 페이지(zxyin.github.io/LazyDrag, 저장소 zxYin/LazyDrag @ cd1d76b, 2026-04-20)는 index.html과 imgs만 있고 "Code will be open-sourced."라고 적혀 있다. 논문에도 "upon acceptance"라고 되어 있다. 같은 저자는 ColorCtrl_Code처럼 `_Code` 저장소에 코드를 따로 올리는데, `zxYin/LazyDrag_Code`는 404이고 stepfun-ai 조직에도 없다 [V GitHub API 2026-10-01] |
| 모델 | FLUX.1 Krea-dev. HF `black-forest-labs/FLUX.1-Krea-dev` sha 8162a9c7b05a641be098422bf2fcf335615c2f28 (lastModified 2025-07-31). gated(auto 승인). flux-1-dev-non-commercial-license. FLUX.1-dev의 finetune. 전체 57.9 GB, diffusers transformer 23.8 GB(bf16) [V HF API] |
| inversion | UniEdit-Flow의 공식 inversion을 쓴다(DSL-Lab/UniEdit-Flow 존재, 라이선스 표기 없음) [V] |
| 구성 | correspondence 기반 제어는 single-stream attention 층에만 적용한다. 50 step. ID Pres./Attn Refine은 처음 40 step에서 켠다. bf16 [V A.1, B.4] |
| 입력 | handle/target **점 쌍**과 editable region(mask). drag mode(탄성 변위, 3D 회전과 늘림)와 move mode(이동과 스케일)가 있다. 텍스트 prompt는 선택이다 [V 3.2, A.2] |
| 시간·VRAM (H800 1장) | Default: inversion 6.79 + map 0.54 + 생성 6.77 = **14.10 s, 62 GB**. Optimized(20 step, strength 0.7): **4.31 s, 49 GB** [V Tab.6]. T016에서 "runtime 없음"이라고 적은 것은 v3에서 바뀌었다 |
| 성능 | DragBench MD 21.49(TTO 없음) [V Tab.1] |
| torch/diffusers | [U]. 코드가 없다. 같은 저자의 ColorCtrl_Code는 diffusers==0.34.0을 쓴다 [V]. .venv-dragflow는 diffusers 0.32.2, torch 2.5.1이므로 **별도 venv가 필요할 것**으로 본다 [I] |
| A6000 48 GB | Default 62 GB는 1장에 들어가지 않는다. T5와 VAE를 다른 GPU에 두거나 offload해야 한다. 양자화를 쓰면 recipe 이탈이다 [I] |

### C2. ADVV 통합

- **이름과 정책:** AGENTS는 DragFlow를 다른 생성기로 "DragFlow인 것처럼" 바꾸는 것을 금지한다. 별도 backend(`generator.backend: lazydrag`)는 이름을 명시하므로 규칙과 충돌하지 않는다.
  - 다만 프로젝트 이름이 "DragFlow"이므로 SPEC과 AGENTS에 "기본이자 주 backend는 DragFlow, LazyDrag는 비교·선택 backend"라는 문장을 추가해야 한다(사용자 승인 필요).
- **계보 기록:** run 단위로 backend를 하나만 쓴다. run config, generation_info, export manifest의 각 행에 `generator_backend`, `upstream_commit`, `model_revision`을 남긴다. report와 수락률은 backend별로 분리한다. DragFlow 결과와 섞어 export하는 것은 기본으로 막는다.
- **코드가 공개되지 않은 문제:** 논문을 보고 다시 구현한 것은 "LazyDrag 공식 구현"이 아니다. 그래서 `lazydrag_reimpl`처럼 따로 이름을 붙여야 하고, 논문 수치 재현 검증이 선행되어야 한다. AGENTS의 "공식 구현의 고정 commit 기준" 정신과 맞지 않으므로 **보류를 권장한다.**
- **sampler v2 계획 변환** [I]:
  - relocation → move mode. s = 영역 centroid(=source_point), e = target_point, editable = region mask를 target까지 덮도록 확장. 그대로 대응된다.
  - deformation → drag mode, 점 쌍 1개. 의미가 DragFlow와 다르다(DragFlow는 영역 평행이동, LazyDrag는 탄성 변형).
  - rotation → LazyDrag에는 anchor 회전 연산이 없다. 경계 점 여러 개를 anchor 기준으로 회전시켜 점 쌍으로 만들 수는 있지만, WTA가 Voronoi로 나눠 조각별 평행이동이 된다. 동등하지 않으므로 이 backend에서는 rotation을 끄는 것을 권장한다.
  - `effective.json` 계약(실제 전달 좌표와 mask)은 같은 방식으로 기록할 수 있다.

## 비교와 추천

| | B 초기화 + round 축소 | C LazyDrag backend |
|---|---|---|
| 구현 비용 | 작음. wrapper 패치 약 80줄, 플래그 1개, CPU 테스트(warp 좌표·nearest·mask 교집합) | 큼. 코드가 없어 재구현해야 하고, 새 venv, Krea 가중치 58 GB, 2-GPU 배치, 새 backend와 계약·문서 |
| 기대 이득 | drag 구간 2–7배, 후보 1장 전체 약 1.3–1.9배 [I] | 1장 수십 초 수준 [I, H800 14 s 기준을 A6000으로 환산] |
| 품질 위험 | 중간. ghost, 경계 이음매, full-target supervision에 대한 반응이 미지수. A1 대조군으로 분리 가능 | 큼. 재현성 미검증, rotation 없음, 논문 수치 재현 불확실 |
| 규칙 위험 | 낮음. DragFlow를 유지한 recipe variant로 기록 | 중간. 문서 승인 필요. 비공식 재구현은 계보 표기 문제 |
| 라이선스 | DragFlow·FLUX.1-dev와 같음 | Krea non-commercial(FLUX와 같은 계열). LazyDrag 코드 라이선스 [U] |

**추천 순서:** B를 먼저 한다.
1. A1 대조군(설정만 바꾸면 되고 코드 변경이 없다)으로 "1+(N−1)만으로도 도달하는지" 확인한다.
2. B1(f2 noise fill)을 구현하고 N∈{10,20,30}을 측정한다.
3. C는 공식 코드가 공개될 때까지 보류한다. 공개되면 commit, 라이선스, 요구 버전부터 다시 조사한다(zxYin/LazyDrag 저장소 감시).

## 출처 (확인 2026-10-01)
- RegionDrag: https://arxiv.org/html/2407.18247 , 코드 https://github.com/Visual-AI/RegionDrag @ 5cf492212c24edae5ca2376579f2e81f46fc322f (region_utils/drag.py, run_eval.py, ui.py)
- DragFlow: https://arxiv.org/html/2510.02253v1 , third_party/DragFlow @ b3a8fa7136df5131d07f98425c0d189fb6ca74b0
- LazyDrag: https://arxiv.org/abs/2509.12203 (v3), https://arxiv.org/html/2509.12203 , https://zxyin.github.io/LazyDrag , https://github.com/zxYin/LazyDrag @ cd1d76b2cda21f472f9c2e6aa9cb729aad642e0a
- FLUX.1 Krea-dev: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev (sha 8162a9c7b05a641be098422bf2fcf335615c2f28)
- UniEdit-Flow: https://github.com/DSL-Lab/UniEdit-Flow · ColorCtrl_Code: https://github.com/zxYin/ColorCtrl_Code (requirements.txt)
- LightningDrag: https://arxiv.org/html/2405.13722v2 · FlowOpt: https://arxiv.org/abs/2510.22010 · FastDrag, DragNoise, GoodDrag: T016 출처 목록
