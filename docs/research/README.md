# DragFlow 가속 탐색 기록

확인일: 2026-10-01. 하위 에이전트 조사·측정 보고서 원문을 보존한다. 결정이 보류된 항목은 구현하지 않았다.

| 문서 | 내용 |
| --- | --- |
| [dit_speedup_survey.md](dit_speedup_survey.md) | DiT/FLUX 가속 논문·기법 조사, DragFlow 병목 분석(fp32+qint8, IP-adapter 장치 왕복, reclaim_memory 동기화, 미사용 single block) |
| [dragflow_speedup_measurement.md](dragflow_speedup_measurement.md) | GPU 측정: `generator.speedups` 결과 동일 패치 4개 = 공식 경로와 noise floor 안에서 동일, drag round 8.6 → 3.1 s(2.78×), 첫 GPU 피크 37.6 GiB. TF32는 3.91×이나 수치가 달라짐 |
| [regiondrag_lazydrag.md](regiondrag_lazydrag.md) | (B) RegionDrag식 1:1 영역 latent 초기화 + round 축소의 DragFlow 통합 설계·실험 설계, (C) LazyDrag 별도 backend 검토 |

## 현재 결정 상태

- 적용: 결과 동일 패치 4개는 opt-in 플래그로 구현·측정 완료(기본 off). 실제 run `v2_pilot_002`에서 후보 1장 약 9분.
- 보류: TF32 — 전체 생성 이미지 품질 비교 전까지 off.
- 보류: (B) RegionDrag식 초기화 — DragFlow 설정 변형으로 기록해야 하며 대조군 포함 A/B 실험 필요. 추정 후보당 약 430 s → 275 s(N=20). LPIPS 측정 도구 설치 여부도 미정.
- 보류: (C) LazyDrag — 공식 코드 미공개(프로젝트 페이지만 존재), FLUX.1-Krea-dev(gated, non-commercial), H800 기준 49–62 GB. rotation은 대응 불가. 공식 코드 공개 시 재검토.
- 미해결: 같은 후보에서 T011(v2_pilot_001)의 forward가 이후 측정보다 약 4배 느렸던 원인.
