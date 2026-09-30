# ADVV 평가 계획

## 1. 두 단계 목표

**1차 목표**는 총 N장의 현실적이고 원본 의미가 유지된 증강 이미지를 자동 생성하는 파이프라인이다. 목표 수량 달성, 이미지 품질, 의미 보존, 다양성, 처리 비용, 중단·재개의 동작을 먼저 평가한다. **2차 목표**로 후속 모델의 일반화 성능 향상을 검증한다.

현재 GT 라벨은 없다. 1차 평가를 위해 전체 데이터에 box/mask/class GT를 만들 필요는 없다. 소규모 원본·생성 쌍의 현실성과 의미 유지 여부를 사람이 판정한 평가 표본부터 준비한다. Qwen에게 다시 묻는 것만으로 품질을 검증하지 않는다.

## 2. 1차 파이프라인 지표

원본별 attempt를 집계하며 후보 자체가 바뀌는 새 생성 시도와 동일 후보의 기술 재시도를 구분한다. 생성 실패도 attempt에 포함한다. preflight/profile 실패는 candidate 통계와 별도로 보고한다.

```text
completion_ratio = N_exported / N_target
success = (N_exported == N_target and final_manifest_saved)
generation_success_rate = N_generated / N_attempted
physical_yes_rate = N_physical_yes / N_physical_valid_responses
semantic_yes_rate = N_semantic_yes / N_semantic_valid_responses
joint_accept_rate = N_both_yes / N_generated
export_yield = N_exported / N_attempted
```

유효 응답 분모에는 해당 check의 YES/NO/UNCERTAIN을 포함하고 오류·not_run은 제외한다. semantic은 physical YES 후보에서만 수행하는 기본 구조이므로 **조건부 지표**임을 표시한다. 분모 0이면 null/N/A다. 오류율·불확실률도 check별로 따로 보고한다.

총 목표 N, 실제 고유 채택 수, NO/UNCERTAIN/기술 실패·중복 수, 원본별 채택 기여, source 보류 수, 사용자 중단 여부, 경과 시간, 후보당 생성/검증 시간, 목표 달성까지의 비용, peak VRAM을 기록한다. 중단된 실행의 일부 결과를 목표 달성으로 보고하지 않는다.

정상 실행에는 총 시도/시간 제한이 없다. 성능 벤치마크나 CI에서 별도 관측 종료 시점을 두면 평가 절차의 중단임을 명시하고 production 설정에 숨겨 넣지 않는다. 낮은 수락률의 원본이 덜 기여하는 분포 변화도 보고한다.

## 3. 현실성과 의미 보존의 독립 검수

검수자는 Qwen 판정을 모르는 상태에서 원본·후보를 보고 두 항목을 별도로 평가한다. 산업 결함·사고와 희귀 생물 각각의 실패 사례를 포함한다.

| 검수 항목 | 예시 |
| --- | --- |
| 현실성·물리적 개연성 | 불가능한 접촉·구조·재질 변형이 생겼는가 |
| 사고 의미 | 충돌/전도 등 핵심 사건이 유지되는가; 정상 주차로 바뀌지 않았는가 |
| 결함 의미 | 결함이 제거되거나 다른 결함 유형으로 바뀌지 않았는가 |
| 생물 정체성 | 구별 특징이 유지되는가; 다른 생물처럼 바뀌지 않았는가 |
| 희소하지만 가능한 장면 | 특이함·손상·위험한 상황 자체 때문에 탈락하지 않았는가 |
| 다양성·실제 변화 | 원본과 거의 같거나 동일 패턴의 후보만 반복하지 않는가 |

정확한 결함 종류나 종 식별처럼 전문 지식이 필요하면 독립 전문가 또는 확인 가능한 메타데이터로 검수한다. 사람도 판단할 수 없으면 uncertain으로 두며 임의의 정답 NO로 바꾸지 않는다. 원본 자동 profile의 오류도 별도 분석한다.

주요 품질 지표는 채택 이미지의 공동 적합률(두 조건 모두 충족), 물리 오수락률, 의미 변경 오수락률, 적절한 희소 이미지의 오배제율이다. sample 수와 검수 불확실률을 같이 보고한다. 층화 표본이면 추출 확률과 가중치를 기록한다.

exact duplicate는 출력에서 제외한다. near-duplicate와 변화량은 먼저 분석 지표로 사용하며 threshold를 사후 변경해 결과를 좋게 보이지 않는다. 작은 변화만 반복하여 N을 쉽게 채우는 현상을 실패 분석에 포함한다.

## 4. VQA 구성의 효과 분리

추가 검증 질문이 유용한지 보려면 동일한 **고정 후보 pool**에서 아래 필터를 비교한다.

| 비교군 | 후보 선택 |
| --- | --- |
| F0 | VQA 없음 |
| F1 | 현실성·물리 검증만 |
| F2 | 원본 의미 보존 검증만 |
| F3 | 두 조건 모두 YES인 ADVV |

각 방식으로 N장을 채우도록 독립 생성한 결과만 비교하면 시도 수와 후보 난이도가 달라져 필터 효과와 생성 예산이 섞인다. 필터 비교에는 고정 pool을 사용하고, production의 목표 수량 반복 성능은 별도로 측정한다.

F2를 평가할 때는 physical 탈락 후보에도 semantic을 실행하는 **evaluation protocol**을 명시한다. production에서 skip된 semantic을 NO로 간주하거나 추정하여 채우지 않는다.

기본 NO 삭제 정책으로는 탈락 이미지 검수가 불가능하다. 검수/ablation용 run은 사전에 `storage.delete_rejected_images=false`로 설정하여 별도 연구 공간에 보관한다. 기본 사용자 run의 NO 삭제 요구를 바꾸지 않으며, 이미 삭제된 후보를 완전히 복원할 수 있다고 주장하지 않는다.

## 5. 2차 후속 성능 평가

과제가 정해지면 원본-only, 일반 증강, DragFlow-only, 두 검증 ADVV, 같은 양의 무작위 후보 선택을 비교한다. 같은 후보 pool과 동등한 라벨 유효성 기준을 적용한다. 라벨 수량·원본/증강 비율·학습 step budget·seed·모델을 기록한다.

증강 전에 독립 group 단위 train/val/test를 고정한다. dataset-building으로 이미 만들어진 파생본도 원본과 같은 group에 두며, 최종 테스트는 가능한 한 독립적인 실제 이미지로 구성한다. 평가 데이터를 보고 verifier prompt·sampler를 조정하지 않는다.

classification/detection/segmentation에 필요한 GT는 이 단계에서 작성·검수한다. 의미 YES만으로 classification label을 확정하거나 원본 box/mask를 그대로 복사하지 않는다. 주 지표와 absent-class 처리는 과제별로 사전에 고정한다.

반복 seed별 결과와 불확실성을 보고하고, 데이터가 허용하면 group 단위 신뢰구간을 계산한다. Qwen fine-tuning은 domain 오류 분석 뒤 별도 학습/평가 split을 갖춘 선택적 연구로 둔다. 성능 향상이 없거나 품질이 떨어진 결과도 보존한다.
