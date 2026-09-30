# DragFlow 편집 설정 시각화

상태: **CPU 렌더러 구현·검증 완료, 실제 DragFlow N=1 실행의 적용값·비교 그림 확인**. `src/advv/visualization.py`와 `configs/advv.example.yaml`을 사용한다. 로컬 미리보기와 실행 이미지는 Git에 포함하지 않으며 [검증 범위](RUN_VERIFICATION.md)를 참고한다.

## 1. 기본 화면

각 후보를 생성할 때 다음 PNG를 저장한다. 별도 웹 서버 없이 이미지 뷰어에서 확인한다.

```text
┌──────────────────────────────┬──────────────────────────────┐
│ Original + drag settings     │ Generated image              │
│                              │                              │
│ 파란 반투명 area와 외곽선     │ 표시 없는 생성 결과          │
│ 빨간 S ────→ 초록 E         │                              │
│ start_point   end_point      │                              │
└──────────────────────────────┴──────────────────────────────┘
ID / operation / seed / start(x,y) / end(x,y) / area(px,%)
Physical: YES·NO·UNCERTAIN·PENDING·ERROR·NOT_RUN
Semantic: YES·NO·UNCERTAIN·PENDING·ERROR·NOT_RUN / final status
```

이미지는 두 패널에 같은 배율로 표시하고 종횡비를 유지한다. 캡션·좌표·범례·상태는 이미지 바깥 여백에 두어 내용을 가리지 않는다. 긴 ID나 문장은 줄바꿈하며 바깥으로 잘라내지 않는다. 축소는 미리보기 크기에만 적용하고 원본·생성 파일은 변경하지 않는다.

| 사용자 표기 | 기록의 기준 | 기본 표시 |
| --- | --- | --- |
| `start_point` | `source_point` | 빨간 원 + `S`, 좌표 `(x,y)` |
| `end_point` | `target_point` | 초록 십자 + `E`, 좌표 `(x,y)` |
| `area` | `region_mask_path`의 실제 mask | 파란 반투명 채움 + 외곽선 |
| 이동 방향 | start → end | 노란 화살표 + 대비되는 외곽선 |
| `anchor` | `anchor_point`, 있는 경우 | 보라색 마름모 + `A` |

색상만으로 구분하지 않도록 기호와 범례를 병기한다. 영역은 bounding box로 대체하지 않고 mask의 실제 모양을 표시한다. 면적은 mask의 nonzero pixel 수와 전체 이미지 대비 비율이다. 점이 가깝거나 가장자리에 있어도 라벨과 화살표 끝이 서로 가려지지 않게 배치한다.

화살표는 **요청한 start→end 관계**를 뜻하며 생성 결과에서 그 위치에 실제 도달했음을 증명하지 않는다. 오른쪽 생성 이미지에는 기본적으로 점·화살표를 덧그리지 않는다. rotation/deformation에서는 operation과 요청 회전각·이동량·anchor도 기록하여 단순 평행 이동으로 오인하지 않게 한다.

## 2. 실제 전달값과 좌표 정합성

시각화 함수 안에서 영역·점을 다시 무작위로 뽑지 않는다. 저장된 계획과 backend adapter가 남긴 effective instruction에서 값을 읽는다. internal field와 화면 표기의 관계는 위 표로 고정하며 `start_point`와 `source_point`를 독립 수정 가능한 두 값으로 저장하지 않는다.

좌표는 EXIF 방향 정규화 후 원본의 좌상단을 원점으로 하는 `[x,y]`다. backend의 리사이즈·패딩·양자화된 좌표/영역을 정규화 원본으로 역변환한 뒤, 미리보기 패널의 배율과 여백 offset을 적용한다. mask는 nearest-neighbor로 변환하고 반투명 채움은 그 이후 수행한다. 화면상 좌표와 footer 수치는 서로 다른 단위가 되지 않도록 footer에는 **정규화 원본 px**라고 명시한다.

`metadata.json`에 최소한 source/candidate/plan ID와 hash, 정규화 원본 크기, backend 입력 크기, 요청 좌표·mask hash, 실제 전달 좌표·mask hash, 전처리/역변환, 표시 좌표, 면적과 계산 좌표계, seed, operation, 두 VQA 상태, 렌더링 버전·설정 hash·파일 경로를 남긴다. 요청값과 실제값이 다르면 그 사실과 두 값을 footer 또는 보조 설명에 함께 표시한다.

`area`는 선택한 편집 영역을 뜻한다. upstream이 별도로 확장한 gradient mask나 추정한 target region이 있으면 이름과 범례를 구분한다. 선택 영역이 생성 중 변화한 모든 픽셀과 정확히 같다고 표시하지 않는다. 실제 전달값을 확보하지 못했다면 요청값으로 그린 그림에는 `REQUESTED / NOT EXECUTED`를 표시하고 실행 설정 검증을 완료했다고 보고하지 않는다.

## 3. 저장 위치와 시점

```text
runs/<run_id>/visualizations/<candidate_id>/
├── drag_plan.png     # 원본 + 실제 start/end/area 표시
├── comparison.png    # 왼쪽 drag_plan의 이미지, 오른쪽 생성 결과, 하단 정보
└── metadata.json     # 좌표·mask·전처리·판정·렌더링 상태
```

계획 저장 후 요청값 기준 preview를 만들 수 있다. 실제 backend 입력이 확정되면 drag_plan을 갱신하고, 정상 생성 후 comparison을 저장한다. 아직 VQA를 실행하지 않았으면 `PENDING`, 선행 탈락으로 건너뛴 검증은 `NOT_RUN`이다. 최종 판정 후 동일 후보의 footer와 metadata만 원자적으로 갱신한다.

시각화는 원본·생성 이미지를 복사하여 그린다. **모델 입력과 dataset export는 표시 없는 파일**을 참조한다. visualization 경로는 별도 artifact 링크로만 기록하고 학습 manifest의 `image_path`나 hash·중복 검사·목표 N에 섞지 않는다. report에서 candidate ID별 확인용 파일 링크를 제공한다.

## 4. 판정별 보존과 복구

| 상태 | 원본 설정 PNG | 생성 결과 포함 comparison PNG |
| --- | --- | --- |
| accepted | 보존 | 보존 |
| uncertain / 검증 오류 | 보존 | 해당 후보 이미지 보관 정책에 따라 보존 |
| rejected, 기본 NO 삭제 활성 | 보존 | 삭제; 썸네일·임시 복사본도 함께 삭제 |
| rejected, 명시적 연구용 이미지 보관 | 보존 | 보존 가능; REJECTED 표시 |
| 생성 실패, 후보 없음 | 요청값 preview와 실패 상태 보존 | 생성하지 않음 |

`visualization.rejected_comparison_policy=follow_image_retention`으로 기존 생성 이미지 보관 정책을 따른다. 원본 설정 PNG에는 후보의 픽셀을 포함하지 않으므로 NO 삭제 후에도 실패한 설정을 볼 수 있다. 삭제한 comparison을 재개/리포트 생성 때 복원하거나 dangling 링크를 남기지 않는다.

렌더링 실패는 `visualization_error`로 별도 기록하고 원본·후보·VQA 판정은 보존한다. 시각화 복구를 위해 DragFlow나 Qwen을 다시 실행하지 않는다. 기본 활성화 상태에서는 필요한 시각화 저장까지 완료해야 run 완료로 보고하며, 이미 채운 N장을 추가로 생성하지 않는다.

## 5. 구현 시 확인 사항

- 직사각형 이미지, EXIF 회전, 패딩·리사이즈, 가장자리 좌표에서 점·mask가 같은 위치를 가리키는지 확인한다.
- backend 좌표 반올림 후에도 실제 전달값을 표시하고 mask 면적 계산의 좌표계를 기록한다.
- 가까운 start/end, 작은 area, anchor가 있는 회전·변형의 표시가 읽히는지 확인한다.
- 원본·후보 hash와 VQA 실제 입력이 렌더링 전후 동일한지 검사한다.
- NO 삭제 후 comparison/thumbnail에 후보 픽셀이 남지 않고, source-only drag_plan은 유지되는지 검사한다.
- 시각화만 실패한 뒤 재개해도 generation/verification 호출과 accepted count가 늘어나지 않는지 검사한다.

고정 upstream의 deformation에는 독립적인 scale 입력이 없다. displacement로 설정하며 이를 scale 파라미터처럼 표시하지 않는다. 실제 feature 영역의 bilinear soft mask는 `effective_region.npy`, nonzero 영역을 원본 크기로 nearest 변환한 표시는 `effective_area.png`로 저장한다.
