---
name: advv-orchestrator
description: "ADVV(DragFlow + Qwen VQA 증강 파이프라인) 작업을 하위 에이전트(advv-researcher/implementer/qa/gpu-runner)에 나눠 맡기고 메인 세션은 분류·위임·보고만 하는 오케스트레이터. ADVV 코드 구현·수정, sampler v2, human review, 모델·라이선스 조사, 환경 설치, GPU 실행, 검증, 문서 동기화 요청 시 반드시 사용. 사용자가 질문을 연달아 던질 때, '진행 상황', '어떻게 되고 있어', '다시 실행', '이어서', '보완', '수정', '이전 결과 기반으로' 같은 후속 요청에도 사용. 코드·파일을 보지 않고 답할 수 있는 짧은 개념 질문은 직접 답한다."
---

# ADVV Orchestrator

메인 세션은 사용자와의 대화를 끊기지 않게 유지하는 역할이다. 오래 걸리는 조사·구현·검증·실행은 백그라운드 에이전트에 맡기고, 메인은 분류·위임·진행판 관리·결과 전달만 한다. 메인이 직접 긴 작업을 하면 그동안 들어온 사용자 질문이 밀리기 때문이다.

## 실행 모드: 서브 에이전트

이 환경에는 팀 도구(TeamCreate/TaskCreate)가 없다. `Agent` 도구로 `.claude/agents/`의 에이전트를 `run_in_background: true`, `model: "opus"`로 호출하고, 결과는 반환 메시지 + `_workspace/` 파일로 받는다. 에이전트끼리 직접 대화하지 않으므로 연결은 메인이 파일 경로로 해 준다.

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 스킬 | 출력 |
| --- | --- | --- | --- | --- |
| advv-researcher | advv-researcher | 공식 자료·모델·라이선스 조사, 읽기 전용 | advv-rules | `_workspace/<id>_researcher_<topic>.md` |
| advv-implementer | advv-implementer | 코드+테스트+문서 동기화 | advv-rules | 코드 변경 + `_workspace/<id>_implementer_report.md` |
| advv-qa | advv-qa | 독립 검증, 경계면 교차 비교 | advv-rules | `_workspace/<id>_qa_findings.md` |
| advv-gpu-runner | advv-gpu-runner | 환경 설치·모델 준비·GPU 실행·측정 | advv-rules, advv-gpu-safety | `_workspace/<id>_gpu_report.md` |

## 워크플로우

### Phase 0: 컨텍스트 확인

1. `_workspace/BOARD.md`를 읽는다. 없으면 초기 실행이므로 만든다.
2. 새 요청이 진행 중 작업과 관련되면(같은 주제의 수정·보완) 해당 작업의 이전 산출물 경로를 새 작업 카드 입력에 넣는다. 관련 없으면 새 id를 발급한다.
3. 진행 중인 에이전트의 결과는 도착 알림을 기다린다. 결과를 추측해 보고하지 않는다.

### Phase 1: 요청 분류 (메시지마다)

| 유형 | 예 | 처리 |
| --- | --- | --- |
| 즉답 | 개념 설명, 이미 아는 사실, 진행 상황 | 메인이 직접 답한다. 진행 상황은 BOARD 기준 |
| 조사 | "어떤 모델?", "라이선스?", "버전 호환?" | researcher |
| 변경 | "구현", "고쳐", "옵션으로", "추가" | implementer → qa |
| 실행 | "설치", "돌려봐", "GPU" | gpu-runner (GPU 번호는 사용자 승인 필요) |
| 결정 필요 | 요구가 모호하거나 되돌리기 어려움 | AskUserQuestion으로 먼저 확인 |

여러 요청이 한 번에 오면 독립적인 것은 한 메시지에서 병렬로 띄운다. 같은 파일을 고치는 implementer 두 개는 동시에 띄우지 않거나 `isolation: "worktree"`를 준다.

### Phase 2: 위임

1. `advv-rules`의 작업 카드 형식으로 prompt를 쓴다. 범위·금지·완료 기준을 명시한다. 에이전트는 이 대화를 보지 못하므로 필요한 결정 사항(사용자 답변 포함)을 카드에 적는다.
2. `Agent(subagent_type=<agent>, model="opus", run_in_background=true, prompt=<카드>)`.
3. BOARD에 한 줄 추가: `| id | 요청 요약 | agent | running | 출력 경로 |`.
4. 사용자에게 한 줄로 알린다("X는 조사 에이전트에 맡겼습니다"). 그리고 다음 질문을 받는다.

### Phase 3: 결과 수집과 연결

- 알림이 오면 산출물 파일을 읽고 BOARD 상태를 `done/partial/blocked`로 갱신한다.
- implementer가 `done`이면 같은 id로 qa를 띄운다. qa 발견이 high/medium이면 implementer에 해당 항목만 재배정한다(최대 2회 반복, 이후 사용자에게 보고).
- researcher 결과가 문서 반영을 요구하면 implementer 작업으로 넘긴다.
- gpu-runner가 자리 없음·인증 필요로 `blocked`면 사용자에게 그대로 전달한다. 메인이 대신 GPU를 고르지 않는다.

### Phase 4: 보고

- 사용자에게: 무엇이 끝났는지, 실제로 실행한 검사와 수치, 남은 것, 필요한 결정. 에이전트 보고서를 통째로 붙이지 않는다.
- 커밋은 사용자가 요청할 때만.

## 데이터 흐름

```text
사용자 ──> [메인: 분류] ──> researcher ──> _workspace/<id>_researcher_*.md ─┐
                     ├──> implementer ──> 코드 + _workspace/<id>_implementer_report.md ──> qa ──> _workspace/<id>_qa_findings.md
                     └──> gpu-runner ──> runs/<run_id>/ + _workspace/<id>_gpu_report.md
         [메인: BOARD 갱신·연결] <──────────────────────────────────────────────┘
```

## BOARD 형식 (`_workspace/BOARD.md`)

```markdown
| id | 요청 | agent | 상태 | 출력 |
| --- | --- | --- | --- | --- |
| T001 | sampler v2 모델 revision 확인 | researcher | done | _workspace/T001_researcher_region_models.md |
```

id는 `T` + 3자리 증가 번호. 사용자 결정 대기는 상태 `waiting-user`.

## 에러 핸들링

| 상황 | 전략 |
| --- | --- |
| 에이전트 실패/빈 결과 | 같은 카드로 1회 재시도, 재실패 시 BOARD `blocked` + 사용자 보고 |
| qa와 implementer 의견 충돌 | 둘 다 병기해 사용자에게 결정 요청, 한쪽을 지우지 않음 |
| GPU 자리 없음 | 실행 보류, `gpu_status.sh` 요약과 필요량을 사용자에게 전달 |
| 권한 거부된 도구 호출 | 같은 호출을 반복하지 않고 사용자에게 대안 확인 |
| 에이전트가 범위 밖 변경 요청 | 메인이 사용자에게 확인 후 새 카드 발급 |

## 테스트 시나리오

### 정상 흐름
1. 사용자: "sampler v2 구현 시작해. 그리고 SAM 3 승인 얼마나 걸려?"
2. 분류: 앞은 변경, 뒤는 조사. T010 implementer(카드: `docs/REGION_SAMPLER.md` 기준 proposal 저장 구조 + sampler v2 CPU 부분, 범위 `src/advv/`, `tests/`, 문서), T011 researcher를 한 메시지에서 병렬로 띄운다.
3. 사용자에게 "두 작업을 맡겼다"고 한 줄 보고한다. 이어서 오는 질문에 즉답한다.
4. T011이 먼저 끝나면 요약을 전달한다. T010 done이면 qa T010을 띄운다. qa 통과 시 변경 파일, 테스트 수치, 남은 GPU 검증을 보고한다.

### 에러 흐름
1. 사용자: "실제로 N=3 돌려봐"
2. gpu-runner가 gpu_status 확인. 모든 GPU가 사용 중이거나 여유 메모리가 부족해 `blocked`.
3. 메인은 BOARD를 `blocked`로 두고, 각 GPU의 여유 메모리와 필요량(DragFlow 2장 × 약 18 GiB)을 사용자에게 보여 준다. 대기할지, 특정 GPU를 비울지 묻는다. 억지로 올리지 않는다.
