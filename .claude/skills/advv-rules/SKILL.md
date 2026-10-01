---
name: advv-rules
description: "ADVV 프로젝트에서 코드·문서·설정·테스트를 바꾸거나 검증하거나 조사할 때 반드시 먼저 읽는 공용 규칙과 검증 명령 모음. AGENTS.md 요약, 작업 카드 형식, 완료 보고 형식, conda env advv로 pytest/ruff 실행법, 문서 동기화 대상 표를 담는다. advv-researcher/implementer/qa/gpu-runner 에이전트가 작업 시작 시 사용. ADVV와 무관한 일반 파이썬 질문에는 쓰지 않는다."
---

# ADVV 공용 규칙

원문은 저장소 루트의 `AGENTS.md`다. 이 스킬은 요약이며, 충돌하면 AGENTS.md와 사용자의 최신 지시가 우선한다. 작업 전에 AGENTS.md를 한 번 읽는다. 요약만 보고 판단하면 세부 규칙(예: NO 재질문 금지, fake backend export 차단)을 놓치기 때문이다.

## 절대 규칙 (위반 시 결과물 전체가 무효)

- `../komap/`은 읽지도 쓰지도 않는다.
- `runs/`의 기존 run, `assets/` 원본, `models/`, `third_party/`의 upstream 파일을 덮어쓰거나 지우지 않는다. 새 실행은 새 `runs/<run_id>/`.
- 실제 DragFlow를 affine warp·복사·다른 생성기로 대체하지 않는다. fake backend는 tests fixture 전용.
- 판정은 파싱된 `answer == "YES"`만 인정한다. 문자열 포함 검색·정규식 부분 매칭 금지. NO/UNCERTAIN/파싱 실패/timeout/OOM은 서로 다른 결과.
- 같은 이미지의 NO를 다시 묻지 않는다. 목표 수량을 위해 판정 기준을 완화하지 않는다.
- GPU는 `advv-gpu-safety` 스킬 절차 없이 쓰지 않는다. 다른 사용자의 GPU에는 절대 올리지 않는다.
- 실행하지 않은 검사를 실행했다고 보고하지 않는다. 문서만 바꿨으면 "문서만"이라고 쓴다.

## 코드 규칙

- 패키지는 `src/advv/`. CLI·계약·생성기·검증기·저장소·평가를 분리한다.
- 외부 모델 import/로드는 backend 안에서 지연 수행한다. CPU 테스트가 CUDA·가중치 없이 돌아야 한다.
- 외부 프로그램은 인자 배열 + 명시적 `cwd`. 문자열 shell 조합 금지.
- 표준 logging, 타입 힌트, `errors.py`의 명시적 오류 유형. `except: pass` 금지.
- 주변 코드의 밀도·명명·관용구에 맞춘다(주석은 적고, 이유가 비자명할 때만).
- `src/advv/*.py`가 바뀌면 implementation hash가 바뀌어 기존 run은 `--resume`할 수 없다. 보고에 적는다.
- DragFlow(torch 2.5.1/tf 4.48)와 Qwen(torch 2.6/tf 5.17) 환경은 합치지 않는다. 새 모델은 별도 venv.

## 동작을 바꾸면 함께 갱신할 곳

| 바뀐 것 | 갱신 대상 |
| --- | --- |
| 요구사항·판정·수량·저장 규칙 | `SPEC.md` |
| 모듈·실행 순서·저장 구조 | `docs/ARCHITECTURE.md` |
| 진행 상태·체크리스트·검증 기록 | `docs/IMPLEMENTATION_PLAN.md` |
| 사용법·구현/예정 구분·테스트 수 | `README.md` |
| 설정 키 | `configs/advv.example.yaml` + `src/advv/config.py` 검증 |
| 외부 모델·revision·라이선스 | `docs/REFERENCES.md`, `configs/upstream.lock.json` |
| 영역 sampler v2 | `docs/REGION_SAMPLER.md` |

## 검증 명령

conda 환경 이름은 `advv`다(coordinator, CPU 전용).

```bash
cd /data2/donginson/projects/ADVV
conda run -n advv python -m pytest -q
conda run -n advv python -m ruff check src scripts tests
```

`conda run`은 heredoc stdin을 전달하지 않는다. 스크립트를 stdin으로 넣을 때는 `/data2/donginson/conda/envs/advv/bin/python -`을 쓴다.

문서 링크 점검:

```bash
for f in SPEC.md README.md docs/*.md; do grep -oE '\]\(([^)#h][^)#]*)' $f | sed 's/](//' | while read l; do [ -e "$(dirname $f)/$l" ] || echo "BROKEN $f -> $l"; done; done
```

## 작업 카드 (오케스트레이터 → 에이전트)

```text
TASK <id>: <한 줄 목표>
범위: <건드려도 되는 파일/디렉터리>
금지: <건드리면 안 되는 것>
입력: <읽을 파일, 이전 산출물 경로>
완료 기준: <테스트·산출물·확인 항목>
출력: _workspace/<id>_<agent>_<artifact>.md
```

## 완료 보고 (에이전트 → 오케스트레이터)

산출물 파일 첫 부분과 반환 메시지에 같은 형식으로 쓴다.

```text
결과: done | partial | blocked
변경 파일: <목록 또는 없음>
실행한 검사: <명령과 결과 수치> (실행 안 한 것은 "미실행"으로 명시)
남은 것 / 막힌 이유: <목록>
사용자 결정 필요: <질문 또는 없음>
```
