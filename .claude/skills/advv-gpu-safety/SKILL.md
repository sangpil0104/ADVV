---
name: advv-gpu-safety
description: "공용 서버에서 ADVV의 GPU 작업(DragFlow 생성, Qwen 추론, SAM/Grounding DINO proposal, check_qwen.py, advv run/preflight --gpus)을 올리기 전에 반드시 사용하는 GPU 소유자 확인·배정 절차. GPU 번호 선택, CUDA_VISIBLE_DEVICES 지정, 'GPU 몇 번 비었어', '실행해줘', '통합 테스트 돌려' 요청 시 트리거. CPU 테스트(pytest)에는 필요 없다."
---

# 공용 GPU 안전 절차

이 서버는 여러 사용자가 함께 쓴다. 메모리 사용량만으로는 누구의 GPU인지 알 수 없으므로 PID를 사용자에 매핑해 확인한다. 점유 상황은 수시로 바뀌므로 GPU 번호를 기억해 두지 말고 매번 새로 확인한다. `~/.claude/hooks/gpu-guard.py` 훅이 마지막 방어선으로 차단하지만, 훅에 기대지 말고 이 절차를 먼저 지킨다.

## 절차

1. 상태 확인 (읽기 전용):
   ```bash
   .claude/skills/advv-gpu-safety/scripts/gpu_status.sh
   ```
   GPU별 total/used/free MiB와 프로세스 소유자, 마지막 줄에 내 사용자명(`me=`)을 출력한다.
2. 후보 선정:
   - 다른 사용자 프로세스가 하나라도 있는 GPU는 제외한다.
   - 내 다른 프로젝트가 쓰는 GPU도 여유 메모리가 필요량보다 작으면 제외한다. OOM으로 내 다른 실험을 죽이기 때문이다.
   - 필요량 참고(관측값, 보장 아님): DragFlow 생성 GPU 2장, 공식 경로 장당 피크 약 18.1 GiB. `generator.speedups` exact_all이면 첫 GPU 약 37.6 GiB reserved(빈 48 GB 카드 필요), 둘째 약 19 GiB. Qwen3.5-4B 약 9 GiB. 여유를 20% 이상 둔다.
3. 자리가 없으면 억지로 밀어넣지 않는다. 실행을 보류하고 `gpu_status.sh` 출력과 필요량을 오케스트레이터에 보고한다. 사용자 결정이 필요하다.
4. 실행 직전에 1을 다시 실행해 선택한 GPU가 여전히 비어 있는지 확인한다.
5. 실행은 사용자가 지정했거나 승인한 GPU 번호만 `--gpus`로 넘긴다. 발견된 모든 GPU를 자동으로 쓰지 않는다.
6. 실행 중 다른 사용자와 겹친 것을 발견하면 **내 프로세스만** 즉시 내린다. 남의 프로세스는 어떤 경우에도 죽이지 않는다.
7. 보고에 사용한 GPU 번호, 확인 시각, 관측 피크 메모리를 남긴다.
8. 백그라운드로 띄운 `advv run`을 멈출 때는 SIGTERM을 보낸다. 비대화형 셸에서 `nohup … &`로 띄운 프로세스는 SIGINT를 무시한다. ADVV CLI는 SIGTERM을 중단(interrupted, 재개 가능)으로 처리한다. 내 advv 메인 프로세스에만 보낸다.
