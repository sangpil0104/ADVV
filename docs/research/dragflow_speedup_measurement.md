# T018 GPU report: DragFlow speedup measurement, stages 0 to 2

Result: done. All three stages finished on GPU 6 and GPU 9 with no OOM and no errors. No other user appeared on either GPU.
Changed files: only new outputs under `_workspace/T018_*`. src/, tests/, docs/, configs/, runs/ and assets/ were not touched, nothing was downloaded and nothing was committed.

## Execution record
- GPUs: physical 6 and 9 (RTX A6000 48 GB ×2). Every command used `CUDA_VISIBLE_DEVICES=6,9` and `--gpus 6,9`, so inside the process cuda:0 = physical 6 and cuda:1 = physical 9.
- Ownership checks (`gpu_status.sh`): 08:49:35Z, 08:49:56Z (stage 0 baseline), about 08:51Z (stage 0 exact_all), about 08:56Z (stage 1) and about 09:08:40Z (stage 2). Every check showed GPU6/9 `owners: none, free 49136MiB`.
- Final check at 09:16:53Z: GPU6/9 `used=4MiB owners: none`. None of my measurement processes or the monitor were still running.
- Monitor: `_workspace/T018_gpu_monitor.txt` sampled every 60 s, 27 times from 08:49:48 to about 09:16. No other user was detected and no SIGTERM was sent. The monitor script is `_workspace/T018_logs/monitor.sh`.
- Between stages I confirmed that my process had left the GPUs (used=4MiB) before starting the next one.
- Progress log: `_workspace/T018_progress.md`. Raw logs: `_workspace/T018_logs/stage0_{baseline,exact_all}.log`, `stage1.log` and `stage2.log` (each ends with `EXIT_CODE=`).
- Input candidate (read-only): `runs/v2_pilot_001` / `src_cfe95f7139096ee7_00000000_111434dfad`, seed 1529091675, 960×720 rotation, 5 drag rounds (all TRANSPORT). torch 2.5.1+cu124, `.venv-dragflow`.

| Stage | Command (summary) | Time (UTC) | Exit |
|---|---|---|---|
| 0 | profile baseline → `_workspace/T018_profile/baseline/profile.json` | 08:49:56–about 08:51 (wall 85 s) | status completed (I did not capture the process exit code; the JSON status is completed) |
| 0 | profile `--speedups exact_all` → `_workspace/T018_profile/exact_all/profile.json` | about 08:51–08:56 | status completed |
| 1 | equivalence, 4 cumulative arms + control → `_workspace/T018_equivalence_stage1/equivalence.json` | 08:56–09:08 (about 12 min) | **0** |
| 2 | equivalence `--no-control --arm tf32 --arm exact_all --arm exact_all,tf32` → `_workspace/T018_equivalence_stage2/equivalence.json` | 09:08–09:16:53 (about 8 min) | **2** (see interpretation below) |

The commands are the same as §"측정 명령" in T017fix_implementer_report.md, with A,B=6,9.

## Stage 0: time breakdown (profile, 3 inversion steps, 5 rounds, synchronized timers)

| Item | baseline | exact_all |
|---|---|---|
| Median round time (excluding the first) | 8.70 s | 3.08 s (**2.8×**) |
| Forward with no_grad, median | 3.39 s | 3.42 s (unchanged) |
| Forward with grad, median | 3.44 s | 1.18 s (single blocks skipped, no checkpointing) |
| Backward, median | 2.83 s | 1.57 s (no checkpoint recomputation) |
| reclaim_memory | 67 calls 12.10 s (2.42 s/round) | 0.001 s |
| Python GC | 2609 collections 12.85 s | 1595 collections 1.19 s |
| Device time per drag round (profiler, sum over both GPUs) | 6.66 s: gemm 3.44, attention 1.56, other 0.86, **memcpy_peer 0.52**, HtoD 0.25 | 3.06 s: gemm 1.41, attention 1.00, other 0.42, memcpy_peer 0.14, HtoD 0.07 |
| Wall time of a drag round under the profiler | 8.48 s (about 1.8 s of host/sync wait beyond device time) | 3.08 s (almost all device time) |
| F_orig forward (no_grad) | wall 4.47 s / device 3.55 s (memcpy_peer 0.23) | wall 3.56 s / device 3.55 s (memcpy_peer 0.06, HtoD 0.16) |
| Peak allocated GiB, GPU6 / GPU9 | 16.68 / 18.08 | **34.83** / 16.30 |
| Peak reserved GiB, GPU6 / GPU9 | 17.11 / 18.52 | **37.55** / 18.94 |

Interpretation:
- In the baseline round, about 2.4 s goes to reclaim_memory (empty_cache plus gc), about 0.8 s to device copies (peer plus HtoD) and about 2 s to gradient-checkpointing recomputation and single-block forwards. exact_all removes most of each.
- The no_grad forward (inversion and sampling) is not sped up by the exact patches.

## Stage 1: equivalence (official inversion 25 steps, 5 rounds, control included) — exit 0

- Noise floor (control_baseline against baseline, maximum relative difference): loss 8.28e-6, latent 8.93e-5. The inversion latent was bit-identical.
- Exact tolerance = max(1e-5, 10 × noise floor): loss 8.28e-5, latent 8.93e-4.

| arm | Status | Judgement | Median s/round | Speed ×(round) | Inversion s | Peak alloc GiB 6/9 | Peak reserved GiB 6/9 | Worst loss rel. diff | Worst latent rel. diff |
|---|---|---|---|---|---|---|---|---|---|
| baseline | completed | reference | 8.605 | 1.00 | 76.9 | 16.68 / 18.08 | 17.11 / 18.52 | – | – |
| control_baseline | completed | noise floor | 8.433 | 1.02 | 78.5 | 16.69 / 18.08 | 17.11 / 18.53 | 8.3e-6 | 8.9e-5 |
| skip_unused_single_blocks | completed | **within tolerance** | 6.045 | **1.42** | 77.4 | 16.69 / 18.08 | 17.11 / 18.53 | 1.30e-5 | 5.39e-5 |
| + disable_gradient_checkpointing | completed | **within tolerance** | 4.900 | **1.76** | 76.8 | 29.43 / 18.66 | 30.04 / 19.17 | 1.76e-5 | 5.52e-5 |
| + ip_adapter_on_transformer_device | completed | **within tolerance** | 4.907 | **1.75** | 72.8 | 34.78 / 16.31 | 35.71 / 16.75 | 4.1e-6 | 3.87e-5 |
| exact_all (+ light_reclaim_memory) | completed | **within tolerance** | 3.095 | **2.78** | 73.1 | 34.83 / 16.31 | 37.55 / 18.97 | 1.10e-5 | 3.56e-5 |

- For every exact arm, the inversion latent difference is 0 and every difference is within or close to the control noise floor.
- `ip_adapter_on_transformer_device` alone made no measurable difference to round speed (1.76 vs 1.75×). It moved 5.4 GiB onto GPU6 and freed 1.8 GiB on GPU9.
- Most of the gain from adding light_reclaim to reach exact_all comes from removing reclaim_memory and gc (4.9 → 3.1 s).

## Stage 2: TF32 (`--no-control`) — exit 2

| arm | Status | Judgement | Median s/round | Speed ×(round) | Inversion s | Peak alloc GiB 6/9 | Peak reserved GiB 6/9 | Worst loss rel. diff | Worst latent rel. diff | Inversion latent rel. diff |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | completed | reference | 8.357 | 1.00 | 76.9 | 16.68 / 18.08 | 17.11 / 18.52 | – | – | – |
| tf32 | completed | precision variant (within diagnostic tolerance) | 5.872 | **1.42** | **47.1** | 16.69 / 18.08 | 17.11 / 18.53 | 4.44e-3 | 2.93e-2 | 2.93e-2 (max abs 0.119) |
| exact_all | completed | exact, **outside rtol 1e-5** | 3.096 | **2.70** | 74.9 | 34.83 / 16.31 | 37.55 / 18.97 | 4.67e-6 | 5.84e-5 | 0 |
| exact_all,tf32 | completed | precision variant (within diagnostic tolerance) | 2.137 | **3.91** | **41.1** | 34.83 / 16.31 | 37.55 / 18.97 | 4.68e-3 | 2.93e-2 | 2.93e-2 |

Interpreting exit 2:
- Without a control run, the script judges against rtol 1e-5 alone.
- exact_all's latent difference of 5.84e-5 exceeds 1e-5 but is below the stage 1 noise floor of 8.93e-5, the difference between two official baseline runs.
- Under the stage 1 tolerance (8.93e-4) this is within tolerance. Stage 1 also judged the same arm within tolerance at 3.56e-5. **This is noise, not a numerical-equivalence failure.**
- Per round, the exact_all latent difference grows: 2.9e-8 → 1.2e-7 → 2.6e-6 → 1.6e-5 → 5.8e-5. The control run shows the same growth pattern.

TF32 numbers (no conclusion about image quality):
- The inversion latent relative difference is already 2.9e-2 before any drag round, about 330× the stage 1 latent noise floor. It stays around 2.84–2.90e-2 through the rounds.
- The loss relative difference is between 8.1e-5 and 4.7e-3 per round.
- TF32 also speeds up the no_grad forward: inversion 77 → 47 s, and 41 s together with exact_all.
- Whether this difference affects image quality needs a full-generation comparison, which is outside this task.

## Summary table

| Flag combination | s/round | Round speed × | Inversion | Peak GiB GPU6 / GPU9 (reserved) | Equivalence |
|---|---|---|---|---|---|
| Official (none) | 8.4–8.7 | 1.0 | 77 s | 17.1 / 18.5 | reference |
| skip_unused_single_blocks | 6.05 | 1.42 | same | 17.1 / 18.5 | equivalent (within noise) |
| + disable_gradient_checkpointing | 4.90 | 1.76 | same | 30.0 / 19.2 | equivalent |
| + ip_adapter_on_transformer_device | 4.91 | 1.75 | same | 35.7 / 16.8 | equivalent |
| exact_all | 3.10 | 2.70–2.78 | same | 37.6 / 19.0 | equivalent (stage 1 exit 0; the stage 2 exit 2 is within noise) |
| tf32 | 5.87 | 1.42 | 47 s | 17.1 / 18.5 | precision variant, latent 2.9e-2 |
| exact_all,tf32 | 2.14 | 3.91 | 41 s | 37.6 / 19.0 | precision variant, latent 2.9e-2 |

## Recommendation
- **Production default: `exact_all`.** Round time is 2.7–2.8× faster, results match within the noise floor, and there was no OOM.
  - Condition: the first GPU (cuda:0) needs a peak of 37.6 GiB reserved, so it needs at least about 45 GiB free (20% margin). In practice that means an empty 48 GB card.
  - The second GPU needs about 19 GiB, the same as before.
- **When the first GPU has only 20–30 GB free: `skip_unused_single_blocks` alone** (1.42×, no memory increase).
  - `disable_gradient_checkpointing` adds 12.7 GiB on cuda:0, so judge it by available memory.
  - `ip_adapter_on_transformer_device` alone gives almost no speed benefit, and it adds 5.4 GiB on cuda:0.
- **TF32: do not turn it on by default.** Its numbers differ clearly from the noise floor (inversion latent 2.9e-2), and its quality is unverified. It is the fastest option (3.9× per round, 1.9× on inversion), so it is worth a full-generation image comparison against the baseline as a follow-up.

## Caveats and items for the implementer (report only, nothing modified)
1. Measurement scope:
   - One candidate (960×720), 5 rounds, all in the TRANSPORT phase.
   - Not measured: INTENSIFY (lr 1000, rounds 50–70), the 70-round total, sampling and the full image.
   - The full-image time saving will be smaller than the round speed-up, because the exact patches do not shorten inversion or sampling.
2. The memory note in the advv-gpu-safety skill ("장당 피크 약 18.1 GiB", about 18.1 GiB peak per card) is the official-path value. It needs a line for exact_all: about 37.6 GiB reserved on cuda:0.
3. In this measurement a baseline round took 8.4–8.7 s and a no_grad forward 3.4 s, much faster than T011's 29 s/round and 16.4 s/forward. The cause is not confirmed; T011 may have used different GPUs, run alongside other load, or measured differently. The T016 estimate of "16.4 s/forward" needs re-examination.
4. Stage 2's exit 2 is a known consequence of `--no-control`, where the rtol is fixed at 1e-5. The latent noise floor (8.9e-5) is larger than rtol 1e-5, so this is not a script bug, but it means `--no-control` runs can routinely end in exit 2. Options for the implementer to consider (do not apply yet):
   - Recommend keeping control on in the docs.
   - Raise the latent default rtol to the 1e-4 range.
5. Stage 0 baseline and exact_all were launched with nohup without capturing the process exit code. Both profile.json files have status completed, and the documented contract writes exit 3 for an incomplete run, so I infer exit 0.
