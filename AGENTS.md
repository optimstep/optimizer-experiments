# Experiment operating guide

## Purpose and reproducibility

- Treat `configs/*.yaml` as the source of truth. Do not hide experimental
  hyperparameters in shell scripts or interactive commands.
- Change one intended variable per ablation. Keep the seed, model, tokenizer,
  global batch, schedule, evaluation cadence, and update-scale convention fixed.
- Always run a two-step local smoke test before renting hardware.
- Record the git commit, complete resolved configuration, GPU model, CUDA/PyTorch
  versions, wall time, and Vast.ai hourly rate with every result.
- Never commit datasets, checkpoints, TensorBoard caches, credentials, SSH keys,
  instance metadata, or local hostnames.

## Local workflow

```bash
make bootstrap
make prepare
make smoke CONFIG=configs/muon.yaml
make test
```

Use `python3`. Results belong under `results/<run-name>/`; artifacts belong under
`artifacts/`. Both are ignored by git. Save loss every step, evaluations and
checkpoints every 25 steps, and expensive optimizer diagnostics every 5 steps.

## Renting on Vast.ai

1. Prefer one A100 40 GB for the 16,384 batch benchmark. Use two GPUs only when
   two independent arms can saturate them; do not introduce DDP merely to fill
   the second GPU.
2. Select a verified host with adequate disk, reliability, and direct SSH.
   A cheap instance that cannot resolve DNS is not usable. Test DNS, HTTPS,
   available disk, and `nvidia-smi` immediately after startup.
3. Use a maintained PyTorch CUDA image. Do not use verbose HTTP/curl diagnostics
   because they may expose authentication material.
4. Export instance-specific values locally; never write them into tracked files:

   ```bash
   export REMOTE=root@HOST
   export VAST_INSTANCE_ID=12345678
   export VAST_DOLLARS_PER_HOUR=0.82
   export MAX_DOLLARS=8
   export VAST_STARTED_EPOCH=$(date +%s)
   ```

5. Start `make hard-stop` from the local machine before launching training. The
   local timer is independent of the rented box and destroys the instance at the
   budget deadline. A timer on the remote machine alone is insufficient.
6. Run `make remote-setup`, then a smoke test remotely. Only then launch the full
   configuration. Keep a single scheduler process; verify parent PIDs and GPU
   process counts before trusting `pgrep`, because DataLoader workers repeat the
   training command line.

## Monitoring and shutdown

- Sync logs and TensorBoard events with `make sync` every 5–15 seconds. Render or
  refresh plots about once per minute; plotting every sync wastes CPU and I/O.
- Check GPU utilization, log modification time, last completed step, finite loss,
  checkpoint creation, and the scheduler process at least every five minutes.
- A stale dashboard is not evidence of a stalled run. Inspect the remote log and
  event-file timestamps independently before restarting anything.
- Do not terminate a merely slow optimizer. Stop only clear divergence, repeated
  non-finite values, a reproducible exception, or a budget deadline.
- Download logs, resolved configs, diagnostics, plots, and checkpoints before
  destroying the instance. Mark a suite complete only after verifying local files.
- Unless explicitly asked otherwise, destroy—not stop—the instance after the
  queue finishes. A stopped Vast.ai instance may continue charging for storage.

## Known gotchas

- Progress bars use carriage returns. Convert `\r` to `\n` before grepping logs,
  or output can look empty or overwhelm diagnostics.
- Never let two runs write the same log, checkpoint, or TensorBoard directory.
  A scheduler race previously produced plausible but invalid mixed artifacts.
- Compare optimizers at a common effective update scale. These configs RMS-match
  Muon matrix updates and use the same learning rate; retain raw update RMS,
  applied update RMS, weight RMS, update/weight ratio, and alignment diagnostics.
- Muon inverse-half failures can be mathematical or numerical. Keep exact/eigh or
  SVD diagnostics on small Gram matrices when changing the iteration.
- Shared-interface Shampoo statistics must normalize each neighboring Gram by its
  mean trace before mixing; otherwise fan-in/out or gradient scale dominates.
- Joint Muon `p=0.5` is sensitive to normalization and NS convergence. Monitor
  spectral/residual statistics for at least the first 100 steps.
- Cautious sphere attraction currently uses a clear reference implementation that
  materializes FP32 snapshots and masks. Preserve it as the correctness oracle.
  Optimize with reduction plus fused pointwise Triton kernels only after adding
  numerical equivalence tests; do not silently change ongoing ablations.
- The pruned tokenizer reduces embedding overhead but must be identical across all
  compared arms. Reusing the Arrow pretokenization cache is important for stable
  throughput.

## Adding an experiment

Create a YAML file extending `base.yaml`, use a unique result directory, add a
focused test if optimizer math changes, run local and remote smoke tests, then add
the config to `scripts/run_suite.sh`. Document whether the run is a correctness
reference, performance implementation, or exploratory ablation.

For next-token work, keep only baselines and promoted treatments in
`configs/next_token/`. Express widths, short smoke durations, and parameter sweeps
as `--set` overrides or suite environment variables. Do not commit a YAML for
every point in a sweep.
