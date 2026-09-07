# scLong learning-rate source contract

Pinned source: BaiDing1234/scLong revision
`41b72021540918e4386c4f2264351d7a6cbeed99`.

- `README.md` pretraining command passes `--learning_rate 0.00005`.
  Its script name is `pretrain_gocont_4096_all_1b_mix.py`, whereas the pinned
  tree contains `pretrain_dual_4096_all_1b_mix.py`; this upstream naming mismatch
  is recorded, not silently represented as an executable verified command.
- The actual pretraining Python file's parser defaults to `1e-4` if no argument
  is supplied. The scheduler uses the supplied learning rate as its peak.
- Constructor: first cycle 15, cycle multiplier 2, warmup 5, floor `1e-6`,
  peak decay 0.9. `scheduler.step()` has no epoch argument and is called once
  after each training epoch, not after each minibatch or optimizer update.
- We use the advertised launch peak `5e-5` and the actual sequential code
  semantics. The explicit `step(epoch=...)` branch in upstream differs from
  its sequential recurrence; our native schedule follows the branch used.

For zero-based completed epochs `e`, locate cycle `c`, length `T`, and position
`t` by subtracting lengths starting at 15, with `T_next = 2*(T-5)+5`.
Peak `p = 5e-5 * 0.9**c`; learning rate is
`1e-6 + (p-1e-6)*t/5` for `t<5`, otherwise
`1e-6 + (p-1e-6)*(1+cos(pi*(t-5)/(T-5)))/2`.
Cycle starts are e=0,15,40,85. Epochs 1 and 2 use `1e-6` and `1.08e-5`.
The last discrete epoch before restart approaches but does not reach the floor;
restart returns to the floor. There is no within-epoch LR interpolation.

Implementation: `cell/schedule.py`, shared by native pretraining and LoRA.
Applying this pretraining schedule to our LoRA phase is OUR policy, not a claim
about scLong's downstream optimizer. Optimizer type, betas, loss warmup and
Teacher EMA are unchanged. `configs/cell/lr_sclong.json` locks the current
policy; current data selection references it. Superseded Census recipes are
historical and must not launch as the current default.

Schedules are reconstructed from the saved epoch/step, so mid-epoch resume
does not advance LR. The old optimizer-step schedule remains explicitly named
`legacy_step_cosine` for reproduction. New resolved configs record scheduler
identity; never silently resume old checkpoints under a changed schedule.

The active task remains training readiness plus smoke tests only, not formal
multi-epoch execution. A one-step smoke at epoch zero cannot exercise warmup
completion/restarts or nonzero distillation loss weighting; separate scheduler
and five-loss backward tests cover these mechanisms.

## Verified numerical comparison (2026-09-07)

On the server, with torch `2.13.0+cu130`, the official scheduler class was
isolated from `utils.py` using AST for a test-only comparison (no other upstream
code or model imported). Source SHA256:
`267fa396fa421ffe3111a3f516bf00987c012cc3a380ee48d7465159fafc3218`.
Using the constructor above and sequential `step()`, all 300 epoch learning
rates matched the native implementation exactly: maximum absolute error 0.
This is scheduler parity, not a real-data training result.
