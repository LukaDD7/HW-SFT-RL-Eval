# Training-volume and configuration alignment

This is the comparison-cohort contract for T1–T5. It aligns accounting and
comparison boundaries; it is not a runnable training config or a claim that
training has completed. Pin the actual resolved configs and reconcile every
planned quantity with logs before reporting results.

## Source and evidence status

- T1/T5 values come from the current Dual-Track-OPD midterm drafts
  `configs/experiment/midterm_t1_base_grpo.yaml`,
  `configs/experiment/midterm_t5_base_sft_grpo.yaml`, and
  `configs/data/midterm_rl_rstg_zvp_v1.yaml`, reviewed on 2026-11-25. The drafts
  must receive immutable code/config identities in their task reports.
- T2/T3 values come from the PR #18 handoff, especially
  `rlsd_rlrt_reproduction_20261124/configs/experiment_recipe.json` and method
  section 8.4. Its recipe is not a complete runtime config.
- T4 has no new numerical ISO/MOPD training recipe in PR #18. Do not infer its
  training volume from the final checkpoint name.

## RL comparison cohorts and volume

`Prompt consumptions` count prompt draws, not necessarily unique prompts.
`Trajectories` are prompt consumptions multiplied by rollouts per prompt.
Planned optimizer updates use the configured mini-batch and PPO-epoch semantics;
actual updates still require log reconciliation after filtering, resume, early
stop, or skipped updates.

| Task / RL stage | Source records before filtering | Global rollout steps | Prompt batch | Mini prompt batch | Rollouts/prompt | Prompt consumptions | Trajectories | Planned optimizer updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 / GRPO | 15,000 | 50 | 16 | 8 | 8 | 800 | 6,400 | 100 |
| T5 / post-SFT GRPO | 15,000 | 50 | 16 | 8 | 8 | 800 | 6,400 | 100 |
| T2 / Base hybrid | 30,000 | 400 | 256 | 256 | 8 | 102,400 | 819,200 | at most 400 |
| T3 / post-MOPD hybrid | 15,000 | 400 | 256 | 256 | 8 | 102,400 | 819,200 | at most 400 |

For T1/T5, each rollout step has `16 / 8 = 2` mini-batches and PPO epochs is 1,
so the planned 50 rollout steps produce 100 optimizer updates. For T2/T3, the
256-prompt mini-batch equals the full rollout batch; PPO epochs is 1. A hybrid
global step with no nonzero advantage advances the global-step/teacher schedule
but skips the actor optimizer update, hence “at most 400,” not an asserted 400.

Before filtering, mean record exposures are `800/15000 ≈ 0.053` for T1/T5,
`102400/30000 ≈ 3.413` for T2, and `102400/15000 ≈ 6.827` for T3. Equal prompt
or trajectory volume does not imply equal epochs or unique-question coverage.
PR #18 explicitly sets `max_steps=400`; its reference trainer gives that field
precedence over `total_epochs=3`.

### T5 SFT is a separate training stage

Do not fold SFT volume into the subsequent GRPO row. The current T5 draft has
148,054 SFT train rows, global batch 64, and 3 epochs. With its floor/drop-last
step formula, this is 2,313 steps per epoch, 6,939 global/optimizer steps, and
444,096 consumed examples. It drops 22 rows per epoch (66 row slots across three
epochs). Filtering, resume, and actual logs can change the realized accounting
and must be reported.

### T4 and the T3 parent checkpoint

The candidate shared MOPD checkpoint is
`lzy-vlm-lab-opd/qwen3-vl-8b-iso-mopd-route-v2-step351` at immutable revision
`f9582f3594ba0a47e513a50b658204e7c322fbbb`. If T3 reuses it, T4's final MOPD
artifact and T3's RL parent must be this exact checkpoint or an independently
verified byte/content-equivalent export. Record ISO and MOPD data, steps, and
optimizer volumes separately; PR #18 does not supply that upstream provenance.

## Configuration alignment and comparability

The following settings are aligned **within** each cohort, not imposed on the
other cohort. Changing them requires a new pre-registration, run name, smoke,
and comparability review rather than silently reusing an old report.

| Field | T1/T5 GRPO cohort | T2/T3 hybrid cohort |
| --- | --- | --- |
| RL data | Both use pinned RL-RSTG-ZVP 15k | T2 uses both pinned sets (30k); T3 uses only RL-RSTG-ZVP 15k |
| Reward | `midterm-grpo-baseline-verifier-v1` | Original `rstg_oss_verifier_v3`; source/adapter still required |
| Learning rate | `5e-6` | `1e-6`, constant |
| PPO clip / epochs | `0.2 / 0.28` / 1 | `0.2 / 0.28` / 1; dual clip 3 |
| Advantage normalization | No std normalization | GRPO sample std, denominator `G-1`, eps `1e-6` |
| Loss accounting | `token-mean`; verify backend semantics | Global valid-response-token mean, including judgeable homogeneous groups in the denominator |
| KL | Disabled | Disabled |
| Prompt / response tokens | 2,048 / 12,288 | 4,096 / 4,096 |
| Training seed | 42 | 1 |
| Actor token budget/GPU | 24,576 | 8,192 |
| Save / validation frequency | 10 / 10 | 50 / 50 |

These are deliberate comparison boundaries:

1. **T1 versus T5** can isolate the effect of the SFT parent only after their
   GRPO data revision, reward, resolved config, and volume are confirmed equal.
   The SFT stage itself remains an intentional difference.
2. **T2 versus T3** is a route comparison, not an initialization-only ablation:
   initialization and 30k-versus-15k data both change. Do not attribute its
   outcome to initialization alone.
3. **GRPO-cohort versus hybrid-cohort** scores are descriptive, not controlled
   algorithm comparisons. Step count, batches, optimizer-update semantics,
   learning rate, reward, normalization, length budgets, seeds, and engineering
   settings differ. Do not label a cross-cohort delta as an algorithm-only gain.
4. **T4 versus T3** can describe the incremental effect of the hybrid stage only
   when the exact T4 final/T3 parent checkpoint identity and downstream protocol
   are verified.
5. If the team wants a new all-arm controlled comparison, it must pre-register a
   separate common recipe. Do not overwrite these approved cohort settings or
   relabel existing runs to manufacture comparability.

Every report must name its cohort, record planned and actual global rollout steps
and optimizer updates separately, and carry forward any comparability limitation
into the consolidated midterm conclusions.
