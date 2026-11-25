# Matched training volume and configuration

This is the controlled-comparison contract for T1–T5. It supersedes the short
T1/T5 GRPO drafts for new comparison runs. It is a planned recipe, not a
complete runtime config or evidence that training has completed. No result from
the old unequal configuration may be relabeled as a controlled comparison.

## Source and evidence status

- The authoritative T1/T5 recipe is
  [`configs/experiment/midterm_grpo_aligned_20261125.json`](../../configs/experiment/midterm_grpo_aligned_20261125.json).
  Its `config_status` is `PLANNED_BLOCKED_NOT_A_COMPLETE_RUNTIME_CONFIG` and its
  `execution_status` is `NOT_LAUNCHED`.
- T2/T3 retain the PR #18 handoff definitions in
  `rlsd_rlrt_reproduction_20261124/configs/experiment_recipe.json` and method
  section 8.4. That package is a recipe/reference snapshot, not the complete
  original runtime.
- The old Dual-Track-OPD `midterm_t1_base_grpo.yaml` and
  `midterm_t5_base_sft_grpo.yaml` drafts are marked
  `superseded_for_controlled_comparison`. Their historical values must not be
  used for a new controlled run or silently presented as matched training.
- T4 has no new numerical ISO/MOPD training recipe in PR #18. Do not infer its
  training volume from the final checkpoint name.

## Common RL volume

`Prompt consumptions` count prompt draws, not necessarily unique prompts.
`Trajectories` are prompt consumptions multiplied by rollouts per prompt. Each
global step draws 256 prompts × 8 rollouts = 2,048 sequences. Every arm uses one
256-prompt actor mini-batch and one PPO epoch, so there is at most one optimizer
update per global step. A step with no nonzero policy-gradient signal skips the
actor update; actual updates require log reconciliation after filtering, resume,
early stop, or skipped updates.

| Task / RL stage | Source records before filtering | Global rollout steps | Prompt batch | Mini prompt batch | Rollouts/prompt | Prompt consumptions | Trajectories | Planned optimizer updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 / GRPO | 30,000 | 400 | 256 | 256 | 8 | 102,400 | 819,200 | at most 400 |
| T2 / Base hybrid | 30,000 | 400 | 256 | 256 | 8 | 102,400 | 819,200 | at most 400 |
| T3 / post-MOPD hybrid | 15,000 | 400 | 256 | 256 | 8 | 102,400 | 819,200 | at most 400 |
| T5 / post-SFT GRPO | 15,000 | 400 | 256 | 256 | 8 | 102,400 | 819,200 | at most 400 |

Before filtering, mean record exposures are `102400/30000 ≈ 3.413` for T1/T2
and `102400/15000 ≈ 6.827` for T3/T5. Equal prompt/trajectory volume does not
imply equal unique-question coverage. `max_steps=400` takes precedence over
`total_epochs=3` in the reference trainer; these are not 1,200-step runs.

### Pairwise data and initialization

| Comparison | Required shared inputs | Intentional difference | Valid interpretation |
| --- | --- | --- | --- |
| T1 versus T2 | Same immutable Base checkpoint; MOPD-route-v2 15k + RL-RSTG-ZVP 15k, in that order, 30k before filtering | GRPO versus hybrid | Controlled algorithm comparison only after checkpoint, resolved-config, reward, and runtime equivalence are verified |
| T5 versus T3 | Same pinned RL-RSTG-ZVP 15k | SFT versus MOPD parent and GRPO versus hybrid | Training-route comparison, not an initialization-only or algorithm-only ablation |
| T1 versus T5 | Common public RL configuration only | Base versus SFT parent and 30k versus 15k RL data | Not an SFT-only ablation |
| T2 versus T3 | Common public RL configuration only | Base versus MOPD parent and 30k versus 15k RL data | Original route comparison, not an initialization-only ablation |

The pinned dataset revisions are:

- `lzy-vlm-lab-opd/MOPD-route-v2-train-15k` at
  `6374961a4945f5ee7462520ecd54bcc93ee9d0bc`;
- `lzy-vlm-lab-opd/RL-RSTG-ZVP-train-15k` at
  `df005b52aead28acf57b05a1f30487a220d398bb`.

Concatenate all selected rows without deduplication or downsampling; use the
original shuffled dataloader with seed 1. Actual filtering and sampling still
need resolved manifests and logs.

### T5 SFT is a separate training stage

Do not fold SFT volume into the subsequent GRPO row or hide it when reporting
end-to-end compute. The unchanged T5 SFT stage has 148,054 train rows, global
batch 64, and 3 epochs. With its floor/drop-last step formula, this is 2,313
steps per epoch, 6,939 global/optimizer steps, and 444,096 consumed examples. It
drops 22 rows per epoch (66 row slots across three epochs). Filtering, resume,
and actual logs can change the realized accounting and must be reported.

### T4 and the T3 parent checkpoint

The candidate shared MOPD checkpoint is
`lzy-vlm-lab-opd/qwen3-vl-8b-iso-mopd-route-v2-step351` at immutable revision
`f9582f3594ba0a47e513a50b658204e7c322fbbb`. If T3 reuses it, T4's final MOPD
artifact and T3's RL parent must be this exact checkpoint or an independently
verified byte/content-equivalent export. Record ISO and MOPD data, steps, and
optimizer volumes separately; PR #18 does not supply that upstream provenance.
T4 versus T3 can describe the incremental effect of the hybrid stage only when
that exact parent identity and the downstream evaluation protocol are verified.
Consequently, matched T3/T5 RL budgets alone do not establish equal end-to-end
SFT-versus-MOPD compute.

## Common RL configuration

Use the original PR #18 hybrid runtime for all four arms. T1/T5 select its
`opsd.grpo_opsd_mode="grpo"`; T2/T3 retain `"hybrid"`. This is the only common
field intentionally changed by the aligned T1/T5 JSON. In GRPO mode the
reference stack sets effective teacher lambda to zero, skips teacher prompts
and scoring, and uses unit detached credit factors; it does not switch to the
old baseline backend or loss implementation.

The JSON is the exhaustive machine-readable contract. Tests compare every
common field against the T2/T3 source recipe. High-risk settings are summarized
here:

| Field | Common T1/T2/T3/T5 value |
| --- | --- |
| Reward | Original `rstg_oss_verifier_v3`; source and dependencies required |
| LR / schedule / optimizer | `1e-6`, constant, AdamW betas `[0.9, 0.999]`, weight decay `0.01` |
| PPO clip / dual clip / epochs | `0.2 / 0.28`, dual clip `3`, one epoch |
| Gradient clipping | Max norm `1` |
| Advantage | Per-UID GRPO groups of 8; sample std with denominator `G-1`, epsilon `1e-6` |
| Group/loss masks | Binary reward groups enabled; unjudgeable groups masked from advantages, returns, and valid-token loss denominator |
| Loss reduction | Original distributed valid-response-token reduction with dynamic micro-batches |
| KL | Disabled |
| Prompt / response tokens | 4,096 / 4,096 |
| Data / rollout seed | 1 / 1 |
| Rollout sampling | Temperature 1, top-p 1, top-k -1 |
| Image pixels | 262,144–1,048,576 |
| Actor / reference token budget per GPU | 8,192 / 8,192 |
| Rollout max batched tokens | 32,768 |
| Save / validation frequency | 50 / 50 |
| Hardware | One node, 8 GPUs |

All four arms retain `enable_grpo_opsd_simple=true` and
`binary_reward_groups=true`. GRPO advantage uses one scalar outcome per
response, normalized within its eight-member UID group and broadcast over the
response tokens. The actor's current/old PPO ratio and student-log-probability
gradient remain unchanged in pure GRPO mode. Teacher-specific fields copied in
the common JSON are inactive in that mode, not permission to introduce teacher
routing into T1/T5.

## Required pre-run resolutions

The aligned numerical contract is defined, but execution remains blocked until
all items in the JSON `run_blockers` are resolved and linked to immutable
artifacts:

1. **Original runtime:** supply the complete original training repository,
   backend/environment revisions, and fully resolved T1/T5 configurations.
   Reference-code tests are not a substitute for server integration.
2. **Original reward:** obtain `rstg_oss_verifier_v3` and its dependencies. The
   old baseline verifier does not support the MOPD `exact`, `math`,
   `transcription`, and `auto` reward types. Do not substitute it or relax the
   MOPD stage/reward checks to make the old pipeline run.
3. **Prompt/data conversion:** retain the original `unified.jinja` `<thought>`
   protocol, SHA-256
   `42393b26003d0db3bcb69ed0ffe9ddf3d1a4f7a1a2df1727b011199d9f1e18d3`.
   Published `<think>` prompts are provenance, not directly reusable rendered
   training prompts.
4. **Validation:** obtain the original independent validation asset; do not use
   the old overlapping training-data monitor as a substitute.
5. **Base identity:** the original T2 recipe still says
   `RECORD_ACTUAL_DOWNLOADED_REVISION`. Prove that T2 used T1's pinned Base
   revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, or rerun T2 with that
   exact revision before calling T1/T2 controlled.
6. **T5 parent identity:** record the final SFT checkpoint's immutable revision,
   complete SFT config, data identity, and log-reconciled training volume.
7. **Semantic smoke:** verify actual config propagation, eight siblings per UID,
   masks and reductions, teacher-free GRPO behavior, reward coverage, checkpoint
   identity, and zero-advantage optimizer accounting before full training.

Changing any common setting requires a new pre-registration, run name, smoke,
and comparability review. Reports must distinguish planned from actual global
rollout steps and optimizer updates. Until these checks and the corresponding
runs/evaluations are complete, no method-advantage claim is verified.
