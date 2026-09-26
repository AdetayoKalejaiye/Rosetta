# Rosetta Stone

Confidence = Laplace-smoothed prediction pass rate; entries with failed predictions keep them in their history.

## Proposed circuit
- components: L31.MLP, L30.H6, L27.MLP, L30.MLP, L28.MLP, L27.H7, L28.H8, L24.H11
- baseline for outside-ablation: mean
- behavior retention: -0.01

## L31.MLP  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** writes directly in the answer-token direction at the final position
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Contradictions / caveats:**
- ablation baselines disagree: zero_ablation=-1.060, mean_ablation=-3.014, resample_ablation=-3.824
**Predictions:**
- mean_ablation on L31.MLP @ induction-heldout: logit_diff to decrease (threshold 1.21) -> PASSED, measured -2.958
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L30.H6  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** writes directly in the answer-token direction at the final position
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Contradictions / caveats:**
- ablation baselines disagree: zero_ablation=-0.956, mean_ablation=-1.359, resample_ablation=-1.621
**Predictions:**
- mean_ablation on L30.H6 @ induction-heldout: logit_diff to decrease (threshold 0.54) -> PASSED, measured -1.511
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L27.MLP  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** writes directly in the answer-token direction at the final position
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Predictions:**
- mean_ablation on L27.MLP @ induction-heldout: logit_diff to decrease (threshold 0.29) -> PASSED, measured -0.935
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L30.MLP  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** writes directly in the answer-token direction at the final position
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Contradictions / caveats:**
- ablation baselines disagree: zero_ablation=-1.651, mean_ablation=-0.847, resample_ablation=-0.959
**Predictions:**
- mean_ablation on L30.MLP @ induction-heldout: logit_diff to decrease (threshold 0.34) -> PASSED, measured -1.293
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L28.MLP  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** writes directly in the answer-token direction at the final position
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Contradictions / caveats:**
- ablation baselines disagree: zero_ablation=-0.060, mean_ablation=-0.635, resample_ablation=-0.838
**Predictions:**
- mean_ablation on L28.MLP @ induction-heldout: logit_diff to decrease (threshold 0.25) -> PASSED, measured -1.006
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L24.H11  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** induction-style copying (attends to the token after a previous occurrence of the current token); attention sink on position 0 (often a no-op/default)
**Contexts:** repeated-sequence / copying contexts
**Source:** template
**Predictions:**
- mean_ablation on L24.H11 @ induction-heldout: logit_diff to decrease (threshold 0.08) -> PASSED, measured -0.289
**Next experiment:** test on a prompt family with the repeat at a different offset to separate positional from content-based attention

## L27.H7  (confidence 0.33, 0 passed / 1 failed)
**Candidate function:** attention sink on position 0 (often a no-op/default)
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Predictions:**
- mean_ablation on L27.H7 @ induction-heldout: logit_diff to no_change (threshold 0.00) -> FAILED, measured -0.298
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L28.H8  (confidence 0.33, 0 passed / 1 failed)
**Candidate function:** attention sink on position 0 (often a no-op/default)
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Predictions:**
- mean_ablation on L28.H8 @ induction-heldout: logit_diff to no_change (threshold 0.00) -> FAILED, measured +0.678
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output
