# Rosetta Stone

Confidence = Laplace-smoothed prediction pass rate; entries with failed predictions keep them in their history.

## Proposed circuit
- components: L2.MLP, L2.H1, L2.H0, L2.H2, L1.H4, L1.H3
- baseline for outside-ablation: mean
- behavior retention: 0.77

## L2.MLP  (confidence 0.75, 2 passed / 0 failed)
**Candidate function:** writes directly in the answer-token direction at the final position
**Contexts:** (contexts not yet narrowed - see next_experiment)
**Source:** template
**Contradictions / caveats:**
- ablation baselines disagree: zero_ablation=-1.830, mean_ablation=-2.143, resample_ablation=-5.015
**Predictions:**
- mean_ablation on L2.MLP @ induction-heldout: logit_diff to decrease (threshold 0.86) -> PASSED, measured -2.057
- activation_patching on L2.MLP @ induction-heldout: logit_diff to increase (threshold 0.10) -> PASSED, measured +12.975
**Next experiment:** run path patching against the strongest downstream head to test whether anything reads this component's output

## L2.H1  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** induction-style copying (attends to the token after a previous occurrence of the current token); writes directly in the answer-token direction at the final position
**Contexts:** repeated-sequence / copying contexts
**Source:** template
**Predictions:**
- mean_ablation on L2.H1 @ induction-heldout: logit_diff to decrease (threshold 0.09) -> PASSED, measured -0.185
**Next experiment:** test on a prompt family with the repeat at a different offset to separate positional from content-based attention

## L2.H0  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** induction-style copying (attends to the token after a previous occurrence of the current token); writes directly in the answer-token direction at the final position
**Contexts:** repeated-sequence / copying contexts
**Source:** template
**Predictions:**
- mean_ablation on L2.H0 @ induction-heldout: logit_diff to decrease (threshold 0.11) -> PASSED, measured -0.191
**Next experiment:** test on a prompt family with the repeat at a different offset to separate positional from content-based attention

## L2.H2  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** induction-style copying (attends to the token after a previous occurrence of the current token); writes directly in the answer-token direction at the final position
**Contexts:** repeated-sequence / copying contexts
**Source:** template
**Predictions:**
- mean_ablation on L2.H2 @ induction-heldout: logit_diff to decrease (threshold 0.09) -> PASSED, measured -0.120
**Next experiment:** test on a prompt family with the repeat at a different offset to separate positional from content-based attention

## L1.H4  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** induction-style copying (attends to the token after a previous occurrence of the current token)
**Contexts:** repeated-sequence / copying contexts
**Source:** template
**Contradictions / caveats:**
- attention/readout pattern suggests task relevance, but measured ablation effects are small - pattern may be present-but-unused
**Predictions:**
- mean_ablation on L1.H4 @ induction-heldout: logit_diff to no_change (threshold 0.00) -> PASSED, measured -0.019
**Next experiment:** test on a prompt family with the repeat at a different offset to separate positional from content-based attention

## L1.H3  (confidence 0.67, 1 passed / 0 failed)
**Candidate function:** induction-style copying (attends to the token after a previous occurrence of the current token)
**Contexts:** repeated-sequence / copying contexts
**Source:** template
**Contradictions / caveats:**
- attention/readout pattern suggests task relevance, but measured ablation effects are small - pattern may be present-but-unused
**Predictions:**
- mean_ablation on L1.H3 @ induction-heldout: logit_diff to no_change (threshold 0.00) -> PASSED, measured +0.009
**Next experiment:** test on a prompt family with the repeat at a different offset to separate positional from content-based attention
