# T031 Preregistration — Raw-sequence latent formation-displacement identifiability gate

## Question

Can raw, legally available horizontal/typewell sequences identify the single hidden formation displacement exposed by E003 strongly enough to justify a joint nonlinear TVT model?

This task is an auxiliary-state gate only. It may not emit or score a new competition prediction. A joint state-plus-spline experiment requires a separate preregistration after every state gate passes.

## Evidence basis

- E003's six hidden surface offset changes are available as training labels for all 773 wells, with at least five finite components per well.
- Their pairwise correlations are 0.98385–1.0 and PC1 explains 99.5626%, so their finite-value mean is a coherent one-dimensional target.
- Coordinate/smooth-surface reconstruction fails on this target, while the privileged target itself explains multi-RMSE action.
- Every train/test horizontal well has a contiguous TVT_input boundary, complete MD/X/Y/Z through the suffix, and partial GR. Test typewells expose TVT/GR only.
- Existing 142 summaries weakly identify risk/action and therefore form a negative-strength comparator rather than the final representation.

## Frozen boundary

The exact target, legal/prohibited inputs, sequence bins, branch list, hyperparameters, fold logic, target-free stress groups, controls, compute limits, and go/stop gates are in `config.json`. No branch, architecture width, seed, group, threshold, or target definition may be added after the first state metric.

## Decision logic

- **GO:** Pearson >=0.55, Spearman >=0.50, MAE <=12 ft, transfer floors pass, raw sequence materially beats both the 142-summary ridge and coordinate-only control, and destructive controls remove the signal.
- **RESEARCH ONLY:** intermediate signal; report and close without joint TVT implementation.
- **STOP:** Pearson <0.35 or MAE >16 ft; close compact raw-sequence state recovery and do not rescue it with architecture/seed search.

If GO passes, create T033. Do not implement T033 inside T031.
