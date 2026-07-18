# Copied Public Notebook Audit

Notebook: `notebooks/public_notebook/rogii-kim-om020.ipynb`  
Kaggle ref: `ashok205/rogii-kim-om020`  
Kaggle visibility at snapshot: private  
Last recorded public score: **7.119**

## What it contains

The 58-cell notebook combines a ridge/artifact model, particle filters and beam candidates, a learned trajectory branch, polynomial projection in `U = TVT + Z`, guarded same-well contact reconstruction, visible-prefix candidate selection, and an optional model-package correction.

## Why it is not an acceptable gold baseline

1. It mounts eight dataset slots plus competition data. Seven are named; one is blank/opaque and has no portable identifier.
2. Three named datasets have an unknown Kaggle license at the snapshot date.
3. Several learned artifacts were trained elsewhere, so their training data, folds, target construction, and leakage controls are not established locally.
4. The visible three test wells are training-derived examples. The notebook's contact override reconstructs those three at about 0.008–0.010 ft prefix RMSE, which validates identity/overlap logic only; it does not validate hidden-test generalization.
5. Embedded component OOF scores around 10.4 do not explain the final 7.119 leaderboard score cleanly enough to isolate causal contribution.
6. The final profile forcibly enables a model-package correction after selecting the preset, making the displayed preset and actual execution behavior diverge.

## Permitted use

- Use physical ideas, equations, candidate definitions, runtime engineering, and failure diagnostics as hypotheses.
- Do not use the opaque mount or any artifact without provenance and license approval.
- Reimplement accepted ideas from first principles under the project validation protocol.
- No component enters a final solution unless its training path, legal inputs, CV behavior, artifact hash, and notebook-runtime parity are recorded.

## Dependency conclusion

The notebook is an **idea catalogue and leaderboard reference**, not a reproducible experiment baseline. The first independent baseline must run from competition data plus locally generated artifacts whose lineage is fully recorded.
