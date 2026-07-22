# E011 Private Kaggle Run Instructions

The package is prepared and uploaded. The assistant has not executed the canonical notebook and is not authorized to do so.

## Import

Import this local notebook into a **private** Kaggle notebook:

`notebooks/training_and_submission/e011_spline4_deployment_kaggle.ipynb`

Notebook SHA-256:

`63f299fcf069bbcdf6a8f17c7e66f0e88b1ff7f238a1b0e5b13934d353f5e9e9`

## Attach exactly two inputs

1. Private dataset `ashok205/rogii-e011-deployment-inputs`, **version 1**.
2. Competition data for `rogii-wellbore-geology-prediction`.

Do not attach replacement source archives, copied predictions, public notebooks, or other datasets.

## Runtime settings

- Visibility: private
- Accelerator: CPU
- Internet: disabled
- Run all cells from a clean session
- Do not edit notebook cells or constants
- Do not submit `submission.csv` to the competition

## Expected top-level outputs

- `e011-run-receipt.json`
- `e011-output-manifest.json`
- `rogii-e011-deployment-results-v1.zip`
- `submission.csv`
- `e011-well-predictions.json`

A Kaggle status of `COMPLETE` is not sufficient evidence by itself. After the run, provide the notebook owner, slug, and exact version number so the outputs can be listed, downloaded, hashed, parsed, and independently verified.

## Frozen identities

- Runtime source commit: `e6fbfaa02e8a72dc3c67713da51fe843417cc114`
- Model SHA-256: `dd92e8b12f275b6d453e55d6cce2e7c10a43c8a34957d6d47b1b5e5ff5b3c5a8`
- Bundle SHA-256: `15a82b9eee1b13d126c9e57b6402c1b307b89238718f0f292eca7773321f66a2`
- Input receipt SHA-256: `2ce82b05cbf52460af47d52202072c5999d42ed1fa59d3d9c9669f6a23a4d955`

If any preflight check fails, preserve the generated `e011-run-receipt.json` and do not bypass the failure.
