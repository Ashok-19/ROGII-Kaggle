# E011 Kaggle Run Instructions

The notebook and input dataset are prepared. The assistant will not run the notebook unless explicitly authorized.

## Run

1. Import or open:
   `notebooks/training_and_submission/e011_spline4_deployment_kaggle.ipynb`
2. Attach:
   - `ashok205/rogii-e011-deployment-inputs`, version 1
   - the `rogii-wellbore-geology-prediction` competition data
3. Run all cells.
4. Save a notebook version after the run finishes.

No special internet, CPU, accelerator, thread, or environment setting is required.

## Expected outputs

- `submission.csv`
- `e011-run-receipt.json`
- `e011-well-predictions.json`

The run is successful when all cells finish and these files are present under the notebook outputs. Do not send `submission.csv` to the competition unless submission is separately authorized.

After completion, provide the notebook owner, slug, and version number so the outputs can be retrieved and checked for row coverage, IDs/order, finite predictions, and basic readability.
