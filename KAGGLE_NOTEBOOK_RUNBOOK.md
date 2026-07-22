# ROGII Kaggle Notebook Runbook

The purpose of a Kaggle notebook is to run an experiment or inference job successfully and save the outputs needed for evaluation or submission.

## Default workflow

1. Prepare one runnable notebook and the data/model files it needs.
2. Attach the required dataset and competition data.
3. Run all cells.
4. Confirm the notebook completed without an exception.
5. Confirm the primary output files exist and can be opened.
6. Record the notebook reference/version and retrieve the outputs.

Do not add operational checks unless they prevent a realistic failure that would make the result wrong or unusable.

## Required checks

A notebook should check only what is necessary for correctness:

- required input files and directories exist;
- the competition test data and sample submission can be located;
- prediction rows match the sample-submission IDs and order;
- predictions are finite;
- required output files are written successfully;
- exceptions are visible and, when practical, summarized in a small failure receipt.

These checks protect the result itself. They should remain simple and actionable.

## Checks that are not required by default

Do not block a run merely because of:

- internet state;
- CPU, GPU, accelerator, or native thread-pool details;
- Python/platform/version differences that do not cause an actual incompatibility;
- repeated SHA-256, byte-count, archive-member, or source-identity verification;
- exact local-versus-Kaggle byte equality;
- verbose manifests, environment inventories, or redundant receipts.

Use one of these only when a specific experiment has demonstrated that the condition materially affects correctness.

## Packaging

Prefer the simplest reliable packaging method:

- a normal Kaggle dataset containing the required code/model files;
- a compact archive when many files must stay together;
- one canonical notebook per experiment or inference path.

The notebook may load packaged project code, but it does not need to prove the identity of every file before running. Rebuild or update the package when the implementation changes.

## User and assistant responsibilities

The assistant prepares the notebook, required dataset files, and concise run instructions. The assistant does not run a Kaggle notebook unless the user explicitly authorizes it.

The user imports or opens the notebook, attaches the required inputs, runs it, and saves a notebook version. Afterward, the user provides the notebook owner, slug, and version when output retrieval is needed.

## Output retrieval

After a run:

1. verify the notebook completed;
2. list the notebook output files;
3. retrieve the primary result files;
4. parse them and check rows, IDs, finite values, and the experiment-specific metric or result;
5. update the experiment record.

Missing or malformed output is a reason to fix and rerun the notebook. A hash mismatch, device difference, or environment difference is not a failure unless it caused an incorrect result.

## Submissions

Creating an output named `submission.csv` is allowed as part of inference. Sending it to the competition still requires explicit user authorization.
