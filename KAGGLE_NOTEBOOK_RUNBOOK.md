# ROGII Private Kaggle Notebook Runbook

This runbook adapts the verified private-notebook workflow from the PTCG repository to ROGII. Repository code, frozen configuration, input hashes, and downloaded raw outputs remain the source of truth. A notebook status of `COMPLETE` is not evidence until every expected output is listed, downloaded, hashed, parsed, and checked.

## Compute placement

- **Local machine:** source edits, metadata inspection, deterministic bundle construction, unit/contract tests, tiny smoke cases, packaging, and output verification.
- **Local resource ceiling:** do not start a workflow that needs more than two CPU threads. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `BLIS_NUM_THREADS` to at most `2`. Tree models must use `n_jobs <= 2`.
- **Private Kaggle notebook:** any workflow expected to take more than roughly 15 minutes, consume substantial memory/I/O, or benefit from remote compute. E010 full validation belongs here.
- **Kaggle submission:** never create one without explicit user authorization.

## Required state before a Kaggle run

No heavy notebook is ready until all of the following are fixed and recorded:

1. an exact committed Git SHA and explicit dirty-state statement;
2. one resolved configuration and SHA-256;
3. exact source, fold, parent-artifact, and data identity hashes;
4. a unique run ID and declared question;
5. accelerator, internet, CPU-thread, wall-time, memory, and output caps;
6. fail-closed stop conditions;
7. one canonical private input dataset reference and integer version;
8. one canonical local notebook path;
9. expected output filenames and schemas;
10. a retrieval and independent-verification procedure.

Notebook-only algorithm logic is forbidden. The notebook must import the exact hash-sealed `src/` package from the prepared input bundle. Rebuild and republish the bundle whenever any included source, config, fold, or parent artifact changes.

## Canonical dataset and notebook policy

- Reuse one private dataset record per stable input role; publish new integer versions rather than creating a new dataset slug for every retry.
- Store sealed archive bytes with a neutral extension such as `.zip.bin` so Kaggle does not auto-extract or rewrite them.
- The archive must contain a machine-readable manifest with every relative path, byte count, and SHA-256.
- The notebook must reject missing, duplicate, extra, size-mismatched, or hash-mismatched inputs before importing project code.
- Automatic dataset/model attachment through notebook APIs is not trusted. The user manually attaches the exact prepared dataset version and competition data in the Kaggle UI.
- Keep one canonical notebook file and update it in place after source changes.

## User and assistant responsibilities

The assistant prepares and verifies:

- the canonical local notebook;
- the minimal private input dataset staging directory and dataset version;
- exact dataset/competition attachments and runtime settings;
- fail-closed preflight, execution, output archive, and manifests;
- post-run status inspection, output listing, download, hashing, schema validation, evidence updates, and continuation.

The user performs only the platform actions that cannot be completed reliably through the API:

1. import the prepared local notebook into a private Kaggle notebook;
2. attach the exact private dataset version and the ROGII competition data;
3. select the specified accelerator and keep internet disabled;
4. run all cells and provide the notebook reference/version after completion.

The user should not recreate bundles, paste large source blocks, or manually download result files.

## Fail-closed notebook requirements

Every notebook must:

- set all CPU thread controls before importing NumPy/scikit-learn;
- use at most two threads and verify the active native thread pools where possible;
- verify the sealed input archive hash and every manifest entry;
- locate official competition data without assuming a mount slug, then verify the frozen data signature and well count;
- record Python, NumPy, scikit-learn, platform, CPU count, thread limits, accelerator, and internet expectation;
- write all durable outputs under `/kaggle/working`;
- emit a compact top-level run receipt and a downloadable result archive;
- include exact Git/config/input/output hashes, start/end UTC times, wall time, status, selected/reported candidate, controls, and runtime/memory metrics;
- raise on any missing output, failed control, non-finite value, malformed ID/order, hash mismatch, or uncaught exception.

A failed notebook must still preserve a compact failure receipt when possible. It must never silently substitute E006, alter E010 thresholds, add candidates, or continue after an identity failure.

## Proven output-retrieval procedure

After the user runs the notebook:

1. verify the exact notebook version and require status `COMPLETE`;
2. call `kaggle_list_notebook_files` and require every expected filename;
3. call `kaggle_download_notebook_output` with owner, slug, exact version, and exact file path;
4. download the returned signed URL immediately because it is short-lived;
5. store raw outputs only under ignored `scratch/` or private paths;
6. verify bytes, SHA-256, archive members, schemas, source identity, runtime/device, controls, and metrics;
7. independently reproduce deterministic summaries before changing an experiment verdict;
8. if a filename is absent, rerun the canonical notebook instead of repeatedly requesting a 404 output.

## E010 canonical heavy run

- Notebook: `notebooks/training_and_submission/e010_candidate_selector_kaggle.ipynb`
- Builder: `tools/build_e010_kaggle.py`
- Code SHA for input version 1: `fa21e2951a700aeeff355e1cf439c55abc3ab408` or a later explicitly resealed operational commit.
- Runtime: private Kaggle CPU notebook; internet disabled; CPU threads capped at `2`; no GPU required.
- Frozen statistical wall-time gate: 60 minutes.
- Expected top-level outputs:
  - `e010-run-receipt.json`
  - `e010-output-manifest.json`
  - `rogii-e010-results-v1.zip`

The notebook is accepted only after those files are listed and independently verified. No Kaggle competition submission is authorized by an E010 validation run.
