# Host PPTX Notes

Source: `data/AI_wellbore_geology_prediction_task_en.pptx`

Extracted artifacts:

- Text: `eda_findings/host_pptx/slides_text.md`
- Rendered slides: `eda_findings/host_pptx/slides/slide-01.png` through `slide-14.png`
- Embedded media/images: `eda_findings/host_pptx/media/`
- Rendered PDF: `eda_findings/host_pptx/AI_wellbore_geology_prediction_task_en.pdf`

## Host Guidance With Modeling Consequences

1. The core task is geological correlation, not ordinary curve extrapolation. The deck frames the target as deriving hidden horizontal-well TVT from horizontal `XYZ/GR`, known prefix TVT before Prediction Start, and assigned typewell `TVT/GR/Geology`.
2. TVT can increase, decrease, or stay nearly constant after Prediction Start. This supports residual modeling and smoothing, but argues against hard monotonic constraints.
3. Gamma Ray signatures are the main correlation signal. Slides show matching horizontal GR patterns to typewell GR patterns on the TVT scale.
4. The host explicitly notes that horizontal-well GR before Prediction Start can correlate better with post-PS horizontal GR than the assigned typewell GR does. This suggests self-correlation features using the known prefix, not only typewell features.
5. Drilling azimuth matters because the apparent TVT trend depends on the lateral direction relative to geological dip.
6. Offset wells may help because neighboring wells can share similar dip behavior. This should be tested with location/azimuth features and neighbor-derived priors.
7. Evaluation is row-level RMSE over all predicted one-foot points, matching Kaggle's metric page.

## Extra EDA Triggered By The Deck

Additional files generated after reading the PPTX:

- `eda_findings/tables/spatial_offset_per_well.csv`
- `eda_findings/tables/spatial_offset_diagnostics.csv`
- `eda_findings/figures/spatial_end_delta_map.png`
- `eda_findings/figures/hidden_azimuth_distribution.png`

Findings:

- Horizontal `MD` spacing is exactly 1 ft across train rows, matching the one-foot prediction framing.
- Typewell TVT spacing is usually finer than 1 ft: median 0.5 ft and 5-95% range 0.25-0.5 ft, with rare larger gaps.
- Hidden lateral azimuth is highly structured: median azimuth is 133.1 degrees, with another large group near 300 degrees.
- Nearest prediction-start neighbors are very close and often parallel: nearest-neighbor median distance is 478 ft and median azimuth difference is 0.43 degrees.
- However, simple neighbor target priors are weak globally: nearest-neighbor end-delta correlation is only 0.124; 3-neighbor mean end-delta correlation is 0.121. Offset wells may still help as features, but they should be validated carefully.
