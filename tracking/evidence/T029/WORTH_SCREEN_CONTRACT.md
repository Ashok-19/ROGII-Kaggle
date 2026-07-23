# T029 independent-mechanism selection contract

Frozen before the first result.

- Inputs: verified saved legal OOF actions only: T025 pseudo-coefficient direction, T027 original-view coefficient direction, and T016 nested datum correction.
- Reproduce each parent action exactly before combining.
- Measure row-centered correction-direction correlation.
- Evaluate every single leg, every pair, and all three under exact nonnegative box optimization.
- Evaluate 12,341 nonnegative simplex points on a 0.025 grid, including pooled-best, minimax-domain, and best-every-domain candidates.
- Select a formal successor only if one simplex candidate improves every spatial/typewell domain, gains at least 0.15 versus E011 and 0.03 versus T025, and at least one direction pair has absolute correlation no greater than 0.95.
- Full-data weights are oracle diagnostics only; a formal experiment must regenerate/select parent actions and blend weights inside untouched outer contexts.
