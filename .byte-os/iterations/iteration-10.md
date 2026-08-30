# Iteration 10: Measure actual SIGNOR text exposure

## Evidence

- The frozen source contains 12,637 direct human protein pairs, including 185
  self-relations.
- The named, bounded corpus exposes 10,308 non-self partner pairs after the
  deterministic eight-item limit.
- Released GGI test positives have 24 source overlaps but only 23 actual text
  exposures; test negatives have two of each.
- Masking has zero non-self partner exposure and zero GGI partner exposure.
- Shuffling leaves 377 chance source-pair matches, 12 positive train matches,
  and zero GGI test matches.

## Finding and action

Source overlap alone overstates what the embedding endpoint sees. The leakage
auditor now inspects only the bounded SIGNOR section, excludes self-subject
symbols from partner exposure, and records source and text rates separately.

## Result

The fixed split has sparse, positive-enriched partner leakage. Masking is a
strict identity-removal control; shuffling strongly disrupts identity while
honestly retaining chance collisions.
