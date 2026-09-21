# Findings

Populated after Phase 2. Sanity check that the blooming story is real.

## Top-3 explained variance per layer

| Layer | PC1 | PC2 | PC3 | Cumulative |
| --- | --- | --- | --- | --- |
| 0 (wte) | 0.020 | 0.011 | 0.010 | 4.0% |

Consistent with the spec's warning: 3 linear axes over 768 dims capture very
little variance. Colour and covariance carry the rest of the story.

## Phase 1 sanity: ` Monday` semantic neighbours

Top-8 cosine neighbours of `token(" Monday", id=3321)` in the raw `wte` table:
`[" Tuesday", " Thursday", " Wednesday", " Friday", " Sunday", " Saturday",
"Monday", "Tuesday"]`. All seven other weekdays plus the space-less variants.
Passes the spec's Phase 1 gate.

## 20 largest-covariance tokens at keyframes 6 and 12

*(filled in after `make assets`)*

## Screenshots

*(embedded after Phase 4 polish)*

## Guided-tour verification

Each stop must exist in the data. Spec candidates:
- [ ] glitch-token cluster
- [ ] number line
- [ ] months and days
- [ ] one ambiguous word blooming across layers
- [ ] code tokens

Replace any that do not verify.
