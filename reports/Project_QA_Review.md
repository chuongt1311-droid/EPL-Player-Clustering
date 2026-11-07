# Project QA review (2026-11-06)

Method: the `data:validate-data` checklist (methodology, calculations, reasonableness, caveats) applied to the plan and report, plus the claim-to-evidence and reproducibility questions from `peer-review` (its confidentiality gate was not applicable: this is the user's own work), plus a manual read of the pipeline scripts. No subagent was used. Scope: everything up to the merged 9-cluster labels; the player-share step does not exist yet.

## Overall assessment: Share with caveats

The pipeline is sound and the numbers checked out. Four wording and rigor issues should be fixed or stated, and two checks should be added before the player-share step.

## Spot-checks that passed
| Check | Result |
|---|---|
| Season pass counts add to 4,508,312; 2025-26 (330,467) + 2026-27 (70,404) = 400,871 | verified |
| Pitch orientation: crosses start at median x 91.0 to 91.1 in both halves, home and away; none start in own half; shots median 91.9 in all four period/side groups | verified (a flipping error would show here) |
| Flag join: 99.56% to 99.66% matched per season | verified |
| Completion rate equals mean xP in every cluster to two decimals | verified |
| xP calibration (seven bins, match-grouped folds) within 1 point | verified earlier |
| Rule types vs clusters (long ball 74% in one cluster, cross 69%) | verified earlier |

## Issues found
1. **[Medium] The xP "check" is partly circular.** xP is a function of the same inputs the clusters use (position, displacement, chipped, cross). That the clusters order by xP says the input space predicts completion, not that the clusters are independently meaningful. The independent evidence is the actual completion rate, which was never an input: it runs from 0.96 to 0.23 across the clusters. Report discussion item 1 was reworded accordingly.
2. **[Medium] Researcher degrees of freedom.** We searched k (4 to 20) and flag weight (0.5, 1, 1.5, 2, 3) and kept a cell that was stable. The selection was made on stability, which is also the metric reported. Mitigation: a temporal holdout (below) and agreement between neighbouring settings (ARI 0.937).
3. **[Low] xP was not exactly reproducible. FIXED 2026-11-06.** `HistGradientBoostingClassifier` was called without `random_state`; scikit-learn 1.9.0 turns early stopping on automatically above 10,000 rows (`early_stopping='auto'`), which uses a random 10% validation split. Fix: `random_state=0` in `build_pass_table.py`, table rebuilt twice with identical xP (max difference 0.0). Effect of the seed: versus the unseeded build, mean absolute xP difference 0.0095 (99th percentile 0.073, correlation 0.996); mean xP per cluster changes by at most 0.001; all other columns and the row order are unchanged, so the saved cluster labels remain aligned. Pooled out-of-fold AUC 0.9144 (was 0.9139), Brier 0.0853 (was 0.0857); by season AUC 0.903 to 0.923, mean xP within 0.7 points of the actual completion rate in every season.
4. **[Low] The validation scripts lived only in the session scratchpad.** Copied to `validation/` with a README.
5. **[Low] Team effect not controlled for position.** Team-season alone explains almost none of the variance in player-season cluster shares (raw R-squared 0.03 to 0.09, adjusted about 0 for all nine clusters, 266 team-seasons, 3,555 player-seasons with 500+ passes). That is not evidence that team style is irrelevant: position differences (goalkeeper, centre-back, winger) dominate and are spread evenly across teams, which hides a style effect within position. It has not been measured.
6. **[Low] Goalkeeper passes remain in the long-forward cluster (27% start at x < 16.5).**

## New evidence from this review: temporal holdout
Clusters fitted only on 2013-14 to 2018-19 versus clusters fitted only on 2019-20 to 2026-27 (k = 10, weight 2, same evaluation sample): ARI 0.834. Control (two random era-mixed samples of the same size): ARI 0.967. Each era's model against the saved final labels: 0.879 (early) and 0.935 (late). So the clustering is reproducible across eras but not identical: about 0.13 of agreement is lost to era differences, partly real play changes and partly tagging (report 4.6). Era differences are therefore a limitation for role-drift conclusions.

## Required caveats before sharing
- Nine clusters are slices of a continuum (silhouette about 0.21).
- Names are the author's reading of the numbers.
- xP is difficulty of the pass type, not of the moment; it is not player skill.
- The Chipped tag shifted across eras and inside 2023-24; the merged long-forward cluster is the safe version across seasons.
- Era-split clusterings agree at ARI 0.83.
- Player-level results will reflect position and team style as well as individual passing.

## Player-level checks: DONE 2026-11-06 (results in report 4.7)
Items 1 to 5 below were run on 3,555 player-seasons (500+ passes): split-half reliability 0.95 to 0.99; role explains 0.53 of the total variance, with reliable individual variation left within role (0.88 to 0.97); team-season within role explains 0.15 to 0.20 for the long forward ball, wide down the line and link-up shares but about nothing for four others; goalkeepers make 18% of all long-forward-ball passes and need separate treatment; the top-3 lists per cluster are football-sensible.

## Added to the plan before the player-share step
1. Split-half reliability of player shares (odd vs even matches) as the first player-level check.
2. How much of the share variance is explained by position, and what remains within position (needs WhoScored position strings, which the pass table does not yet carry).
3. Team-season effect within position.
4. Sanity examples: pick two or three well-known passing profiles and see whether the shares match football knowledge.
5. Goalkeepers: decide whether to exclude them or give them their own treatment.
