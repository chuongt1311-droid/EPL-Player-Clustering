# Pass Archetypes in the English Premier League: Unsupervised Clustering of 4.5 Million Open-Play Passes

Status report, 2026-11-04. Work in progress: objective, data, method and first clustering run are complete; flag weighting, season drift and the player-level analysis are not.

---

## Abstract

We ask what kinds of passes are played in the Premier League and whether a player can be described by the mix of pass kinds he plays, as an alternative to position labels. We assembled 4,508,312 open-play passes from 13 data files covering the 2013-14 season to 25 October 2026 (season 2026-27 incomplete), each described by where it starts, how far it moves toward goal and sideways, and whether it was lofted or a cross. A gradient-boosted model supplies an expected completion probability (xP, AUC 0.920 out-of-fold on one season) which defines pass risk as 1 - xP and is used to check the clusters, not to build them. K-means with k = 9 on six features gives stable (ARI 0.95) but only weakly separated clusters (silhouette 0.21), which we interpret as slices of a continuum. The nine clusters are readable in football terms and their mean xP runs from 0.96 (wide back pass) to 0.31 (long ball from deep), as expected. The player-share analysis has not yet been done.

Update (flag-weight and PCA tests, section 4.4 and 4.5): halving the weight of the two 0/1 flags barely changes the partition (ARI 0.95 against weight 1), while doubling it separates crosses from other lofted passes (95% of crosses land in one cluster, against 69% at weight 1). A weight of 2 with k = 10 is stable (ARI 0.96) and gives ten readable clusters. PCA is not needed: it is a rotation of six uncorrelated-enough features, keeping all six components reproduces the clustering exactly (ARI 1.000), and keeping 2 or 3 components, as in the source post, destroys it (ARI 0.27 and 0.49). Per-season check (4.6): cluster shares stay within 2.5 percentage points across 14 seasons, but the share of long forward passes tagged Chipped swings from 34% to 58% and back to 43%, which looks like a tagging change and makes the split between lofted forward balls and long balls unreliable across eras.

---

## 1. Objective

Primary question: **What kinds of passes do Premier League players play, and can a player be described by the share of each kind in his passing?**

Three parts, in order:

1. Find a vocabulary of pass types (unit of analysis: one pass).
2. Describe each player (or player-season) as a vector of shares over those types (soft roles).
3. Use the vectors: compare players, find similar players, track role drift over 13 seasons.

Success criteria agreed before clustering: (a) the clusters are stable across random seeds and subsamples; (b) a coach could say what each cluster is; (c) player shares carry information that position labels do not. Explicitly not claimed: that "true" pass types exist, or that cluster names are anything other than our reading of the numbers.

Scope decision: **open play only.** Throw-ins, goal kicks, corners and free kicks are excluded (they are already labelled by the data provider; leaving them in would mainly rediscover those labels).

---

## 2. Data

### 2.1 Event data
WhoScored event data, converted to SPADL actions (type, result, body part, start and end coordinates in metres on a 105 x 68 pitch). Coordinates are absolute in the source; away-team rows were flipped (x' = 105 - x, y' = 68 - y) so every team attacks toward x = 105. Files: `Practice/Possession Value/out/vaep/actions_<tag>.parquet`, one per season (tags h1314 to h2425, about 610k actions each) plus `pl` (2025-26 full and 2026-27 to 25 October 2026).

### 2.2 Open-play passes
Rows with type `pass` or `cross` (SPADL's `cross` excludes corner and free-kick crosses, which are separate types). After joining the qualifier flags (2.3), 4,508,312 passes remain. Per file:

| Tag | Passes | Tag | Passes |
|---|---|---|---|
| h1314 | 339,534 | h2021 | 357,814 |
| h1415 | 333,913 | h2122 | 340,954 |
| h1516 | 331,283 | h2223 | 342,590 |
| h1617 | 333,446 | h2324 | 355,156 |
| h1718 | 342,854 | h2425 | 337,635 |
| h1819 | 346,174 | pl (2025-26 + 2026-27 so far) | 400,871 |
| h1920 | 346,088 | | |

### 2.3 Qualifier flags from the raw events
SPADL discards WhoScored's pass qualifiers, in particular ball height. We re-read every season's `raw.json`, extracted six qualifiers (Chipped, Longball, HeadPass, Throughball, KeyPass, Cross) and joined them to the actions on (game_id, original_event_id). Script: `extract_pass_flags.py`; output `out/passflags_<season>.parquet`. Match rate against the actions exceeds 99.5% in every season (99.56% to 99.66%); unmatched passes (about 0.4%) were dropped.

Flag frequencies in a spot check of the first 20 matches of four seasons (about 19k passes each): Chipped 8.5 to 10.4%, Longball 11.3 to 15.2%, HeadPass 4.7 to 6.8%, KeyPass 1.7 to 2.0%, Throughball 0.29 to 0.35%, LayOff 1.0 to 3.5%. Throughball is too rare to separate from other forward passes into the box, and LayOff and Longball vary between eras, so only Chipped and Cross are used.

---

## 3. Methods

### 3.1 The Chipped qualifier
The provider's definition was not found (a web search returned nothing relevant), so we characterised it empirically on 2024/25 (337,635 open-play passes; Chipped = 8.0%):

| Property | Chipped | Not chipped |
|---|---|---|
| Median length | 33.0 m | 14.2 m |
| Median forward gain (dx) | +19.2 m | +2.6 m |
| Median start x / end x | 57 / 82 | 49 / 55 |
| Completion rate | 46% | 86% |
| Also a header | 0% | 6.2% |
| Also tagged Longball | 51% | 6.2% |
| Is an event-tagged cross | 26% | 0.7% |

At equal length bins chipped passes complete far less often (27% vs 83% at 10 m or less; 46% vs 92% at 10 to 20 m) but, beyond 40 m, more often than grounded ones (51% vs 31%). We read Chipped as a lofted pass. This is an empirical reading, not the provider's documented meaning. The flag's share by season is 8.6% to 10.2% from 2013-14 to 2023-24 and about 8.0% to 8.2% in 2024-25 onward (period 2025-26/2026-27 pooled); whether the recent dip is play or tagging is unchecked.

### 3.2 Pass risk: expected completion (xP)
Risk is defined as **1 - xP**, where xP is the predicted probability that the pass completes.

- Model: gradient-boosted trees (`HistGradientBoostingClassifier`, 200 iterations).
- Inputs: start x, start y, end x, end y, length, dx, dy, header flag, Chipped flag, Cross flag. No outcome information and no defender or pressure information.
- Label: `result_name == success` (offsides and failures count as not completed).
- Validation on 2024/25, folds split by match (GroupKFold, 5):

| Model | AUC | Brier | Log loss |
|---|---|---|---|
| Constant (average completion 82.8%) | 0.500 | 0.1405 | 0.4544 |
| Geometry only | 0.916 | 0.0767 | 0.2544 |
| Geometry + Chipped | 0.920 | 0.0754 | 0.2503 |
| Geometry + Chipped + Cross | 0.920 | 0.0751 | not recorded |

A time-ordered split (train first half of the season, test second half) gave AUC 0.920, Brier 0.0754 for the geometry + Chipped model. Calibration (geometry + Chipped + Cross, folds by match): predicted vs actual completion in seven xP bins agreed to within 0.7 points in every bin (for example 0.502 vs 0.503 in 0.4 to 0.6, 0.982 vs 0.984 in 0.95 to 1.0). Before the Cross flag the model predicted 26.0% completion for crosses against an actual 21.6%; with the flag, 21.4% against 21.6%.

For the all-season table, xP was fitted in two folds by game-id parity (train on even games, predict odd, and the reverse), so no pass is scored by a model trained on its own match. The model uses `random_state=0` (added 2026-11-06; an unseeded earlier build differed from the seeded one by a mean absolute 0.009 in xP, correlation 0.996, and rebuilding twice now gives identical values). The all-season out-of-fold xP was checked after the seeded rebuild: pooled AUC 0.9144 and Brier 0.0853 over 4,508,312 passes, and by season AUC 0.903 to 0.923 and Brier 0.074 to 0.097, with mean xP within 0.7 points of the actual completion rate in every season (for example 82.60% vs 83.10% in 2024-25, 79.03% vs 79.05% in 2013-14). Accuracy is slightly lower in the early seasons (AUC 0.903 to 0.913 in 2013-14 to 2017-18, against 0.918 to 0.923 from 2019-20), consistent with noisier early event coordinates or different tagging.

Risk bands used for description: low xP >= 0.90 (65% of 2024/25 passes), medium 0.75 to 0.90 (12%), high < 0.75 (23%).

### 3.3 Rule-based pass types (a yardstick, not the clustering)
A hand-written list of nine types, first match wins. dx = end_x - start_x, dy = end_y - start_y, L = length (m), gain = distance to goal centre at the start minus at the end, Box = x >= 88.5 and 13.84 <= y <= 54.16.

| Type | Rule |
|---|---|
| Cross | event tagged cross |
| Box entry / through ball | end in box, start outside box, dx >= 10 |
| Final-third combination (includes cutbacks) | start_x >= 70 and end_x >= 70 |
| Switch | abs(dy) >= 30 and L >= 30 |
| Long ball | L >= 32 and dx >= 20 |
| Progressive | gain >= 10 |
| Back / recycle | dx <= -5 |
| Short forward | dx >= 5 |
| Sideways | everything else |

The thresholds are our starting guesses and were not tuned. Cutbacks were a separate rule at first but only 242 passes (0.1%) met it in 2024/25, so they were merged into the final-third combination; through balls and key passes were not separated because the Throughball tag marks only about 0.3% of passes. Rule types on 2024/25 (before the cutback merge): back/recycle 20.4% of passes (mean xP 0.98), sideways 22.0% (0.91), short forward 16.8% (0.90), final-third combination 15.7% (0.82), progressive 13.0% (0.74), long ball 4.9% (0.36), switch 2.3% (0.69), box entry 2.1% (0.49), cross 2.7% (0.27).

### 3.4 Clustering features
Six features per pass:

| Feature | Definition | Scaling |
|---|---|---|
| start_x | start x, 0 to 105 m | z-score |
| start_w | abs(start_y - 34), distance from the centre line | z-score |
| dx | metres gained toward goal | z-score |
| dy_folded | dy multiplied by +1 if start_y <= 34, else -1 (left/right mirror; + means moving toward the far side, - toward the near touchline) | z-score |
| chipped | Chipped flag | raw 0/1 |
| cross | cross flag | raw 0/1 |

Z-scores use pooled means and SDs over all 4.5M passes (start_x 50.5 m / SD 23.5, start_w 16.6 / 9.4, dx 4.3 / 15.2 in 2024/25). The flags are kept raw because z-scoring a rare binary (Chipped SD about 0.27) would put chipped and grounded passes about 3.7 SD apart; raw 0/1 puts them 1 unit apart, roughly one SD of the numeric features. This weighting is a choice; it is tested at 0.5, 1 and 2 in section 4.4.

Not used as inputs: end_x (equal to start_x + dx and correlated 0.80 with start_x), length, angle and gain (functions of dx and dy; gain correlates 0.95 with dx and angle -0.80), and **xP** (a function of the same inputs; it is used afterwards to check the clusters).

### 3.5 Clustering and validation
- Algorithm: k-means (scikit-learn), Euclidean distance.
- Fit set: 300,000 passes sampled with an equal number per season label (capped by availability for the incomplete 2026-27 season).
- Candidate k: 4 to 20.
- Silhouette: on a 20,000-pass subsample of the fit set.
- Stability: for each k, two repeats of fitting k-means (n_init = 2) on two disjoint 100,000-pass stratified samples with different seeds, assigning a common 100,000-pass evaluation sample with both, and computing the adjusted Rand index (ARI) between the two labellings. We report the mean and the minimum over the two repeats.
- Final model: k = 9, n_init = 10, random_state = 0, fitted on the 300k set, then every pass assigned to the nearest centroid. Clusters were renumbered by median dx.

---

## 4. Results

### 4.1 Choice of k

| k | Silhouette | ARI mean | ARI min | Smallest cluster |
|---|---|---|---|---|
| 4 | 0.227 | 0.968 | 0.964 | 15.6% |
| 5 | 0.218 | 0.955 | 0.940 | 7.5% |
| 6 | 0.207 | 0.769 | 0.567 | 6.7% |
| 7 | 0.215 | 0.577 | 0.505 | 5.6% |
| 8 | 0.212 | 0.798 | 0.629 | 5.1% |
| **9** | 0.211 | **0.952** | **0.950** | 4.7% |
| 10 | 0.213 | 0.834 | 0.834 | 3.0% |
| 11 | 0.215 | 0.620 | 0.612 | 2.8% |
| 12 | 0.201 | 0.548 | 0.451 | 2.8% |
| 13 | 0.204 | 0.789 | 0.786 | 2.5% |
| 14 | 0.204 | 0.825 | 0.689 | 0.3% |
| 15 | 0.202 | 0.817 | 0.687 | 0.3% |
| 16 | 0.198 | 0.683 | 0.673 | 0.3% |
| 17 | 0.197 | 0.648 | 0.632 | 0.3% |
| 18 | 0.192 | 0.690 | 0.636 | 0.3% |
| 19 | 0.197 | 0.601 | 0.572 | 0.3% |
| 20 | 0.193 | 0.758 | 0.707 | 0.3% |

(Inertia falls smoothly from 696,235 at k = 4 to 298,074 at k = 20 with no sharp elbow.) k = 9 was chosen as the largest k with high stability (ARI >= 0.95) and no tiny cluster. Stability is not monotone in k: k = 4, 5 and 9 reproduce, k = 7, 11 and 12 do not. Silhouette is about 0.21 across the whole range.

### 4.2 The nine clusters (k = 9)
dx = median forward gain; "sideways" = median dy_folded.

| # | Share | Start x | Dist. from centre | dx | Sideways | Chip | Cross | xP | Working name |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 11% | 43.6 | 24.9 | -10.5 | +12.2 | 2% | 1% | 0.96 | Wide back pass, inward |
| 1 | 12% | 66.7 | 9.1 | -1.9 | +6.3 | 2% | 0% | 0.83 | Central link-up, attacking half |
| 2 | 18% | 73.7 | 27.3 | -0.6 | +4.0 | 4% | 4% | 0.81 | Wide short pass, final third |
| 3 | 11% | 34.1 | 8.6 | +3.2 | +16.6 | 4% | 0% | 0.93 | Central diagonal across, from deep |
| 4 | 13% | 32.9 | 8.1 | +7.6 | -11.5 | 3% | 0% | 0.89 | Central ball out to the flank |
| 5 | 12% | 64.6 | 13.5 | +8.2 | -10.0 | 4% | 0% | 0.86 | Forward ball out wide, attacking half |
| 6 | 14% | 36.9 | 26.0 | +11.1 | +1.3 | 9% | 0% | 0.69 | Wide pass forward, down the line |
| 7 | 5% | 74.8 | 21.8 | +11.7 | +29.4 | 76% | 43% | 0.47 | Lofted delivery into or across the box |
| 8 | 5% | 21.9 | 11.2 | +44.7 | +0.3 | 37% | 0% | 0.31 | Long ball from deep |

Median lengths: cluster 7 33.6 m, cluster 8 47.5 m, the rest 9.7 to 19.1 m. Completion rate equals mean xP to two decimals in every cluster. 42% of cluster 8 starts inside x < 16.5 m (3% of cluster 0, 17% of cluster 4, 14% of cluster 3, 9% of cluster 6), consistent with goalkeeper and deep-defender distribution; we have not separated goalkeepers.

### 4.3 Agreement with the rule-based types
Where each rule type lands (share of that rule type in the cluster):

- Long ball: 74% in cluster 8.
- Cross: 69% in cluster 7.
- Progressive: 45% in cluster 6, 21% in cluster 4.
- Final-third combination: 52% in cluster 2, 24% in cluster 1, 21% in cluster 5.
- Back/recycle: 44% in cluster 0, then 20% / 18% / 9% in clusters 1, 2, 3.
- Switch: 42% in cluster 7, 33% in cluster 3, 13% in cluster 0.
- Sideways: spread over every cluster (10 to 21% each).

What the clusters are made of in terms of rule types: cluster 0 is 75% back/recycle; cluster 7 is 43% crosses, 27% switches, 11% box entries, 10% final-third; cluster 8 is 80% long balls; cluster 6 is 47% progressive.

### 4.4 Flag-weight test (chipped and cross at 0.5, 1 and 2)
The same weight was applied to both flags (chipped, cross ∈ {0.5, 1, 2}); everything else as in 3.5. Full k sweep (4 to 20) for each weight; k = 9 compared in detail.

At k = 9:

| Flag weight | Silhouette | Stability (ARI) | ARI of partition vs weight 1 (common 100k passes) | Share of all crosses in their top cluster | Share of all chipped passes in top-2 clusters |
|---|---|---|---|---|---|
| 0.5 | 0.214 | 0.920 | 0.947 | 59% | 55% |
| 1 | 0.211 | 0.952 | (reference) | 69% | 60% |
| 2 | 0.217 | 0.683 | 0.705 | 95% | 72% |

(ARI between weights 0.5 and 2 is 0.690.)

The flagged clusters at k = 9:

| Weight | Flagged clusters (share of passes; chip rate; cross rate; median dx; median length; mean xP) |
|---|---|
| 0.5 | one: 5%; 66%; 35%; +11.5 m; 34.0 m; 0.53 |
| 1 | one: 5%; 76%; 43%; +11.6 m; 33.6 m; 0.47 |
| 2 | two: crosses 3%; 84%; 94%; +3.4 m; 25.3 m; 0.23 and lofted forward balls 4%; 93%; 0%; +26.6 m; 38.5 m; 0.48 |

Stability across k, mean ARI (weights 0.5 / 1 / 2): k = 7: 0.64 / 0.58 / **0.97**; k = 9: 0.92 / **0.95** / 0.68; k = 10: 0.84 / 0.83 / **0.96**; k = 13: 0.67 / 0.79 / **0.95**; k = 4 and 5: 0.97 and 0.97 / 0.97 and 0.96 / 0.93 and 0.60. Silhouette stays between 0.20 and 0.23 for every weight and every k, so it does not discriminate between the settings; stability and interpretability are the deciding criteria.

Findings:
1. **Below 1 nothing changes much.** Weight 0.5 and weight 1 give nearly the same partition (ARI 0.947); the flags just define one lofted-delivery cluster either way.
2. **At 2 the structure changes.** The single lofted cluster splits into a cross cluster and a lofted-forward cluster, and several other clusters shift slightly (partition ARI 0.705 against weight 1). k = 9 is not a stable choice at weight 2 (0.683), but k = 10 is (0.958), as are k = 7 (0.969) and k = 13 (0.948).
3. **The weight is a modelling decision, not an estimated quantity.** The data do not choose it; the data only tell us what each choice implies.

Detail at weight 2, k = 10 (fit on the same 300k sample; clusters renumbered by dx; "sideways" = dy_folded):

| # | Share | Start x | Dist. from centre | dx | Sideways | Chip | Cross | xP | Working name |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 11% | 43.7 | 24.9 | -10.3 | +13.0 | 1% | 0% | 0.96 | Wide back pass, inward |
| 1 | 12% | 67.5 | 9.7 | -2.0 | +6.6 | 1% | 0% | 0.83 | Central link-up, attacking half |
| 2 | 17% | 72.5 | 27.5 | -0.7 | +3.8 | 1% | 0% | 0.85 | Wide short pass, final third |
| 3 | 11% | 34.5 | 8.8 | +2.7 | +16.3 | 0% | 0% | 0.94 | Central diagonal across, from deep |
| 4 | 3% | 91.6 | 24.3 | +3.4 | +22.9 | 84% | 95% | 0.22 | Cross |
| 5 | 12% | 33.1 | 8.0 | +7.3 | -11.2 | 2% | 0% | 0.89 | Central ball out to the flank |
| 6 | 12% | 64.7 | 13.3 | +8.0 | -10.1 | 3% | 0% | 0.86 | Forward ball out wide, attacking half |
| 7 | 13% | 36.1 | 25.4 | +10.4 | +0.9 | 2% | 0% | 0.71 | Wide pass forward, down the line |
| 8 | 4% | 46.8 | 18.6 | +26.8 | +22.0 | 94% | 0% | 0.48 | Lofted forward / diagonal ball |
| 9 | 4% | 16.7 | 9.3 | +45.0 | -4.8 | 29% | 0% | 0.33 | Long ball from deep (49% start at x < 16.5) |

The eight non-flag clusters are nearly identical to the weight-1, k = 9 solution; the difference is that crosses (cluster 4, xP 0.22, starting around x = 92) are separated from lofted passes that travel forward from midfield (cluster 8, xP 0.48).

Robustness of the weight-2, k = 10 solution. Stability (mean ARI) around k = 10 at three weights:

| Flag weight | k = 9 | k = 10 | k = 11 |
|---|---|---|---|
| 1.5 | 0.68 | 0.69 | 0.72 |
| 2 | 0.68 | **0.96** | 0.76 |
| 3 | 0.79 | 0.80 | **0.96** |

Stability is high at one k per weight (k = 10 at weight 2, k = 11 at weight 3) and nowhere in this range at weight 1.5, so a single stable cell is thin evidence on its own. The two stable solutions agree: weight 2, k = 10 and weight 3, k = 11 match at ARI 0.937 on all 4,508,312 passes. The cross cluster is the same in both (3% of passes, 95% of its passes shared, xP 0.20 to 0.22), and the seven non-flagged clusters are carried over with 95% to 98% of their passes. The extra cluster at weight 3 comes from splitting the lofted-forward and long-ball group into three: a lofted diagonal from midfield (3%, start x 57 m, dx +20 m, 30 m across, xP 0.58), a lofted straight forward ball (4%, start x 38 m, dx +30 m, xP 0.44) and a long ball from deep (3%, xP 0.29).

Is k = 11 worth it? Variance explained across all passes: xP 43.1% (weight 2, k = 10) vs 45.0% (weight 3, k = 11); completion 19.7% vs 20.7%. Reliability of player-season shares (player-seasons with at least 500 passes, n = 3,555; share of the between-player variance in a cluster's share that exceeds binomial sampling noise): 0.89 to 0.99 for every cluster in both solutions; the weakest is the k = 11 lofted diagonal (0.887, median 26 passes per player-season). This measures sampling noise only; it does not separate player skill from team style.

**Decision (2026-11-06): chipped and cross weight 2, k = 10.** Reasons: stable (ARI 0.96), milder weight than 3, one fewer share per player, and the k = 11 gain is small (about 2 points of xP variance). k = 11 stays available as a refinement of the lofted group; whether the diagonal / straight split separates different kinds of players has not been tested (the pass table has player ids but no names).

### 4.5 Is PCA needed?
No. The reasons, with numbers (6 features, fit sample of 300k, weight 1):

- **PCA does not change what k-means finds when all components are kept.** It is a rotation of the same six columns and k-means distances are rotation-invariant. Keeping 6 components reproduces the no-PCA k = 9 partition exactly (ARI 1.000).
- **There is no collinearity to remove.** end_x, length, angle and gain were already left out because they are functions of the retained features. Explained variance of the principal components of the z-scored numerics plus flags: PC1 37.5%, PC2 23.7%, PC3 20.3%, PC4 16.3%, PC5 1.7%, PC6 0.5%. The first four components are mixtures of the four numeric features (each of the four numeric features loads strongly on at least two of them), so they are not a compressed version of them; PC5 and PC6 are essentially the chipped (loading 0.94) and cross (0.95) flags.
- **Compressing loses information we need.** Keeping the first 2 components retains 61% of the variance and gives ARI 0.274 against the no-PCA clustering; 3 components retain 82% and give ARI 0.492; 4 components retain 97.8% and give ARI 0.957, but this simply discards both flags (PC5 and PC6), so the 0.957 hides the fact that the cluster of lofted deliveries can no longer be found. 5 components retain 99.5% (ARI 0.984).
- **Cost is not an issue.** Six features and 300k points fit in seconds; PCA would not speed anything up.
- **Interpretability.** The raw features are in metres on the pitch; principal components are mixtures that would have to be read through loadings.

The source blog post that motivated this project clustered 20 correlated per-90 stats and used two principal components because 2-D plotting needed them. That situation does not apply here. PCA stays useful only as a way to plot clusters in two dimensions, which does not influence the clustering itself.

### 4.6 Per-season drift (weight 2, k = 10, all 4,508,312 passes)
Share of each season's open-play passes in each cluster (%; 2026-27 is only the first 80 matches):

| Season | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2013-14 | 9.9 | 12.7 | 17.0 | 10.7 | 3.5 | 11.8 | 12.3 | 12.9 | 4.5 | 4.8 |
| 2015-16 | 9.4 | 12.4 | 17.9 | 10.2 | 3.4 | 11.4 | 12.4 | 13.5 | 4.5 | 4.9 |
| 2017-18 | 10.2 | 12.1 | 17.8 | 10.9 | 3.0 | 11.5 | 12.0 | 13.3 | 4.6 | 4.5 |
| 2019-20 | 11.0 | 10.6 | 18.2 | 11.3 | 2.9 | 12.2 | 11.2 | 13.9 | 4.8 | 3.8 |
| 2021-22 | 11.0 | 10.7 | 17.6 | 11.7 | 2.8 | 12.9 | 11.2 | 13.6 | 4.7 | 3.8 |
| 2022-23 | 10.9 | 10.8 | 16.9 | 12.4 | 2.7 | 13.3 | 11.1 | 13.3 | 4.7 | 3.8 |
| 2023-24 | 10.7 | 12.1 | 16.7 | 12.3 | 2.6 | 13.3 | 11.7 | 12.9 | 3.9 | 3.8 |
| 2024-25 | 11.0 | 11.6 | 17.0 | 12.2 | 2.7 | 13.3 | 12.3 | 12.4 | 3.6 | 3.8 |
| 2025-26 | 11.0 | 11.7 | 16.5 | 12.4 | 2.9 | 13.1 | 12.5 | 12.2 | 3.6 | 4.2 |
| 2026-27 (partial) | 10.9 | 12.3 | 16.4 | 12.2 | 2.8 | 13.2 | 12.9 | 11.5 | 3.5 | 4.3 |

(Cluster names as in the weight-2, k = 10 table of 4.4: 0 wide back pass, 1 central link-up, 2 wide short pass in the final third, 3 central diagonal across, 4 cross, 5 central ball out to the flank, 6 forward ball out wide in the attacking half, 7 wide pass forward down the line, 8 lofted forward / diagonal, 9 long ball from deep.) The largest range of any cluster over the 14 seasons is 2.5 percentage points (cluster 7), so the vocabulary is broadly stable across 13 years.

Observed changes:
- **Slow trends:** clusters 0 (wide back pass), 3 (central diagonal across) and 5 (central ball out to the flank) rise by about 1 to 2 points between 2013-14 and 2024-25; cluster 4 (crosses) falls from 3.5% to 2.6 to 2.9%, in line with the event-tagged cross share (3.42% to 2.52 to 2.92%); cluster 1 dips to 10.6% in 2019-20 and returns to 11.6 to 12.3%.
- **xP does not drift against reality:** league completion rate and mean xP differ by at most 0.6 points in every season (for example 83.10% vs 82.62% in 2024-25), so the risk model stays calibrated across eras.
- **Cluster 8 (lofted forward) drops abruptly in 2023-24** (4.7% to 3.9%), the same year the Chipped share falls from 10.17% to 8.59%.

Is the Chipped dip play or tagging? Test on passes defined by geometry alone (length at least 30 m, dx at least 15 m, not an event-tagged cross):

| Season | Share of all passes | Chipped share among them | Completion if chipped | Completion if grounded |
|---|---|---|---|---|
| 2013-14 | 9.7% | 34.5% | 43.8% | 38.0% |
| 2016-17 | 10.1% | 35.0% | 46.8% | 37.5% |
| 2018-19 | 8.4% | 45.1% | 44.6% | 37.4% |
| 2020-21 | 8.0% | 54.3% | 43.9% | 38.7% |
| 2022-23 | 8.3% | 57.8% | 43.5% | 40.9% |
| 2023-24 | 7.7% | 49.9% | 47.7% | 37.7% |
| 2024-25 | 7.6% | 44.0% | 49.7% | 37.7% |
| 2025-26 | 8.1% | 42.0% | 44.9% | 36.2% |

The number of long forward passes moves slowly (9.7% to 7.6% to 8.1%), but the share of them tagged Chipped swings from about 34% (2013 to 2017) to 58% (2022-23) and back to 42 to 44% (2024 onward). A swing of that size in how often a long forward pass is lofted, over a few seasons, is more likely a change in how the passes are tagged than in how teams play, but this cannot be proved from these data. The two groups' completion rates (about 44 to 50% vs 36 to 41%) stay roughly constant, so the tag keeps separating two kinds of pass; what moves is how many long forward passes receive it. Consequence: the split between cluster 8 (lofted forward, 94% chipped) and cluster 9 (long ball from deep, 29% chipped) is partly a tagging artefact across eras. Combined, clusters 8 and 9 follow the geometry-defined group (9.3% in 2013-14, 8.5% in 2022-23, 7.4% in 2024-25, 7.8% in 2025-26).

Is the tag at least stable inside a season? Monthly share of geometry-defined long forward passes tagged Chipped:

| Season | Aug | Sep | Oct | Nov | Dec | Jan | Feb | Mar | Apr | May |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | 0.54 | 0.59 | 0.56 | 0.57 | 0.58 | 0.55 | 0.58 | 0.60 | 0.61 | 0.60 |
| 2023-24 | 0.66 | 0.63 | 0.59 | 0.51 | 0.46 | 0.47 | 0.45 | 0.40 | 0.44 | 0.49 |
| 2024-25 | 0.45 | 0.43 | 0.45 | 0.43 | 0.46 | 0.43 | 0.43 | 0.41 | 0.45 | 0.44 |

2022-23 and 2024-25 are flat (0.54 to 0.61 and 0.41 to 0.46). 2023-24 steps down from 0.59 to 0.66 in August to October to 0.40 to 0.49 from December, inside one season; the SD of the tagged share across matches is 1.95 times the sampling expectation that season against 1.34 to 1.60 in every other season. A one-third drop in how often long forward balls are lofted in the middle of a season is far more likely a change in tagging than in play, though this cannot be proved from these data. Across teams within a season the share varies more than sampling noise (SD 0.03 to 0.06 against 0.013 to 0.014 expected; lowest to highest team 0.35 to 0.52 in 2024-25), which could be team style or annotation differences; the data cannot separate the two.

**Decision (2026-11-06): merge clusters 8 and 9 into one "long forward ball" cluster, giving 9 clusters for the main analysis** (labels in `out/labels_k9_merged.parquet`; the unmerged weight-2, k = 10 labels stay in `out/labels_k10_w2.parquet`). The merged cluster follows the geometry and does not depend on the Chipped tag. Its profile: 8.6% of passes; median start x 33.7 m; median dx +36.1 m; median length 43.0 m; 62% chipped; mean xP 0.41 (completion 0.41); 27% start inside x < 16.5 m (so it still contains goalkeeper distribution). Its season share falls gradually from 9.3 to 9.7% (2013-14 to 2017-18) through 8.3 to 8.6% (2018-19 to 2022-23) to 7.4 to 7.8% (2023-24 onward), a range of 2.3 points, which fits the geometry-defined trend in the table above. The other eight clusters are unchanged from the weight-2, k = 10 table in 4.4 (numbers 0 to 7). The split between the lofted forward ball and the long ball from deep may be used as an optional detail from 2024-25 onward, where the tag is stable.

### 4.7 Player-season shares and the player-level checks (9 merged clusters)
Unit: one player in one season with at least 500 open-play passes (3,555 player-seasons). Each is a vector of nine shares summing to 1. Positions come from the WhoScored position string of the matches the player started (mode per player-season; roles GK, DEF, MID, WIDE, FW; the role map from `possvalue/season_trust.py`). Files: `build_player_table.py` -> `out/player_matches.parquet`; `player_shares.py` -> `out/player_season_shares.parquet` and the checks below. Role counts: DEF 1,524; MID 1,109; WIDE 368; FW 331; GK 223.

**1. Split-half reliability** (matches alternately assigned to two halves by date, correlation across player-seasons, then Spearman-Brown corrected to the full sample): 0.95 (wide back pass), 0.98 (central link-up), 0.98 (wide short final third), 0.98 (central diagonal across), 0.96 (cross), 0.98 (central out to flank), 0.97 (forward out wide), 0.96 (wide down the line), 0.99 (long forward ball). A player's mix is a stable property within a season, not sampling noise.

**2. Position.** Role explains 0.53 of the total variance of the share vectors. Per cluster: central link-up 0.75, long forward ball 0.74, forward out wide 0.70, wide down the line 0.67, wide back pass 0.54, central out to flank 0.41, wide short final third 0.39, central diagonal across 0.32, cross 0.26. What is left within role is highly reliable (split-half reliability of the within-role residuals 0.88 to 0.97), so roughly half of the between-player variation is not position and is not noise. This satisfies success criterion (c) in part: shares carry reliable information beyond position labels. It does not show that the extra information is skill, since footedness, side of the pitch, team, and era could all contribute.

Mean shares (%) by role:

| Role | 0 wide back | 1 central link-up | 2 wide short final third | 3 central diagonal | 4 cross | 5 central out to flank | 6 forward out wide | 7 wide down the line | 8 long forward |
|---|---|---|---|---|---|---|---|---|---|
| DEF | 14.3 | 5.1 | 15.4 | 13.3 | 2.9 | 13.0 | 7.4 | 19.8 | 8.9 |
| MID | 8.3 | 16.7 | 15.4 | 10.5 | 2.0 | 12.9 | 18.1 | 9.9 | 6.2 |
| WIDE | 10.4 | 16.0 | 33.2 | 3.8 | 7.3 | 3.6 | 11.4 | 9.8 | 4.6 |
| FW | 7.5 | 25.1 | 30.4 | 3.4 | 4.6 | 3.6 | 15.5 | 6.8 | 3.1 |
| GK | 0.0 | 0.0 | 0.0 | 24.0 | 0.0 | 32.5 | 0.0 | 1.1 | 42.5 |

**3. Team-season within role** (share of the within-role variance explained by the team-season, adjusted for the number of groups; 266 team-seasons): long forward ball 0.20, wide down the line 0.17, central link-up 0.15, forward out wide 0.12, wide back pass 0.04, and about zero (-0.05 to 0.00) for cross, wide short final third, central diagonal across and central out to flank. Team style leaks into some shares (mainly the long ball and the wide forward and link-up shares) and hardly into the others. Not separated from team effects: a player's own tendency to play in a given team.

**4. Goalkeepers** (players with 500+ passes): their mean mix is 42.5% long forward ball, 32.5% central out to flank, 24.0% central diagonal across, and nothing in the other clusters. They make 18.3% of all long-forward-ball passes, 10.6% of the central-out-to-flank passes and 8.4% of the central-diagonal passes. In a mixed comparison they dominate these three shares; outfield and goalkeeper analyses should be run separately.

**5. Sanity examples** (top three player-seasons per cluster in 2024-25, 1,000+ passes; the lists were read, not blind-tested): wide back pass: Pervis Estupinan, Matt Doherty, Lucas Digne (full-backs); central link-up: Mateo Kovacic, Cole Palmer, Ilkay Gundogan; wide short final third: Mohamed Salah, Antoine Semenyo, Matheus Nunes; central diagonal across: Tosin Adarabioyo, Bernd Leno, Jan Bednarek; cross: Antonee Robinson, Tyrick Mitchell, Vitalii Mykolenko (full-backs); central out to flank: Bernd Leno, Bart Verbruggen, Andre Onana (goalkeepers); forward out wide: Kovacic, Sander Berge, Martin Odegaard (midfielders); wide down the line: Lewis Hall, Tino Livramento, Ola Aina (full-backs); long forward ball: Jordan Pickford, Robert Sanchez, Mark Flekken (goalkeepers). The labels match football expectations; the lists show that most of the extreme shares belong to the expected position, which is also why the position effect is large.

### 4.8 Similar-player search
Goal: given a player-season, list the outfield player-seasons with the closest pass mix (a shortlist for replacement ideas). Goalkeepers are excluded (own mix, 4.7). Files: `similarity_eval.py` (metric validation, noise scale), `similar_players.py` (search: `python similar_players.py "Name" --season 2024-25 --pool same-season --k 8`; pools `all`, `recent` = 2025-26 and 2026-27, `same-season`), `out/similarity_noise.json`.

**Choosing the distance.** Four candidates were tested on the 3,332 outfield player-seasons (500+ passes) with two questions: (a) given one half of a season's matches for a player, how often is the other half of the same player-season the nearest of all 3,332 other halves (chance 0.03%); (b) is the nearest player-season in a different season the same player.

| Distance | (a) half vs half, top 1 | (a) median rank of the true match | (a) in top 5 | (b) same player other season, top 1 | (b) top 5 |
|---|---|---|---|---|---|
| Raw Euclidean on shares | 10.7% | 18 | 28.4% | 13.0% | 32.2% |
| **Hellinger (Euclidean on square-root shares)** | **12.3%** | **14** | **32.8%** | **15.6%** | **37.4%** |
| z-scored shares | 10.8% | 16 | 29.8% | 13.5% | 35.3% |
| z-scored square-root shares | 12.1% | 14 | 32.2% | 15.5% | 36.6% |

Hellinger was chosen (best, and it needs no tuning). The halves each have only about half of a season's passes, so these accuracies are conservative for full-season vectors.

**What the accuracy says.** The search is a shortlist, not a twin finder: the true match is the nearest of 3,332 only about one time in eight, but its median rank is 14 (top 0.4%), against a chance rank of about 1,666. With nine shares, many players of the same type sit close together. This also means the search finds players with a similar role-and-style type of pass mix, not an individual's identity.

**Noise scale.** The median Hellinger distance between a player-season's own two halves is 0.103; rescaled to two different full-season vectors the sampling noise is 0.073. Neighbours closer than 0.073 are not distinguishable from sampling noise and are flagged "(within noise)".

**Examples** (same-season pool, 2024-25, from `similar_players.py`; read by hand, not blind-tested): Declan Rice: Mateus Fernandes (0.072, within noise), Enzo Fernandez (0.080), James Maddison (0.083), Alexis Mac Allister (0.084), Daichi Kamada (0.099). Trent Alexander-Arnold: Pedro Porro (0.132), Andy Robertson (0.170), James Justin (0.190). With the pool `all`, Rice's neighbours come from 2013 to 2019 (Tielemans, Tom Davies, Meyler, Henderson), all within noise: his mix is a generic central-midfielder mix, and cross-era neighbours carry the era differences of 4.6.

**Limits.** (1) Only pass types are compared: no passing volume, carries, defending, or pass value (xT/VAEP), so a "similar" player may be a different level of player. (2) Nine shares crowd the space, especially for central midfielders. (3) Role and team-style effects (4.7) are inside the distance; `--same-role` restricts candidates to the same position. (4) Only 500+ pass seasons are included, so injured or part-time players such as Rodri in 2024-25 are absent. (5) Cross-era searches mix tagging and style changes. (6) Neighbours' season shares are a noisy single-season view.

### 4.9 Adding value: pass quality alongside the profile
Goal: know not only that two players pass alike but how much their passing adds. Files: `pass_value.py` -> `out/player_season_quality.parquet`; `similar_players.py --quality` shows it next to each neighbour.

**Which value measure.** xT gain, not VAEP. The project's own trust test (`Practice/Possession Value/out/vaep/season_trust_log.txt`) found split-half reliability at 20 matches of 0.87 for xT gain against 0.39 for VAEP/90 (0.72 without shots), and season-to-season correlation of 0.78 against 0.37. Pass xT = xT(end cell) - xT(start cell) for completed passes, 0 for failed ones, on the 12 x 8 grid fitted only on the 6,763,079 actions before 2024-25 (peak 20.9 goals per 100). The grid has not seen 2024-25 or later, so quality is defined only for 2024-25 and 2025-26 (2026-27 has no player with 900 minutes and 500 passes yet): 496 outfield player-seasons with 900+ minutes (251 and 245).

**What each cluster is worth** (2024-25 onward passes; xT x100 = goals per 100 passes):

| Cluster | Share of passes | xT per pass | Completion | Share of all pass xT |
|---|---|---|---|---|
| 0 wide back pass | 11.0% | -0.07 | 0.95 | -5.6% |
| 1 central link-up | 11.7% | 0.15 | 0.84 | 12.4% |
| 2 wide short final third | 16.7% | 0.08 | 0.87 | 9.8% |
| 3 central diagonal across | 12.3% | 0.03 | 0.95 | 2.2% |
| 4 cross | 2.8% | 1.89 | 0.24 | 37.2% |
| 5 central out to flank | 13.2% | 0.03 | 0.92 | 2.4% |
| 6 forward out wide | 12.4% | 0.13 | 0.89 | 11.2% |
| 7 wide down the line | 12.2% | 0.12 | 0.75 | 10.1% |
| 8 long forward ball | 7.6% | 0.38 | 0.40 | 20.3% |

Crosses are under 3% of passes but 37% of all pass xT; long forward balls are 7.6% of passes and 20% of pass xT; the wide back pass is slightly negative. Risk and value go together: the high-value clusters are the low-completion ones.

**Player metrics** (per player-season): passes per 90; pass xT per 90 (x100); xT per pass; completion over expected (actual completion minus the mean xP, per pass, in points); xT versus mix (xT per pass minus what the player's cluster shares would earn at league-average value per cluster). Split-half reliability (matches alternated by date; Spearman-Brown, full season): passes per 90 0.94; xT per 90 0.80; xT per pass 0.83; completion over expected 0.79; xT versus mix 0.58. Role-adjusted z-scores (within DEF, MID, WIDE, FW) are used for comparison.

**Mix versus quality.** The player's mix alone (expected xT per pass from the shares) explains 0.69 of the variance of xT per pass between player-seasons. The remaining 0.31 is partly noise (reliability of the residual 0.58) and partly what the player does with the same kind of pass, for example completes it more often or places it further up the pitch.

**Using it.** `python similar_players.py "Declan Rice" --season 2024-25 --pool same-season --quality` lists neighbours with xT/90, xT per pass and completion over expected; `--min-xt-z 0` keeps only neighbours at or above the role average. Example, Rice 2024-25 (xT/90 3.6, role z -0.5): Enzo Fernandez 8.7 (z +1.0), James Maddison 12.2 (z +1.9), Mac Allister 5.1 (z 0.0), Mateus Fernandes 3.6 (z -0.5, within noise of Rice's profile). Trent Alexander-Arnold 2024-25: Pedro Porro xT/90 13.7 (z +2.8), Andy Robertson 11.3 (z +2.0), James Justin 3.8 (z -0.7).

**Limits.** (1) Pass xT only counts how much a completed pass moves the ball toward threat; it ignores carries, ball winning, positioning and what happens without the ball, so low pass xT does not mean a poor player: Declan Rice, valued for his all-round game, sits below the midfield average here. (2) Completion over expected is not pressure-adjusted and differs by role (defenders +1.8 points, forwards -2.5), so only the role z-score is meaningful. (3) Two seasons, 496 player-seasons, so the quality search pool is small. (4) Teammates and team context add to xT. (5) xT varies by role and team strength; a better-team player gets more of the ball in good places. (6) No uncertainty interval per player yet.

### 4.10 Role drift
Question: how do pass mixes change over the 13 seasons, at the league level and for individual players? Files: `player_drift.py` (full analysis; `python player_drift.py "Name"` prints one player's mix by season), `out/player_drift_pairs.parquet`. Data: the 3,332 outfield player-seasons (500+ passes) of 4.7. "Peer-adjusted" means the season-and-role average (in square-root-share space) is subtracted first, which removes league-wide trends and role differences.

**League level, by role** (mean share, early 2013-14 to 2015-16 against late 2023-24 to 2025-26; changes beyond two standard errors):
- Defenders (317 -> 370 player-seasons): fwd-out-wide +3.7 points (5.7 -> 9.4), central link-up +2.6, wide back pass +1.3; wide down the line -3.0, long forward ball -3.3 (10.4 -> 7.1), cross -1.0.
- Midfielders (253 -> 257): wide short in the final third +2.9, wide back pass +1.2; central diagonal -1.3, forward out wide -1.3, long forward ball -1.4.
- Wide players (122 -> 79): wide short in the final third +7.2 (30.1 -> 37.3), wide back pass +1.7; forward out wide -3.8, central link-up -1.6.
- Forwards (78 -> 45): central link-up -10.5 (30.7 -> 20.2), forward out wide -4.9; wide short in the final third +7.5, wide back pass +3.4, cross +2.1.
Fewer long balls and more short build-up from defenders, and forwards and wide players doing more of their passing on the wing and less in the central link-up, are the headline changes. Caution: the role label is the position a player most often started in, and formations changed; the number of WIDE and FW labels roughly halves, so part of the role-level drift is relabelling, not behaviour.

**Individual drift.** Mean distance (square-root-share space) between a player's own mixes k seasons apart, with the sampling-noise scale for two full-season vectors at 0.073 and the distance between two random different players of the same role and season at 0.274:

| Seasons apart | Pairs | Raw distance | Peer-adjusted distance |
|---|---|---|---|
| 1 | 1,929 | 0.120 | 0.133 |
| 2 | 1,404 | 0.141 | 0.150 |
| 3 | 1,018 | 0.155 | 0.164 |
| 4 | 705 | 0.174 | 0.179 |
| 5 | 485 | 0.190 | 0.191 |
| 6 | 310 | 0.199 | 0.195 |

A player's mix is clearly more similar to his own earlier mix than to a random peer's (0.12 to 0.20 against 0.27), drifts away steadily with time, and the one-season distance (0.12) is well above sampling noise (0.073), so real change is present every year.

**What drives the change** (peer-adjusted one-season distance, mean over pairs):

| Group | Pairs | Mean | Median |
|---|---|---|---|
| Same club, same role | 1,573 | 0.116 | 0.103 |
| New club, same role | 152 | 0.150 | 0.132 |
| Same club, role label changed | 172 | 0.249 | 0.252 |
| New club, role label changed | 32 | 0.258 | 0.246 |

A club change adds about 0.03 (a third more than staying); a role change doubles the distance. Year-to-year correlation of the peer-adjusted shares is 0.63 to 0.87 by cluster (0.74 to 0.91 for the same club and role).

**Biggest shifts are real positional changes, verified against the fine position strings.** Eleven of the twelve biggest one-season shifts involve a change in the main WhoScored position of the player (Ben White CB -> right-back, Marcos Rojo left-back -> CB, Calum Chambers right-back -> CB, Dan Burn left-back/CB -> CB, Axel Tuanzebe right-back -> CB, Kristoffer Ajer right-back -> CB, Pascal Struijk CB -> left-back, Joe Gomez twice, Michail Antonio, Fabian Delph left-back -> midfield). The twelfth, Morgan Rogers, keeps attacking midfielder (AMC) as his main position; his role label only flips from WIDE to MID because his wide starts (AMR, AML and similar) together outnumbered his AMC starts in 2024-25 (AMC 41%) but not in 2025-26 (AMC 51%). Among defender-to-defender pairs, about 35% of the 100 biggest shifts switch between centre-back and full-back as main position, against 1% of the 100 smallest. So the pass mix detects a change between centre-back and full-back even where the coarse role label (DEF) says nothing changed. Example arcs (`player_drift.py`): Ben White: wide short passes 4% (CB, 2021-22) -> 26% to 29% (full-back, 2022-23 onward) and diagonal passes 22% -> 5%. Trent Alexander-Arnold: wide short passes 29% (2017-19) -> 13% (2023-24) and 18% (2024-25), forward-out-wide 2% -> 11% to 12%, central link-up 5% -> 9% to 10%, consistent with a more central role.

**Limits.** (1) The role label is the main starting position of the season, so mid-season changes blur. (2) Peer adjustment removes league-wide change but also removes any change shared by all players of a role. (3) Position labels depend on formations, which changed over the period. (4) Era differences in tagging (4.6) are handled only by the merged long-ball cluster; the clusters were fitted on all eras together (era-split agreement ARI 0.83). (5) Survivorship: only players with 500+ passes are included, so players with a short career or lost minutes do not appear. (6) The one-season distance includes injury seasons with fewer passes (more noise).

---

## 5. Discussion

1. **Risk ordering is as expected.** The clusters order sensibly by actual completion rate, which was never an input: the safest is the wide back pass (0.96) and the most dangerous are the lofted delivery and the long ball (0.47 and 0.31 in the first run). xP is a function of the same inputs as the clusters, so its agreement with the clusters is not an independent check; the completion rate is. See also `Project_QA_Review.md`.
2. **The structure is a continuum.** Silhouette is about 0.21 at every k and stability varies strongly with k. The nine clusters are a useful partition, not evidence of nine natural groups. Other k (4, 5) are equally reproducible and much coarser.
3. **Location dominates.** Because start position is an input, clusters mostly separate "where on the pitch and which way", and the rule types, which ignore position, spread over many clusters. This is by design but means the clustering is not a refinement of the rule types.
4. **Rule types that did not survive, and the flag-weight effect.** At weight 1, Switch and Cross merge in cluster 7 (lofted balls into or across the final third), partly because both carry the Chipped flag. The weight test (4.4) shows that this is an effect of the weight: at 2, crosses form their own cluster and the other lofted passes form another. Whether crosses and lofted forward balls should be separate archetypes is a football decision rather than a statistical one; weight 2 with k = 10 supports separating them, weight 1 with k = 9 does not.
5. **Goalkeeper passes sit in the long-ball cluster.** A player's share of cluster 8 will mix goalkeeper distribution with outfield long balls.

---

## 6. Limitations

- Silhouette about 0.21: the clusters are weakly separated and sensitive to the choice of k.
- The thresholds in the rule list, the choice of k = 9, the flag weight of 1 and the cluster names are judgement calls. The flag weight was varied only over 0.5, 1 and 2 with the same value for both flags; the flags were not weighted separately, and the chosen weight determines how many lofted clusters appear.
- No defender, pressure or ball-height data other than the Chipped flag. xP is the difficulty of the type of pass in a place, not of the moment, and it does not measure player skill.
- The Chipped flag's definition is inferred from behaviour. Its share is stable at about 9 to 10% for ten seasons and about 8% since 2024-25; a tagging change cannot be ruled out.
- The detailed xP validation (calibration bins, five-fold) covers one season (2024/25); the all-season xP (two-fold by game parity, seeded) was checked on AUC, Brier and mean-versus-actual completion by season, not on calibration bins.
- The incomplete 2026-27 season (80 matches to 25 October 2026) is included in the pooled data.
- Team style leaks into a player's pass mix; no adjustment has been made. WhoScored coordinate and tagging conventions may differ between eras; z-scoring uses pooled statistics, not per-season ones.
- K-means assumes roughly spherical clusters of similar spread; long balls (dx up to about +94 m) are a long tail.
- Quality review (2026-11-06, `Project_QA_Review.md`): the setting (k, flag weight) was chosen from a grid by stability; clusters fitted on 2013-14 to 2018-19 versus 2019-20 to 2026-27 agree at ARI 0.834 (control 0.967); the xP seed issue was fixed afterwards (see 3.2); team-season effect on player shares has not been measured within position.
- All results come from one sample of 300k passes for the final model (stability was checked over resamples but the final labels come from one fit).

---

## 7. Next steps (not done)

1. Setting decided: weight 2, k = 10 (section 4.4). Optional: test whether the k = 11 lofted diagonal / straight split separates different named players.
2. Done (4.6): per-season cluster shares are stable within 2.5 points; the Chipped tag looks unstable across eras (clusters 8 and 9). Decided: clusters 8 and 9 merged (9 clusters).
3. Decide on merging or splitting clusters (for example 1 and 2, and cluster 7).
4. Player-level step: per-player (or player-season) share of each cluster, minimum-pass threshold, comparison with position labels, similarity search, role drift across 13 seasons.
5. Optionally describe clusters by value (xT or VAEP per cluster) and separate goalkeeper passes.

---

## 8. Reproducibility

All files in `D:\CT's Portfolio\Practice\Player Roles\`:

| File | Purpose |
|---|---|
| `extract_pass_flags.py` | pull pass qualifiers from `raw.json` into `out/passflags_<season>.parquet` |
| `build_pass_table.py` | build `out/passes_all.parquet` (oriented coordinates, features, xP, rule type) |
| `pass_cluster_lib.py` | feature scaling and stratified sampling |
| `pass_cluster_sweep.py` | k = 4 to 20 sweep (silhouette, ARI stability); output `out/sweep_w1.csv` |
| `pass_cluster_describe.py` | fit k, assign all passes, describe clusters; labels in `out/labels_k9_w1.parquet` |
| `pass_weight_pca.py` | flag-weight comparison at k = 9 and PCA test (section 4.4 and 4.5) |
| `out/sweep_w05.csv`, `out/sweep_w2.csv` | k sweeps at flag weights 0.5 and 2 (weight 1 is `out/sweep_w1.csv`) |
| `out/labels_k10_w2.parquet` | cluster labels for weight 2, k = 10 |
| `Pass_Archetypes_Plan.md` | running plan and decisions |

Environment: `D:/CT's Portfolio/bzzoiro pipeline/scout/.venv-metrics` (pandas, scikit-learn). Random seeds: fit sample 0, evaluation sample 99, k-means random_state 0 (final model). The five-fold validation of xP on 2024/25 was run from scratch scripts that are not in this folder; the all-season xP is produced by `build_pass_table.py`.
