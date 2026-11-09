# What Kinds of Passes Do Premier League Players Play?
## Clustering 4.5 Million Passes to Describe Players, 2013 to 2026

*A student project report. Written in the first person, as a walk through how the project was actually made: what I tried, what went wrong, and what I would trust or not trust in the results.*

---

## Abstract

I wanted to answer a simple question: **what kinds of passes do Premier League players play, and can I describe a player by the mix of passes he plays?** I started from a chart I admired (a k-means clustering of defenders by the data site PITCH IQ) and tried to rebuild it, then improve it.

The improvement was to cluster the **passes** first, not the players. I took 4,526,215 open-play passes from 13 data files covering the 2013-14 season to 8 November 2026, described each pass with six numbers (where it starts, how far it travels toward goal, how far sideways, whether it was lofted, whether it was a cross), and ran k-means. After testing choices for the number of clusters, the weight of the two yes/no features, and whether PCA was needed, I settled on **nine pass types**, each with a fan-friendly name (The Recycle, The Link-up, The Wing Combo, The Diagonal, The Cross, The Out-Ball, The Push Wide, The Down-the-Line, The Long Ball).

Each player-season then became a **mix of those nine types**. I checked that mix for reliability (it is a stable property of a player within a season: split-half reliability 0.95 to 0.99), built a similar-player search, added a pass-value measure (xT) as a quality layer, and measured how players change across seasons. Finally I rebuilt the original idea at the player level: k-means on defenders and midfielders, coloured by group, with pass value as dot size, for 2025/26 and the current 2026/27 season.

The honest headline results are modest but real. Pass types form a continuum rather than sharp islands (silhouette about 0.21). Position explains about half of how players differ in their pass mix, but the other half is stable and not noise. The biggest change in the league over a decade is that goalkeepers stopped kicking long: the long forward ball fell from 53% to 33% of their open-play passes. The biggest weakness is that one of my inputs, a "Chipped" tag from the data provider, changed its behaviour partway through the period, and I had to design around that.

---

## 1. Introduction

### 1.1 Where the project came from

I follow the Premier League, I am studying toward a career in football analytics, and I learn best by building. When I found a blog post that clustered Premier League players with k-means (the PITCH IQ post on "Player Roles Clustering"), I decided to follow a rule I had set myself: **"to create, first imitate."** Rebuild what a good example does, understand each step, then change it so the result is mine.

That post used 20 per-90 statistics per player (passing, defending, creation, shooting), scaled them, squeezed them into two dimensions with PCA, ran k-means with seven clusters, and named the clusters by looking at them. The picture it produced was a scatter plot of players coloured by cluster, with names written next to the dots.

Reading it carefully, I noticed what it left unanswered: how many clusters were justified, whether the result would be the same on a different run, and whether the clusters were really "roles" or just reflected how much of the ball a team has. Those questions became the spine of my project.

### 1.2 The question

> **What kinds of passes do Premier League players play, and can a player be described by the share of each kind in his passing?**

Three parts, in order:

1. Find a vocabulary of pass types (the unit is one pass).
2. Describe each player-season as a vector of shares over those types.
3. Use the vectors: compare players, find similar players, follow change over time.

I agreed three success criteria before running anything: the pass types should be **stable** (same answer on different seeds and samples), **readable** (a coach could say what each is without looking at code), and **useful** (a player's mix should tell us something that position labels do not).

I also set what I was *not* claiming: that "true" pass types exist (they do not; the data are a continuum), that my names are anything but my reading of the numbers, or that pass mix measures how *good* a player is.

### 1.3 Why passes only

Midway through, I was tempted to add defending, carries and shooting so I could say something about overall quality. I decided against it. The project is about passing, and a narrow, well-checked claim is worth more than a broad, loose one. Pass value (xT) is included as an optional layer, with the limits stated plainly.

---

## 2. Background: the ideas I had to learn

I had no statistics background beyond the basics, so these are the concepts I worked through, in my own words.

**k-means.** An algorithm that groups points into *k* clusters by repeatedly assigning each point to the nearest cluster centre and moving the centre to the middle of its points. It needs me to say *k* in advance, it likes round, similar-sized groups, and it uses straight-line distance.

**Scaling (z-scores).** Because k-means uses distance, a feature measured in metres would swamp a yes/no feature. A z-score rewrites each value as "how many standard deviations from the average", which puts different features on one ruler. A rare yes/no flag (8% of passes) turns into a huge number after z-scoring, so I kept the two flags raw 0/1 and tested their weight.

**PCA (principal component analysis).** A way to rotate the features and keep only the directions that carry most variation. Keeping all components changes nothing for k-means; keeping only two throws information away. It is mainly good for drawing pictures.

**Silhouette score.** Measures how well each point fits its own cluster compared with the nearest other cluster (from -1 to 1). Values near 0.2 mean clusters overlap a lot.

**Stability (adjusted Rand index, ARI).** I fit k-means on different random samples and compare the groupings. ARI of 1.0 means identical groupings. This tests whether the algorithm gives the same answer twice; it does not prove the clusters are meaningful.

**Reliability (split-half).** Split a player's matches into two alternating halves, build his mix from each half, and correlate across players. High correlation means the mix is a real, repeatable property and not noise.

**xP (expected pass completion).** A model that predicts the probability that a pass completes, using only what is known when the ball is struck (where it starts and ends, its length, whether it was lofted or a cross). Risk = 1 minus xP.

**xT (expected threat).** A way to value a completed pass by how much it moves the ball into more dangerous areas. I used a 12 by 8 grid fitted on seasons before 2024-25.

---

## 3. Data

### 3.1 Source and preparation

The data are event-level files from WhoScored: every on-the-ball action in every Premier League match, converted to a standard format (SPADL) with start and end coordinates on a 105 by 68 metre pitch. Each season file holds about 610,000 actions. Two preparation steps mattered.

1. **Orientation.** Coordinates are absolute, so I flipped the away team (x becomes 105 minus x, y becomes 68 minus y) so every team attacks left to right. I checked this rather than trusting it: after flipping, crosses start at a median x of about 91 in both halves, home and away, and none start in their own half; shots show the same pattern. A flipping error would have shown up there.
2. **Open play only.** I kept passes and crosses and excluded throw-ins, goal kicks, corners and free kicks, which are already labelled by the provider and would only have rediscovered those labels.

The result is **4,526,215 open-play passes** from 14 season labels (2013-14 to 2025-26 complete, and 2026-27 so far: 10 matches per team, to 8 November 2026).

### 3.2 The qualifier flags

The standard format throws away some of the provider's own tags. In particular, it has no information about whether a pass was lofted. I went back to the raw event files and extracted six tags for every pass (Chipped, Longball, HeadPass, Throughball, KeyPass, Cross), joining them on game and event id. Over 99.5% of passes matched in every season; the unmatched 0.4% were dropped.

I only used two: **Chipped** (lofted) and **Cross**. Throughball was too rare (about 0.3% of passes), and Longball and LayOff changed their frequency across eras.

### 3.3 What "Chipped" means

I could not find the provider's definition (a web search returned nothing useful), so I characterised it from the data on 2024-25 (337,635 open-play passes, 8.0% chipped):

| Property | Chipped | Not chipped |
|---|---|---|
| Median length | 33.0 m | 14.2 m |
| Median forward gain | +19.2 m | +2.6 m |
| Completion rate | 46% | 86% |
| Also a header | 0% | 6.2% |
| Also tagged Longball | 51% | 6.2% |

At equal lengths, chipped passes complete far less often. I read Chipped as "a lofted pass". That is an inference from behaviour, not the provider's documentation.

---

## 4. Method, step by step

### 4.1 A rule-based yardstick (before any clustering)

Before using any algorithm I wrote down, from football knowledge, the passes I expected to find, with exact thresholds on a 105 by 68 pitch. For example: a *cross* is an event tagged as a cross; a *long ball* is 32 m or more long and gains 20 m or more toward goal; a *back pass* goes backward by 5 m or more. This gave nine rule-based types (cross, box entry, final-third combination, switch, long ball, progressive, back/recycle, short forward, sideways).

These rules were never used as the clustering. They were a **ruler**: later, I could ask how well the k-means clusters lined up with a hand-made list. They also taught me two things early. Cutbacks were only 0.1% of passes (242 in 2024-25), so I merged them into the final-third combination, and through balls were too rare to separate.

### 4.2 Risk as a number: xP, and how I validated it

I wanted "risk" to be a number, not a word. I defined it as 1 minus the expected completion probability and built a gradient-boosted classifier using only information available at the moment of the pass: start and end position, length, direction, header, chipped, cross. It never sees the result.

I validated it on 2024-25 with folds split **by match**, so a pass is never scored by a model that trained on its own match:

| Model | AUC | Brier (lower is better) |
|---|---|---|
| Always predict the average (82.8%) | 0.500 | 0.1405 |
| Geometry only | 0.916 | 0.0767 |
| Geometry + chipped | 0.920 | 0.0754 |
| Geometry + chipped + cross | 0.920 | 0.0751 |

Calibration (predicted against actual completion in seven probability bins) agreed to within one point in every bin. Before I added the cross flag the model predicted 26.0% completion for crosses against an actual 21.6%; with the flag it predicted 21.4%.

Limits I wrote down: xP knows nothing about defenders or pressure, so it measures how hard a *type* of pass is, not how hard it was in that moment, and it says nothing about player skill.

### 4.3 What I cluster on

After removing redundancy, each pass was described by six numbers.

| Feature | Meaning | Scaling |
|---|---|---|
| start_x | where the pass starts, toward goal | z-score |
| start_w | start distance from the centre line | z-score |
| dx | metres gained toward goal | z-score |
| dy_folded | sideways movement, with the starting side flipped so left and right are treated alike | z-score |
| chipped | lofted flag | raw 0/1, weight tested |
| cross | cross flag | raw 0/1, weight tested |

Reasons: folding left and right removes fake duplicate clusters; end position is start plus displacement, so including it would double-count; length, angle and gain are functions of dx and dy; and xP is deliberately **not** an input. An xP input would repeat the same information, and keeping it out lets me use it afterwards as a check.

### 4.4 Choosing the number of clusters and the flag weight

I fit k-means on a 300,000-pass sample (equal numbers per season) and scored every k from 4 to 20 with silhouette and with a stability test (two fits on different disjoint 100,000-pass samples, compared by ARI).

The first pattern was sobering. **Silhouette was about 0.21 for every k from 5 to 13.** Passes do not form separate islands; they form a smooth continuum, and any partition is a useful slicing rather than a discovery. Stability, on the other hand, jumped around: at weight 1, k = 4, 5 and 9 reproduced (ARI 0.95 or more) but k = 7, 11 and 12 did not (0.55 to 0.62).

Because the yes/no flags are an arbitrary choice, I tested their weight at 0.5, 1, 1.5, 2 and 3:

| Flag weight | Stability at k = 9 | What changed |
|---|---|---|
| 0.5 | 0.92 | almost the same clusters as weight 1 (ARI 0.95) |
| 1 | 0.95 | one lofted cluster (crosses mixed with lofted switches) |
| 2 | 0.68 at k = 9, **0.96 at k = 10** | crosses separate from lofted forward balls |
| 3 | 0.80 at k = 10, **0.96 at k = 11** | the lofted group splits further |

I picked **weight 2 with k = 10**, because its stable cell agreed with the stable cell at weight 3 (the two partitions matched at ARI 0.937 over all 4.5 million passes, and the cross cluster was the same). I was careful to write down that this was a choice made from a grid by looking at stability, and not an estimated quantity.

### 4.5 Do I need PCA?

I asked this directly, because the source post used it. I tested it:

- Keeping all six components reproduces the clustering exactly (ARI 1.000), because PCA only rotates.
- Keeping two components (61% of variance) gave ARI 0.27 against the unreduced clustering; three (82%) gave 0.49.
- Four components (97.8%) gave 0.96, but that number hides that the last two components are the two flags, which means the lofted information disappears.

So **PCA is not needed for the clustering.** I only use it later to draw two-dimensional maps.

### 4.6 The tagging problem, and merging two clusters

When I checked how cluster shares changed over seasons, they were stable to within about 2.5 percentage points, with one exception: the "lofted forward ball" cluster dropped suddenly in 2023-24, in the same year the Chipped share fell from 10.2% to 8.6%. That looked like a change in how passes were tagged, not in how teams play. To test it I took passes defined by geometry alone (30 m or more long, 15 m or more gained, not a cross):

| Seasons | Share of those passes tagged Chipped |
|---|---|
| 2013-14 to 2016-17 | 34% to 35% |
| 2022-23 | 58% |
| 2024-25 onward | 42% to 44% |

Within one season the tag was stable (2022-23 and 2024-25 flat), but **2023-24 stepped down mid-season**: the monthly tagged share went from 0.59 to 0.66 in August to October 2023 to 0.40 to 0.49 from December, while the number of long forward passes moved only slowly. A third of long forward passes do not stop being lofted in the middle of a season. I judged this a tagging change, though I cannot prove it.

My decision: **merge the lofted-forward cluster and the long-ball-from-deep cluster** into one "long forward ball" cluster, which follows the geometry and does not depend on the unreliable tag. That left **nine clusters**. Crosses stay separate because they come from the event type, not the tag.

### 4.7 The nine pass types

Each has a plain-English name and a one-line description:

| # | Name | Description | Share of passes | Completion |
|---|---|---|---|---|
| 0 | The Recycle | wide back pass, inward | 10.5% | 96% |
| 1 | The Link-up | short pass through the middle, attacking half | 11.8% | 83% |
| 2 | The Wing Combo | short pass on the wing, final third | 17.4% | 84% |
| 3 | The Diagonal | central pass across the pitch, from deep | 11.4% | 94% |
| 4 | The Cross | delivery into the box | 3.0% | 23% |
| 5 | The Out-Ball | central ball out to the flank | 12.3% | 89% |
| 6 | The Push Wide | forward ball out wide, attacking half | 11.9% | 87% |
| 7 | The Down-the-Line | wide pass up the touchline | 13.2% | 71% |
| 8 | The Long Ball | long ball forward, often from deep | 8.6% | 41% |

(Shares and completion are over all 14 seasons; completion is the actual rate, which was never an input.) The risk gradient is clear and unsurprising: the safe passes are short and backward; the dangerous ones are long or lofted.

### 4.8 A quality review of my own work

Before building on these results I stopped and audited the project the way a reviewer would. Using a data-validation checklist, I found and fixed or documented these:

1. **A circular check.** I had said the clusters "order sensibly by xP". But xP is a function of the same inputs as the clusters, so that agreement is not independent evidence. The independent evidence is the **actual completion rate**, which was never an input and runs from 96% to 23% across clusters. I corrected the wording.
2. **A model that could not be rebuilt identically.** The xP model had no random seed, and with this much data the library silently holds out a random 10% for early stopping. I added `random_state=0`, rebuilt the table twice, and got identical values. The seed changed xP only slightly (mean absolute difference 0.0095; per-cluster means moved by at most 0.001).
3. **Researcher degrees of freedom.** I had searched a grid of k and weights and kept a stable cell. To test that I did a **temporal holdout**: fit clusters only on 2013-14 to 2018-19 and separately on 2019-20 to 2026-27, and compare. They agreed at ARI 0.834 (against 0.967 for two random samples of the same size). So the structure carries across eras, with about 0.13 of agreement lost to real change and tagging.
4. **Validation scripts living only in a temporary folder.** I copied them into the project.

### 4.9 From passes to players

For each player-season with at least 500 open-play passes (3,555 of them), I computed the share of passes in each of the nine types. Then I tested whether that mix means anything:

- **Reliability.** Splitting matches into alternating halves, the correlation between halves gave full-season reliability of 0.95 to 0.99 for all nine types. A player's mix is a stable property within a season.
- **Position.** The player's main starting position explains about 0.53 of the variance in the share vectors (from 0.26 for crosses to 0.75 for central link-up). What is left within position is still highly reliable (0.88 to 0.97), so roughly half the differences between players are not position and are not noise.
- **Team.** Within position, the team-season explains about 0.15 to 0.20 of the long ball, down-the-line and link-up shares but almost nothing for four other types. Team style leaks into some shares, not all.
- **Goalkeepers.** Their mix is 42.5% long forward ball, 32.5% out-ball and 24.0% diagonal, with nothing elsewhere. They make 18% of all long-forward-ball passes and need separate treatment.
- **A sanity read.** The top three players per type in 2024-25 looked like football: Salah, Semenyo and Nunes for The Wing Combo; Robinson, Mitchell and Mykolenko (full-backs) for The Cross; Pickford, Sánchez and Flekken (goalkeepers) for The Long Ball. I read these by eye; they were not a blind test.

One caveat I wrote on the first page of my notes: reliability shows the mix is stable, not that it is skill. Footedness, side of the pitch, team and era all feed in.

### 4.10 A similar-player search

I built a tool to list the player-seasons with the closest mix. I tested four distance measures with two questions: given half of a player's matches, how often is the other half of the same player the nearest of all 3,332 outfield player-seasons, and how often is the nearest player in a different season the same player?

| Distance | Own other half is nearest | Median rank of own other half | Same player, other season, nearest |
|---|---|---|---|
| Raw Euclidean | 10.7% | 18 | 13.0% |
| **Hellinger (square roots of shares)** | **12.3%** | **14** | **15.6%** |
| z-scored shares | 10.8% | 16 | 13.5% |

Hellinger won. The honest reading is that **the tool gives a shortlist, not a twin**: the true match is the very nearest only one time in eight, though its median rank is 14 out of 3,332 (the top 0.4%, against a chance rank of about 1,666). With nine shares the space is crowded, especially for central midfielders. I also calibrated a noise scale: distances below 0.073 are indistinguishable from sampling noise, and the tool flags them.

Examples (2024-25, same-season pool): Declan Rice's closest profiles were Mateus Fernandes, Enzo Fernández, James Maddison and Alexis Mac Allister; Trent Alexander-Arnold's were Pedro Porro, Andy Robertson and James Justin.

### 4.11 Adding value: xT

A profile says what someone does, not how much it adds. I added **pass xT** (the threat gained by completed passes), per 90 minutes. I chose xT over VAEP because my own earlier trust test had shown it repeats far better: split-half reliability at 20 matches was 0.87 for xT against 0.39 for VAEP per 90, and season-to-season correlation 0.78 against 0.37.

- The grid was fitted only on seasons before 2024-25, so the values are out of sample for 2024-25 and 2025-26 (496 outfield player-seasons with 900 or more minutes and 500 or more passes).
- Crosses are only 3% of passes but produce **37% of all pass xT**; long forward balls are 7.6% of passes and 20% of pass xT; the wide back pass is slightly negative. Risk and value go together: the high-value types are the low-completion types.
- The player's mix alone explains **0.69** of the variance in xT per pass between player-seasons.

The limits are important. Pass xT ignores carrying, ball winning and positioning, so a low pass xT does not mean a poor player. Declan Rice, valued for his all-round game, sits below the midfield average on this measure. I decided, with my project goal in mind, that the project should stay about passing and that more value layers would be optional.

### 4.12 Role drift

I measured how much a player's own mix changes over time, using a distance where 0.073 is sampling noise and 0.274 is the gap between two random players of the same role and season:

| Seasons apart | Own distance |
|---|---|
| 1 | 0.12 |
| 3 | 0.16 |
| 6 | 0.20 |

A player is much closer to his earlier self than to a random peer, but he drifts steadily, and even one season of change is well above noise. A **role change roughly doubles** the one-season drift (about 0.25 against 0.12); a **club change adds about a third** (0.15).

The nicest result: I looked at the 12 biggest one-season shifts and compared them with the finer position labels in the raw data. **Eleven of the twelve** involve a real change of main position: Ben White from centre-back to right-back, Marcos Rojo from left-back to centre-back, Calum Chambers from right-back to centre-back, Dan Burn, Axel Tuanzebe, Kristoffer Ajer, Pascal Struijk, Joe Gomez (twice), Michail Antonio and Fabian Delph. (My first version of this sentence claimed all twelve; I checked and Morgan Rogers keeps the same main position, and I corrected it.) Among defender-to-defender pairs, about 35% of the 100 biggest shifts switch between centre-back and full-back, against 1% of the 100 smallest. The pass mix detects a centre-back or full-back switch even where the coarse label "defender" says nothing changed.

### 4.13 What changed in the league

Comparing 2013-17 with 2022-26 (weighted by passes), by position group:

| Position group | Biggest changes (percentage points) |
|---|---|
| Goalkeepers | long ball -19 (53% to 33%); out-ball +13 (26% to 39%); diagonal +6 |
| Centre-backs | push wide +3, link-up +2; down-the-line -3, long ball -3 |
| Full-backs | push wide +3, link-up +2; wing combo -3, long ball -2 |
| Midfielders | nothing beyond about 2 points |
| Wide players and forwards | wing combo +8, recycle +3; link-up -6, push wide -5 |

The headline is the goalkeepers, with a caveat I put on the chart: goal kicks are excluded, so this is about open-play distribution. The attacker finding is the least reliable, because formations changed and position labels shifted with them (the number of wide and forward player-seasons fell from 254 to 166).

---

## 5. Visualisation: four rounds and what they taught me

I wanted something postable on social media, not an interactive page (a published artifact would not work for that) and not a dense scatter. I went through several rounds and learned from each.

**Round 1.** I made four prototypes. They failed in different ways, and the feedback was blunt and right:

- *Nine pooled passes.* "Cool to look at, but it doesn't give me anything insightful." It showed no surprises; it only illustrated the clusters.
- *Journey map.* Three players' paths across a map of 3,332 dots. "Way too many map points." Too much visual noise to read a path.
- *Position drift contours.* No axes, labels or legend. "No average fan or football executive would know what this is talking about." And I had made a real error: for defenders my arrow pointed into the empty gap between the centre-back and full-back clouds, so it described an average player who did not exist.
- *League map.* Good as an overview, but too many dots for a static image; it made more sense as an interactive explorer.

**Round 2** fixed this with three ideas:

1. A **season-by-season atlas** with a name for each type and the players who play it most, plus a goalkeeper note (The Long Ball: 35% of goalkeepers' passes in 2025/26, 53% in 2013/14).
2. A **colour-coded table** for the decade change, with plain-language rows and columns and a "read it" line.
3. **Player "fingerprint" bars** (a stacked bar per season) instead of a cloud of dots. Ben White's switch is readable at a glance: The Wing Combo 4% to 26% and The Out-Ball 22% to 3% between 2021/22 and 2022/23.

I also caught my own mistakes while reviewing images. In the atlas, titles overlapped the descriptions. In the decade table, my "read it" line said 34% while the table said 33%; I changed the text to compute from the data.

**Round 3: the original idea, upgraded.** I went back to the reference chart I had set out to imitate: k-means on **players**, one position group and one season, PCA scatter, coloured clusters, names. My earlier pooled maps did not match what I had meant by "rooted in k-means". So I rebuilt it as an upgrade. The extra layer is that each player is described by his nine-pass mix and his pass xT is the dot size.

---

## 6. Player-level k-means: defenders and midfielders

This step repeats the original post's method, with the validation the original lacked.

### 6.1 Method

For each position group and season: take each player's nine shares (square-root scaled), run k-means, and choose *k* by silhouette **and** bootstrap stability (30 pairs of 80% resamples). Then draw a PCA scatter, colour by cluster, label the most-used and highest-value players, and size dots by pass xT per 90 (a fixed scale, so charts are comparable).

### 6.2 Defenders, 2025/26 (125 defenders with 500 or more passes)

| k | Silhouette | Stability (ARI) |
|---|---|---|
| 2 | 0.61 | 0.97 |
| 3 | 0.43 | 0.93 |
| **4** | **0.30** | **0.83** |
| 5 | 0.23 | 0.81 |
| 7 | 0.20 | 0.57 |

k = 2 is the clearest, but it only recovers centre-backs against full-backs (the same point as the position effect). k = 4 is the largest number I trusted, and it adds a subtype inside each. The four groups, named by me from their top pass types: **Overlapping full-backs** (27 players; wing combo 30%, down-the-line 22%, cross 7%; pass xT per 90 8.1), **Balanced full-backs** (28; 6.5), **Distributor centre-backs** (52; diagonal 23%, out-ball 22%; 4.5), and **Progressive defenders** (18; recycle 16%, push wide 12%, link-up 8%; 5.2; it includes Guéhi, Richards, Khusanov, Burn, Gvardiol and Reece James).

The map is mostly one dimension: the horizontal axis carries 85% of the differences (centre-backs on the left, full-backs on the right) and the vertical only 6%. While checking the picture I found that I had written the vertical-axis labels backwards (the top means more long balls and down-the-line passes, not more link-ups); I confirmed it against the PCA loadings and against Senesi, a long-ball specialist, sitting at the top, and I fixed the labels.

### 6.3 The current season, 2026/27

The season is only 8 matches per team old (to 25 October 2026). With the usual threshold of 500 passes only seven defenders and two midfielders qualify, which is too few to cluster. I lowered the threshold to **200 passes** (and 360 minutes for xT) and checked what that costs: the split-half reliability of the shares is 0.89 overall but only about **0.78 within defenders and 0.75 within midfielders**, lower than a full season (0.95 or more). I put a caveat on the charts instead of hiding this.

- **Defenders (69 players):** stability is again highest at k = 2 (0.99), then k = 4 (0.83, silhouette 0.28) and k = 3 (0.79). I used k = 4 for consistency with last season: **Overlapping full-backs** (23; pass xT per 90 7.7), **Progressive defenders** (10; 6.4), **Distributor centre-backs** (22; 5.3) and **Long-ball centre-backs** (14; 5.3). The split among centre-backs this season is between those who play more diagonals and out-balls and those who play more long balls.
- **Midfielders (51 players):** k = 2 is 0.96 but only separates attacking from deep midfielders; k = 3 is stable (0.87) and k = 4 is not (0.67), so I used three: **Attacking creators** (16; wing combo 21%; xT 6.7; Ødegaard, Fernandes, Wirtz, Cherki, Szoboszlai), **Progressive holders** (18; push wide 21%; 4.6; Tonali, Mainoo, Kamara, Mac Allister, Anderson) and **Deep distributors** (17; out-ball 15%, recycle 10%; 5.9; Rice, Xhaka, Tielemans, Palacios).

Interpretation: with defenders, k-means finds a few stable groups that mostly track position, with a smaller but real layer of style inside each. With midfielders, the layer is thinner: stability falls quickly after three groups. I did not do attackers because the sample of players per season is smaller.

---

## 7. Results in brief

1. Open-play passes fall into **nine types** that are reproducible but form a continuum (silhouette about 0.21).
2. **Risk and value go together:** crosses are 3% of passes and 37% of pass threat; completion runs from 96% (The Recycle) to 23% (The Cross).
3. A player-season's **mix is a stable trait** (reliability 0.95 to 0.99); about half of how players differ is position, and the other half is reliable and not position.
4. **Position changes show up in the passing:** eleven of the twelve biggest one-season shifts involve a real change of main position.
5. The league changed: **goalkeepers' long forward balls fell from 53% to 33%** of their open-play passes in a decade; attackers moved toward wing combinations; midfielders barely changed.
6. A similar-player search works as a **shortlist** (median rank 14 of 3,332), not a twin finder.
7. Player-level k-means gives **three or four stable groups per position**, not seven.

---

## 8. Mistakes, corrections and what they taught me

| What went wrong | How I found it | What I did |
|---|---|---|
| I called the agreement between clusters and xP independent evidence | Quality review: xP uses the same inputs as the clusters | Reworded; used the actual completion rate as the independent check |
| xP changed slightly on every rebuild | Reading the library behaviour during the review | Seeded the model; rebuilt twice with identical output |
| I claimed all 12 biggest shifts were position changes | Checking against the fine position labels | Corrected to eleven of twelve and explained the exception |
| The Chipped tag changed behaviour mid-2023-24 | Seeing the lofted cluster drop abruptly | Tested with geometry-only passes; merged two clusters |
| The position-drift picture was misleading and unreadable | User feedback and re-reading the chart | Rebuilt as a plain colour table with split positions |
| Vertical axis labels reversed on a player map | Checking labels against loadings and known players | Fixed before sending |
| A sentence on a chart disagreed with the table (34% vs 33%) | Reading the finished image | Made the text compute from the data |
| Titles and subtitles overlapped in the pass atlas; the goalkeeper note ran into the footer | Reading the images | Re-laid out the text with proper spacing |

The pattern: **my mistakes were mostly in wording and presentation, and the way I found them was reading every output against the source numbers.** I now do that before sending anything.

---

## 9. Limitations

- **Continuum, not islands.** Silhouette about 0.21; the nine types are a useful partition. Names are my reading.
- **Choices made from a grid.** The number of clusters and the flag weight were chosen by stability from a grid; the temporal holdout (ARI 0.834) suggests the structure carries across eras but is not identical.
- **Tagging.** The Chipped tag is the weakest input. A change inside 2023-24 forced a merge, and the lofted/long-ball split is only trustworthy from 2024-25 onward.
- **No defenders, pressure or ball height** (other than the tag): xP measures the difficulty of a type of pass, not of the moment, and is not a measure of skill.
- **Pass xT is only passing.** It ignores carries, defending and off-ball work.
- **Team and position leak in.** A player's mix reflects his team's style and his role as well as his habits; the player-level maps mostly redraw position.
- **Small samples.** Quality exists for two seasons (496 player-seasons); 2026-27 is 8 matches per team and shares are less reliable (about 0.75 to 0.78 within position).
- **Position labels** are the position a player most often started in and shift with formations; goalkeepers and goal kicks are handled separately or excluded.
- **Names and groups are my judgement**, and k-means assumes roughly round clusters.

---

## 10. Reflection

What I learned about the method: **choosing the number of clusters is the main decision, and it deserves evidence.** The blog post I started from chose seven clusters from an elbow plot, with no stability check; when I tested the choice myself, defenders supported only two to four. I also learned that a clustering result needs a reality check that is *independent* of its inputs (here, actual completion rate, not xP).

What I learned about football data: event data are a record of what the provider chose to tag, and those choices change. A feature that looks like football (lofted or not) can hide a change in the camera, not in the game. That is why I kept testing whether a change was play or measurement.

What I learned about communication: a clever picture that needs a paragraph to explain is a failed picture. The best images answered one question in plain words (goalkeepers stopped kicking it long) with labelled axes and a "read it" line. I also learned to cut text from the bottom of charts.

What I would do next: (1) test whether the three or four stable player groups per position hold for attackers and for other seasons; (2) add uncertainty intervals for each player's shares and xT; (3) repeat the work for another league to see whether the nine pass types are specific to the Premier League; (4) compare what the clusters say with what scouts and coaches would call the same players.

---

## 11. Tools, assistance and reproducibility

**Assistance disclosure.** I built this project with an AI coding assistant (Claude Code) working under my direction: it wrote and ran much of the code, ran the checks I asked for, and drafted the figures and text, and I made the decisions about what to build, accepted or rejected each step, and reviewed the outputs and charts. If your course or workplace requires specific wording for AI assistance, adjust this section accordingly.

**Environment.** Python with pandas, scikit-learn (k-means, PCA, silhouette, gradient-boosted classifier), scipy, and the socceraction library; plots in matplotlib (cream background, Georgia type). Random seeds are fixed (k-means 0; the xP model 0; the sample seeds 0 and 99).

**Folder `Practice/Player Roles/`.**

| File | Purpose |
|---|---|
| `extract_pass_flags.py` | pull pass tags from the raw event files |
| `build_pass_table.py` | build the 4.5-million-pass table (coordinates, features, xP, rule type) |
| `pass_cluster_lib.py`, `pass_cluster_sweep.py`, `pass_cluster_describe.py` | feature scaling, the k sweep, cluster profiles |
| `pass_weight_pca.py` | flag-weight comparison and the PCA test |
| `build_player_table.py`, `player_shares.py` | player-season table and reliability, position, team and goalkeeper checks |
| `similarity_eval.py`, `similar_players.py` | metric test and the search tool |
| `pass_value.py`, `partial_season_players.py` | xT value layer; threshold-adjusted table for the current season |
| `player_drift.py` | role drift analysis (and one player's season-by-season mix) |
| `player_kmeans.py`, `viz_player_clusters.py` | player-level k-means and its charts |
| `viz_prep2.py`, `viz_draw2.py`, `viz_draw.py`, `build_interactive.py` | atlas, fingerprints, decade table, first-round figures, interactive map |
| `Pass_Archetypes_Report.md`, `Project_QA_Review.md`, `Pass_Archetypes_Plan.md` | the technical report, the quality review, the running plan |
| `validation/` | scripts behind the xP and Chipped numbers |

Everything is reproducible by running the scripts in this order: flags, pass table, sweep and describe, player tables, then the figure scripts.

---

## Appendix A. Glossary

- **Open play**: passes that are not throw-ins, goal kicks, corners or free kicks.
- **Silhouette / ARI**: cluster-separation and cluster-stability measures (sections 2 and 4.4).
- **xP / risk**: expected completion probability and 1 minus that.
- **xT**: expected threat, the value a completed pass adds by moving the ball to more dangerous cells.
- **Hellinger distance**: the straight-line distance between square roots of shares; it treats small shares more fairly than raw shares.
- **Split-half reliability**: how well two alternating halves of a player's matches agree.
- **Fine position**: WhoScored's detailed start-of-match position (for example DC, DR, DL), used to split defenders into centre-backs and full-backs.

## Appendix B. Key parameters

| Setting | Value |
|---|---|
| Passes | 4,526,215 open-play passes, 14 season labels |
| Clustering sample | 300,000 passes, equal per season |
| Features | start_x, start_w, dx, dy_folded (z-scored); chipped, cross (raw 0/1, weight 2) |
| Clusters (final) | 10 fitted, 2 merged into 9 |
| xP model | gradient-boosted trees, 200 iterations, seed 0, folds by match |
| Min passes for player tables | Defenders 150, midfielders 250, attackers 250 (full seasons; see the update at the end); 200 (2026-27 so far) |
| xT grid | 12 by 8, fitted on seasons before 2024-25 |
| Player k-means | square-root shares; k by silhouette and 30 bootstrap pairs of 80% resamples |

## Update (7 Nov 2026): a cutoff and a number of groups for each position

**Why.** I felt four groups were too few to show all the archetypes, and that attackers were being cut too hard: they pass about 27 times per 90 minutes against 40 (defenders) and 43 (midfielders), so a flat 500-pass cutoff kept 75% of defenders and midfielders but only 49% of attackers.

**Cutoffs.** I rebuilt the player table with a floor of 150 passes and measured split-half reliability (odd against even matches, Spearman-Brown corrected) in 100-pass windows within each position. A strict bar of 0.80 for every window gives 200 (defenders), 350 (midfielders) and 350 (attackers). I then lowered it, because that threw away many players: the final cutoffs are **150 passes for defenders, 250 for midfielders, 250 for attackers**. That keeps 2,026 defender, 1,367 midfielder and 1,175 attacker player-seasons (a flat 500 kept 1,517, 1,107, 699). The price: the 100 passes just above each cutoff have reliability of about 0.76 (defenders), 0.71 (midfielders) and 0.77 (attackers), and overall reliability is 0.85 to 0.91 (0.89 to 0.94 at 500). Midfielders need more passes than defenders because their mixes are less distinct.

**Number of groups.** k from 2 to 12 per position, with five checks (silhouette, bootstrap ARI, odd-vs-even-match agreement, early-vs-late-era agreement, smallest group). Stability drops fast after k = 3 or 4 everywhere, and the cutoff changes it only by a few hundredths (the exception: attackers' early-vs-late-era agreement for types fell from 0.53 to 0.43 when the cutoff went from 350 to 250). At k = 6 or 7 the same player lands in the same group only about half the time when his odd and even matches are clustered separately (a pessimistic test, since each half has half the passes).

| | Types (stable) | Bootstrap ARI | Odd/even ARI | Era ARI | Styles (finer) | Bootstrap ARI | Odd/even ARI | Era ARI |
|---|---|---|---|---|---|---|---|---|
| Defenders | k = 4 | 0.93 | 0.68 | 0.67 | k = 7 | 0.89 | 0.50 | 0.50 |
| Midfielders | k = 3 | 0.96 | 0.54 | 0.62 | k = 5 | 0.83 | 0.45 | 0.31 |
| Attackers | k = 4 | 0.90 | 0.49 | 0.43 | k = 6 | 0.87 | 0.42 | 0.52 |

**Decision.** Two levels. **Types** are the largest clearly stable k and carry the conclusions. **Styles** are local stability peaks: more interesting (for defenders: overlapping, crossing and inverted full-backs; diagonal, ball-progressing, direct and wide centre-backs; for attackers: crossers, wing-combination wingers, inside forwards, roaming creators, link-up strikers) but with wobbly borders, so they are labelled fuzzy. The page has a switch.

**A negative result.** There is no "recycler" midfielder. The Recycle share is 7 to 10% in every midfield group, and clustering midfielders on average completion probability, share of backward or sideways passes and average pass length gives one smooth line from safe to forward (silhouette 0.33 at k = 2, no better beyond). Finding recyclers would need features built around risk, not around where passes go.

## Update (7 Nov 2026): a "Plays like" list on the page

The similar-player search from the notebook (Part 7) is now on the page. Selecting a player-season lists the **eight** players with the closest pass mix, one row per player at his closest season, so a player can be "similar to A in 2018/19 and B in 2021/22". Distance is Hellinger (Euclidean on the square roots of the nine shares) across all positions, with options for same season only and same position group only. Each row shows the two pass types in which the match differs most (in percentage points) and flags matches within the **noise scale of 0.082** (the typical distance between a player's own two half-seasons, rescaled to two full seasons). Clicking a name jumps to that player-season.

Honest limits: it is a shortlist, not a twin (at 500+ passes the player's own other half-season was the nearest only about one time in eight, median rank about 14 of 3,300), and the lower pass cutoffs add noisier players to the pool. Some players have no close match, for example Alexander-Arnold in 2024/25 (nearest 0.101, outside the noise scale). The notebook has a Python version of the same list, on exactly the players the page shows, and the two agree to within the 0.1-point rounding of the shares stored in the page.

## Update (7 Nov 2026): carrying

Does carrying add to the picture? Tested before building (notebook Part 12). Carries are the SPADL `dribble` action, inferred from event data; take-ons are attempts to beat an opponent (success rate is not used: it drifts from 0.5 to 0.4 across eras).

- Four per-90 measures (carries, progressive carries, take-ons, carry xT) are reliable (0.76 to 0.96 odd vs even matches) and mostly not in the pass mix (CV R-squared 0.2 to 0.46). Where a carry starts is mostly in the pass mix (0.6 to 0.97), so zone shares are dropped.
- Adding carries to the k-means does **not** give more stable groups; for finer groups it is worse (bootstrap stability for midfielder styles 0.86 to 0.64, attacker styles 0.88 to 0.60). Clustering the carries themselves into types adds almost nothing (+1.3 points of player identification over the per-90 measures).
- Carries do tell players apart: a player's own other half-season is the nearest match 9% of the time with passes alone and 17% with carries added (weight 0.25).
- A three-level carry tier (thirds of a composite within position) gives the same tier on odd and even matches 70% to 79% of the time; take-ons per carry (dribbler index) has reliability 0.84 to 0.89.
- Page: carry block on the card, carry-level filter, dot size by pass or carry xT, and an "include carrying" switch in Plays like. Limits: inferred carries, no speed or pressure; carry xT before 2024/25 uses a grid fitted on the same data; 2026/27 is 8 matches old.

## Update (9 Nov 2026): new matches

Twenty more 2026/27 matches (to 8 November 2026) were added: 100 matches, 10 per team, 4,526,215 open-play passes in total. The whole notebook was rerun from the raw files. The nine pass types barely moved (shares within 0.2 points, completion within 0.002); player types and styles are unchanged for 96% (defenders) to 99.7% (midfielders) of the existing player-seasons; 187 players now have 200+ passes in 2026/27 (146 before), with split-half reliability of the nine shares 0.81 for defenders and 0.73 for midfielders (0.78 and 0.75 at 8 matches). Sections above that quote figures from the 25 October build (for example the 8-match counts) are left as written.
