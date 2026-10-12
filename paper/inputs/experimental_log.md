# Experimental Log

## 1. Experimental Setup

* **Datasets:** Two self-collected monocular traffic datasets. `videoset_01`
  contains 15 clips plus 5 still scenes (1080p, mixed 30/60 fps, five clips with
  variable frame rate, portrait and landscape). `videoset_02` contains 14
  1080p clips (13371 frames, approximately 6.2 minutes) of urban intersections.
  A 277-frame subset of `videoset_01` was labelled frame by frame by a human
  annotator, giving 3612 boxes; this is the only hand-verified ground truth.
  A 332-frame subset of `videoset_02` was auto-labelled and re-labelled to a
  10-class schema. A public aerial dataset (VisDrone) was converted to a
  tricycle subset of 2661 frames with 9635 boxes (2262 train / 399 validation)
  and used only for out-of-domain generalisation checks.
* **Evaluation Metrics:** Detection is reported as mAP50 at input resolution
  1280. Prediction is reported as the median, mean and 90th percentile of the
  Euclidean distance in pixels between the predicted box centre and the
  observed centre at the target horizon, together with the fraction of targets
  that are predictable at all. Trajectory stability is reported as identity
  jumps per thousand frames. The canonical prediction measurement uses six
  clips x 300 frames, detection confidence 0.25, input resolution 1280, and the
  static-structure channel disabled so that only the learned detector and the
  motion model contribute.
* **Baselines Compared:** Detection: chained fine-tuning of a nano backbone
  against from-scratch nano and small backbones, and a weighted-box fusion of
  the two best detectors. Prediction: constant velocity, constant acceleration,
  constant turn rate and velocity, interacting multiple models, NIS-adaptive
  Kalman filtering, ground-plane (bird's-eye-view) modelling with a
  data-driven horizon, tiled inference with standard post-processing, input
  resolutions 960/1280/1600, and four alternative trackers (BoT-SORT with
  ReID, OC-SORT, ByteTrack at lower confidence, ByteTrack with a long
  association buffer).
* **Implementation Details:** All training and inference ran on one laptop GPU
  with 8 GB of memory (compute capability 12.0). Prediction uses resolution
  1280, confidence 0.25, NMS IOU 0.5, and ByteTrack association. Eleven
  detector versions were trained in total; the deployed default is the 7-class
  chained model. Cumulative training time across all versions was 438878
  seconds (approximately 122 GPU-hours). Inference costs 17-24 ms per frame for
  detection at resolution 1280 and 0.24 s per frame end to end including
  association, extrapolation and rendering.

## 2. Raw Numeric Data

Detection accuracy at input resolution 1280 on three validation sets:

| Validation set | Chained 7-class | From-scratch nano | From-scratch small |
| --- | --- | --- | --- |
| Target domain (65 frames) | 0.637 | 0.615 | 0.597 |
| Old street scenes (116 frames) | 0.701 | 0.636 | 0.698 |
| Aerial (399 frames) | 0.407 | 0.310 | 0.408 |

Precision and recall of the two best detectors on the target domain:

| Model | Precision | Recall | Predictable targets per frame |
| --- | --- | --- | --- |
| Chained 7-class | 0.604 | 0.620 | 17.01 |
| From-scratch small | 0.673 | 0.479 | 16.62 |

Weighted-box fusion of the two best detectors:

| Validation set | Chained 7-class | From-scratch small | Fused |
| --- | --- | --- | --- |
| Target domain | 0.615 | 0.552 | 0.634 |
| Old street scenes | 0.674 | 0.687 | 0.679 |
| Non-motorised class AP50 | 0.589 | 0.654 | 0.654 |
| Inference cost (s per frame) | 0.09 | 0.09 | 0.19 |

Canonical prediction measurement (six clips, 300 frames each, resolution 1280,
confidence 0.25, static channel disabled, NMS corrected):

| Quantity | Value |
| --- | --- |
| Targets per frame | 33.3 |
| Predictable targets per frame | 14.87 |
| Predictable fraction | 0.446 |
| Median error at 1 s (px) | 33.4 |
| Per-clip median at 1 s, range (px) | 26.6 - 49.5 |
| Identity jumps per thousand frames | 18.3 |
| Median error at 0.1 s (px) | 3.8 |
| Median error at 0.5 s (px) | 14.3 |
| Median error at 2 s (px) | 67.3 |

Component ablation:

| Change | Before | After | Relative change |
| --- | --- | --- | --- |
| Identity guard: jumps per thousand frames | 72.2 | 43.9 | -39 percent |
| Identity guard: affected trajectories | 0.091 | 0.035 | -62 percent |
| Coasting: predictable targets per frame | 10.26 | 15.84 | +54 percent |
| Coasting: median error of coasted samples (px) | - | 29.3 | - |
| Input resolution 960 to 1280: target-domain mAP50 | 0.585 | 0.637 | +8.9 percent |
| Input resolution 960 to 1280: precision | 0.459 | 0.604 | +31.6 percent |
| Input resolution 960 to 1280: predictable per frame | 15.84 | 17.01 | +7.4 percent |
| Scene prior cache: milliseconds per frame | 318 | 81 | -74.5 percent |
| Off-scene coasting stop over 120 frames | - | 11 stops | - |

Error decomposition along the motion direction. The signed error `e_along` is
positive when the observed future position lies further along the current
motion direction than the prediction, i.e. positive means the prediction
undershoots. `e_cross` is the perpendicular component.

| Horizon (s) | Samples | Median abs error (px) | Median e_along (px) | Mean e_along (px) | Median e_cross (px) | Undershoot fraction | Kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 16635 | 3.8 | -0.8 | -1.2 | approximately 0 | 0.417 | 21.3 |
| 0.5 | 13125 | 14.3 | -6.5 | -5.4 | approximately 0 | 0.346 | 23.2 |
| 1.0 | 10004 | 30.5 | -15.6 | -9.0 | approximately 0 | 0.342 | 25.4 |
| 2.0 | 6248 | 67.3 | -36.9 | -40.6 | approximately 0 | 0.320 | 29.9 |

Calibration of the along-track displacement, regressing observed displacement on
predicted displacement with a free intercept:

| Horizon (s) | Slope | Intercept (px) | R squared | Through-origin gain | Mean abs error improvement from gain | Equivalent time offset (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.891 | +0.2 | 0.619 | 0.899 | +0.55 percent | -0.008 |
| 0.5 | 0.960 | -2.7 | 0.704 | 0.940 | +2.50 percent | -0.064 |
| 1.0 | 0.919 | +0.5 | 0.573 | 0.921 | +2.47 percent | -0.170 |
| 2.0 | 0.633 | +34.0 | 0.340 | 0.711 | +3.41 percent | -0.516 |

Concentration of error at the 1 s horizon:

| Aggregation | Top 1 percent share | Top 5 percent share | Top 10 percent share | Units |
| --- | --- | --- | --- | --- |
| By sample | 0.105 | 0.293 | 0.432 | 10004 |
| By trajectory | 0.117 | 0.352 | 0.529 | 365 |
| By frame | 0.036 | 0.137 | 0.240 | 1584 |

Scale-mixture fits to the normalised error at the 1 s horizon (per class, error
divided by box height):

| Class | Samples | Core weight | Core scale | Heavy scale | Scale ratio | Heavy-component variance share |
| --- | --- | --- | --- | --- | --- | --- |
| Car | 5630 | 0.861 | 0.936 | 3.644 | 3.9 | 0.710 |
| Non-motorised | 1595 | 0.718 | - | - | 7.27 | 0.950 |
| Pedestrian | 528 | 0.749 | - | - | 6.56 | 0.940 |
| Bus | 772 | 0.766 | - | - | 3.27 | 0.770 |
| Sign | 209 | 0.640 | - | - | 2.82 | 0.820 |

Model selection over error distributions, aggregated over 24 class-by-horizon
cells, counting cells won:

| Family | Smooth | Cells won by BIC | Cells won by KS distance |
| --- | --- | --- | --- |
| Student-t | yes | 8 | 6 |
| Two-component Gaussian mixture | yes | 8 | 17 |
| Asymmetric Laplace | no | 4 | 1 |
| Gaussian | yes | 2 | 0 |
| Laplace | no | 2 | 0 |
| Two-piece Gaussian | no | 0 | 0 |
| Skew-normal | yes | 0 | 0 |

Shape diagnostics on standardised error (median-centred, robust scale):

| Quantity | Value |
| --- | --- |
| Log-density slope, near tail (abs z in 1 to 2.5) | -0.82 |
| Log-density slope, far tail (abs z in 2.5 to 4) | -0.53 |
| Far-to-near slope ratio | 0.65 |
| Empirical mass within 0.1 robust sd of the mode, car | 0.20 |
| Gaussian-predicted mass in the same window | 0.08 |
| Student-t predicted mass in the same window | 0.19 |
| Two-component mixture predicted mass | 0.17 |
| Laplace log-likelihood minus Gaussian log-likelihood | 3078 |
| Hazard rate trend | decreasing in every class |

Scale-normalisation control. Errors were divided by box height, which is
available at prediction time and is the natural pixel-scale proxy under
perspective.

| Quantity | Value |
| --- | --- |
| Correlation of log abs error with log box height | 0.192 |
| Variance of log abs error explained by box height | 0.037 |
| Label flip rate between pixel-space and normalised tail labels | 0.125 |

Causal gate for the heavy-component label, leave-one-clip-out, predicting the
responsibility-above-0.5 label:

| Feature set | Label | Out-of-fold AUC | Average precision |
| --- | --- | --- | --- |
| All features | pixel tail | 0.769 | 0.430 |
| All features minus box height | pixel tail | 0.726 | 0.362 |
| All features | normalised tail | 0.753 | 0.448 |
| All features minus box height | normalised tail | 0.749 | 0.448 |
| All features minus every scale proxy | normalised tail | 0.714 | 0.408 |

Operating points of the gate on the normalised label without box height, after
removing the sign class:

| Precision threshold | Recall | Flagged fraction | Retained fraction |
| --- | --- | --- | --- |
| 0.4 | 0.650 | 0.338 | 0.662 |
| 0.5 | 0.432 | 0.179 | 0.821 |
| 0.6 | 0.159 | 0.055 | 0.945 |

Per-class out-of-fold AUC of the normalised-label gate without box height, after
removing the sign class:

| Class | Samples | Heavy-label fraction | AUC |
| --- | --- | --- | --- |
| Pedestrian | 2378 | 0.475 | 0.754 |
| Car | 27006 | 0.196 | 0.744 |
| Truck | 619 | 0.580 | 0.717 |
| Non-motorised | 7033 | 0.173 | 0.654 |
| Bus | 3141 | 0.112 | 0.550 |

Upper bound of any post-hoc gate, obtained by using the true responsibility
instead of a prediction, measured on pixel error:

| Retained fraction | Median error (px) | 90th percentile error (px) |
| --- | --- | --- |
| 1.00 | 14.1 | 94.3 |
| 0.80 | 11.5 | 58.5 |
| 0.60 | 8.9 | 48.0 |
| 0.50 | 9.7 | 47.0 |

Attribution of pixel error mass over all four horizons:

| Class | Samples | Share of total error | Median error (px) |
| --- | --- | --- | --- |
| Car | 27006 | 0.685 | 14.1 |
| Non-motorised | 7033 | 0.178 | 12.7 |
| Bus | 3141 | 0.055 | 14.4 |
| Pedestrian | 2378 | 0.050 | 14.9 |
| Sign | 813 | 0.016 | 18.9 |
| Truck | 619 | 0.015 | 17.1 |

Attribution of pixel error mass by mechanism tag:

| Tag | Samples | Share of total error | Median error (px) |
| --- | --- | --- | --- |
| Manoeuvring (turn rate above threshold) | 7755 | 0.301 | 22.5 |
| Large target (box height at or above median) | 20503 | 0.648 | 18.7 |
| Small target (box height below median) | 20502 | 0.352 | 10.6 |
| Coasted sample | 3233 | 0.092 | 23.4 |
| Static-structure class | 828 | 0.016 | 18.6 |

Analytically evaluated zero-displacement prediction for static-structure
classes, i.e. setting the predicted box equal to the current box:

| Class and horizon | Median error before (px) | Median error after (px) | P90 before (px) | P90 after (px) |
| --- | --- | --- | --- | --- |
| Sign, 0.1 s | 4.3 | 2.9 | 17.4 | 11.8 |
| Sign, 0.5 s | 15.9 | 6.7 | 46.0 | 23.5 |
| Sign, 1.0 s | 31.9 | 9.5 | 78.2 | 30.6 |
| Sign, 2.0 s | 64.3 | 10.6 | 125.1 | 38.9 |

Effect of withholding the prediction box (targets are still detected and
tracked) on pixel error:

| Policy | Retained fraction | Median error (px) | P90 error (px) | Mean error (px) |
| --- | --- | --- | --- | --- |
| Baseline, all samples | 1.000 | 14.1 | 94.3 | 39.8 |
| No prediction for manoeuvring targets | 0.811 | 12.5 | 82.5 | 34.3 |
| No prediction for coasted targets | 0.921 | 13.2 | 93.3 | 39.3 |
| No prediction for both | 0.741 | 11.6 | 81.2 | 33.6 |

Negative results. Every alternative below was implemented, measured under the
same protocol, and rejected.

| Family | Measured outcome |
| --- | --- |
| Constant acceleration | 3 to 8 times worse than constant velocity |
| Constant turn rate and velocity | Damped turn-rate sweep has zero gain at every setting |
| Interacting multiple models | Over 8573 paired samples, more accurate on only 50 percent; median, mean and P90 all slightly worse |
| NIS-adaptive Kalman | P90 degrades from 116.6 to 125.8 while median only approaches a fixed process-noise setting |
| Ground-plane bird's-eye-view | Median 34.6 to 45.9 px, mean 57.7 to 167.1 px; loses most on the depth-change stratum where it should win |
| Tiled inference with standard post-processing | Target domain 0.615 to 0.600, old street scenes 0.674 to 0.634 |
| Input resolution 1600 | Worse than 1280 |
| Larger backbone (small vs nano) | Target domain 0.597 against 0.637; recall lower by 23 percent |
| Alternative trackers (BoT-SORT, OC-SORT) | Median error differs by 1 to 3 px on matched targets; no operating-point gain |
| Long association buffer | Predictable targets rise to 12.11 per frame but jumps double from 72 to 149 per thousand frames |
| Removing stationary-class targets from the metric | Makes the metric worse |
| Lower process noise sweep | Improves median by 5 to 6 percent but degrades P90 by 3 to 8 percent |

Dataset construction statistics:

| Quantity | Value |
| --- | --- |
| Clips in the first self-collected set | 15 |
| Scene stills in the first self-collected set | 5 |
| Hand-verified frames | 277 |
| Hand-verified boxes | 3612 |
| Clips in the second self-collected set | 14 |
| Frames in the second self-collected set | 13371 |
| Duration of the second self-collected set (minutes) | 6.2 |
| Frames pre-labelled automatically and then reviewed | 332 |
| Transient static boxes removed by cross-frame voting | 8055 |
| Mined supplement frames containing bicycles or motorcycles | 191 |
| Additional mined frames | 220 |
| Frames used to extend the validation set | 58 |
| Aerial tricycle-conversion frames | 2661 |
| Aerial tricycle-conversion boxes | 9635 |
| Aerial train / validation split | 2262 / 399 |
| Class entries in the base schema | 12 |
| Class entries in the merged output schema | 10 |
| Unit of the train / validation split | video clip |

Measured disagreement between the automatic pre-labels and the human correction
pass on the same 277 frames:

| Statistic | Value |
| --- | --- |
| Boxes produced automatically | 3970 |
| Boxes deleted by the human reviewer | 592 |
| Deletion rate | 0.149 |
| Boxes added by the human reviewer | 234 |
| Addition rate | 0.065 |
| Boxes re-classified | 177 |
| Deletion rate for the truck class | 0.448 |
| Deletion rate for the bicycle class | 0.405 |
| Deletion rate for the person class | 0.252 |
| Train / validation leakage of an earlier pipeline version that mined samples from all clips | 0.209 |

Adjudication of automatically proposed static-structure positions:

| Outcome | Count |
| --- | --- |
| Position candidates proposed by temporal voting | 282 |
| Retained without further review | 136 |
| Routed to human confirmation | 63 |
| Rejected | 83 |
| Rejection rate | 0.29 |

Deployment-relevant measurements:

| Quantity | Value |
| --- | --- |
| End-to-end cost per frame (s) | 0.24 |
| Detection cost per frame at resolution 1280 (ms) | 17 to 24 |
| Throughput on one 8 GB GPU (frames per second) | approximately 4 |
| Static channel cost per frame with scene prior (ms) | 81 |
| Static channel cost per frame without scene prior (ms) | 318 |
| Wall-clock for an 816-frame clip with scene prior (minutes) | approximately 1 |
| Wall-clock for the same clip without scene prior (minutes) | approximately 4 |
| Export bitrate of the built-in encoder at 1080p (MB per second) | 2.55 |
| Distribution package sizes, slim / full / reviewer (MB) | 5.3 / 18.2 / 46.8 |

Streaming-efficiency measurements. Every configuration consumed 600 source frames at
input resolution 1280 unless stated otherwise; rendering and JSON export were disabled
so that only the analysis path was timed. Detection was subsampled with a frame stride:
intermediate frames were skipped with a container-level grab (no decode).

| Configuration | Milliseconds per source frame | Equivalent source fps |
| --- | --- | --- |
| Stride 1 (baseline), resolution 1280 | 48.2 | 20.8 |
| Stride 2, resolution 1280 | 29.1 | 34.4 |
| Stride 3, resolution 1280 | 23.2 | 43.1 |
| Stride 1, resolution 736 (720p tier) | 44.8 | 22.3 |
| Stride 3, resolution 736 (720p tier) | 22.1 | 45.4 |
| Full offline path (render plus JSON), stride 1, resolution 1280 | 69.2 | 14.4 |

Accuracy of frame subsampling with the ground truth held fixed to the dense stride-1
observations. Predictions from each subsampled run were attached to dense trajectories
by spatial nearest-neighbour matching at the same timestamp (same class, 15 pixel
tolerance), because track identifiers are assigned independently in each run. Three
clips were evaluated; each consumed 600 source frames.

| Clip | Stride 1, median at 1 s (px) | Stride 2, median at 1 s (px) | Stride 3, median at 1 s (px) | Stride 1, median at 2 s (px) | Stride 3, median at 2 s (px) |
| --- | --- | --- | --- | --- | --- |
| First clip | 23.3 | 21.7 | 20.9 | 62.5 | 61.5 |
| Second clip | 28.6 | 18.4 | 14.8 | 61.7 | 26.7 |
| Third clip | 38.2 | 33.8 | 31.7 | 60.8 | 33.9 |

Matching coverage: stride 1 matched 6000 targets with 0 unmatched; stride 2 matched
2918 with 15 unmatched; stride 3 matched 1816 with 27 unmatched.

## 3. Qualitative Observations

- Automatic pre-labelling was not a substitute for human review: a human
  deleted 14.9 percent of the automatically produced boxes and added a further
  6.5 percent, so the automatic labels alone would have trained the detector on
  a systematically different target. The classes with the highest deletion rates
  were trucks (44.8 percent) and bicycles (40.5 percent), both of which are rare
  in the data, which means the automatic labeler's errors concentrated exactly
  where the data was scarcest.
- Changing the class schema mid-project was only survivable because the first
  nine class indices were frozen. When riders were moved out of the person class
  and given their own class, previously labelled frames stayed valid.
- An earlier version of the supplemental-sampling pipeline mined rare-class
  frames from every clip including the validation clips, producing 20.9 percent
  train/validation leakage; the resulting metrics were inflated and had to be
  recomputed. Splitting by video clip is now enforced in the build script.
- Temporal voting alone could not identify what a static position *was*:
  roughly a third of the proposed static positions were rejected by semantic
  adjudication, and the rejected cases were kerbs, flower-bed rims and shop
  signage, all of which are perfectly stationary.
- Annotation was distributed to collaborators as offline batches containing a
  zero-installation browser tool, a class list and a written rule sheet, with
  only labels returned; this removed the project's Python environment as a
  prerequisite for review.
- The system is a batch processor rather than a live service: at 0.24 s per
  frame it processes recorded clips comfortably but cannot keep up with a live
  stream, and the dominant cost is the static-structure channel, which the
  per-camera scene prior removes from the second pass onward.
- Exported video at 1080p uses a codec whose bitrate is 2.55 MB per second, so a
  three-minute demonstration exceeds typical submission size limits unless the
  export is re-encoded.
- The prediction ground truth throughout this work is the future output of the
  same tracker that produced the predictions, which measures self-consistency
  rather than absolute correctness; independent human truth remains the one
  change that could move the reported numbers in either direction.
- Detection on the target domain improved substantially with chained
  fine-tuning relative to training from scratch with the same recipe, even
  though the from-scratch model was trained for fewer epochs; training history
  mattered more than model capacity.
- The open-vocabulary static-structure channel behaved as intended on bollards
  that the fine-tuned detector never learned, and the scene-prior cache
  reproduced identical output while reducing wall-clock cost by a factor of
  four.
- Static-structure classes were assigned a spurious velocity of approximately
  30 px/s by the tracker, because detection jitter on a stationary object is
  interpreted as slow motion. Their extrapolated displacement was therefore
  wrong in essentially every sample; the ratio of absolute along-track error to
  speed-times-horizon was 1.02 to 1.28 for signs, against 0.2 to 0.4 for
  vehicles.
- Errors were almost entirely along the motion direction; the perpendicular
  component had a median within 0.4 px of zero for every class at every horizon.
- The dominant systematic behaviour was overshoot: predictions were too far
  along the motion direction in 58 to 68 percent of samples.
- Equivalent time offset grew with horizon (-0.008 s to -0.516 s), which
  indicated a velocity-magnitude error rather than a time-base misalignment.
- The empirical error distribution was visibly more peaked and more heavily
  tailed than a Gaussian; the density near the mode was 2.5 times the Gaussian
  prediction. A two-piece Gaussian, which shares the peaked shape but imposes a
  kernel, was not selected in any of the 24 class-by-horizon cells, and its
  prediction of near-mode mass (0.08) was as poor as the Gaussian's.
- All smooth scale-mixture families outperformed all fixed-shape families under
  both likelihood and Kolmogorov-Smirnov criteria.
- The hazard rate of the error distribution decreased with magnitude in every
  class, which is the signature of a scale mixture rather than of a fixed-shape
  law.
- Dividing the error by box height did not remove the two-component structure,
  and the heavy-component label changed for only 12.5 percent of samples, which
  showed that the heavy tail was not a perspective artefact.
- The causal gate that predicts heavy-component membership recovered real signal
  (AUC 0.714 even with every scale proxy removed), but the signal was not
  separable enough to isolate the failure mode: raising the precision threshold
  from 0.4 to 0.6 collapsed recall from 0.650 to 0.159.
- In pixel space the gate's apparent benefit came mostly from flagging large,
  near targets, which are precisely the objects a user watches; when the heavy
  label was defined in scale-free units, the same gate no longer improved pixel
  error and, at aggressive thresholds, made it worse.
- The pixel error budget was dominated by cars (68.5 percent of total error) and
  by non-motorised vehicles (17.8 percent), while static structures accounted for
  only 1.6 percent, so mechanism-level fixes to static structures produced large
  local improvements but only a one percent change in the aggregate.
- Per-sample mean error was nearly uniform across classes (28 to 41 px); the
  attribution was driven by class frequency rather than by per-class accuracy.
- The sign class produced an out-of-fold AUC of 0.266 in the normalised gate,
  below chance, because its mixture fit was unstable at 813 samples; removing it
  improved the weakest per-class AUC from 0.266 to 0.550.
- Human review of 40 self-retained frames found almost no residual labelling
  problems, while a 20-frame product spot check surfaced four recurring
  false-positive families: red shop signage classified as traffic lights,
  kerb stones classified as bollards, shop signage classified as trucks, and
  pedestrians walking beside parked vehicles classified as riders.
