## Problem Statement

Fixed traffic cameras observe but do not anticipate: operators see where vehicles
*are*, never where they *will be* within the next fraction of a second to two
seconds. Radar and LiDAR solve this at a hardware cost that is prohibitive for
the large installed base of monocular cameras. The technical problem is
therefore: given only a monocular video segment (2-30 seconds) from an
uncalibrated, possibly handheld camera, produce a short-horizon
(0.1 / 0.5 / 1 / 2 second) prediction of every detected target's future
position, with the prediction computed causally from the current frame and its
past only.

Three properties make this harder than a standard tracking benchmark.
(i) The prediction must be *causal*: no future frame may inform the current
prediction.
(ii) The evaluation must not be self-referential: predicting a target's future
position and scoring it against the same tracker's future output is a circular
measurement that flatters any tracker-consistent motion model.
(iii) The error distribution of short-horizon prediction is not Gaussian and not
unimodal, so aggregate statistics such as mean absolute error conflate several
distinct failure mechanisms.

## Core Hypothesis

Short-horizon monocular prediction is *not* limited by model capacity or by the
sophistication of the motion model. A constant-velocity Kalman filter is already
near the achievable frontier, because at sub-two-second horizons the dominant
error terms are not motion-model error but (a) detection/tracking instability,
(b) scene elements that violate the "everything moves" prior (bollards, signs,
luminaires), and (c) a minority of targets that contribute the majority of the
error variance. Consequently the productive engineering directions are
mechanism-level: stabilise the identity association, treat static scene
structure explicitly, and characterise the error *distribution* rather than
chase mean error with heavier models.

## Proposed Methodology (Detailed Technical Approach)

The system is a causal detection-tracking-extrapolation pipeline. Each frame is
processed once: detection, multi-object association, a still/variable state
machine, per-track Kalman extrapolation at four horizons, static-structure
injection, prediction rendering and a machine-readable accuracy report.

Only the following four components are claimed as contributions; the remaining
plumbing follows standard practice.

### C1. Static-structure latent map with semantic verification and scene prior

Fixed scene elements (bollards, signs, luminaires, traffic lights) are detected
intermittently by any object detector, producing flicker. We build a *latent
map* of such positions in three stages:

1. **Scan.** An open-vocabulary detector is run over the whole segment at low
   confidence with a prompt ensemble (e.g. `concrete bollard`, `barrier post`,
   `traffic bollard`), trading precision for recall.
2. **Temporal vote.** Detections are clustered by position across frames; a
   candidate is retained when it is observed in at least 80% of the frames in
   which it is not occluded and in at least 4 frames.
3. **Semantic adjudication.** Because temporal stability alone cannot
   distinguish a bollard from a kerb, a flower-bed rim or a shop billboard -
   all of which are equally stationary - each candidate is cropped with 1.8x
   context and scored zero-shot against positive and negative prompt sets with
   a CLIP-style image-text model. Verdicts are three-way:
   `keep` / `review` / `reject`. Only `keep` positions are drawn; `review`
   positions are retained for human adjudication rather than silently deleted,
   because deleting a true target is a much more costly error than leaving a
   false one.

Candidates confirmed once are cached per camera as a **scene prior**
$\Phi_{\text{cam}}$, so the expensive scan and adjudication run once per
installation rather than once per video.

### C2. Identity-guard and occlusion coasting

Two failure modes of the association layer dominate outlier error.

*Track guard.* When a target's centroid displacement exceeds
$\max(40\,\text{px},\; 4\times \text{median step})$ **and** its box overlap with
the previous frame is below 0.1, the identity is severed and a new track is
started. This converts identity switches into short tracks instead of long
trajectories with absurd velocity.

*Coasting with an off-scene stop rule.* When a mature track
($\geq 5$ frames, speed $\geq 8$ px/s) is not reported by the detector, its
position is extrapolated for up to 12 frames so that the rendered box passes
behind the occluder instead of vanishing. Coasting is stopped as soon as the
extrapolated centroid leaves the image, which removes the "empty box walking
off screen" artefact.

### C3. Still/variable state machine with a net-displacement criterion

A track is classified `constant` (not rendered, not predicted) after six
consecutive frames of non-motion, and returns to `variable` as soon as it
moves. Two design choices matter. First, the tolerance is scaled by box height,
because a near object's centroid jitters more in pixels than a far one's.
Second, a *net-displacement* criterion (mean of the first two samples vs mean of
the last two in the window) is evaluated before the range criterion: jitter
cancels in the net displacement while genuine slow motion does not. Without it,
slow pedestrians are misclassified as stationary. Static-structure classes are
deliberately exempt from this suppression - they must always be visible.

### C4. Distributional diagnosis of prediction error

We treat the prediction error along the motion direction as a random variable
and fit a sequence of nested models to it: Gaussian, Laplace, two-piece
(split) Gaussian, asymmetric Laplace, Student-$t$, and a two-component Gaussian
mixture. The last two are *scale mixtures*, i.e. they express the error as a
narrow core plus a heavy-tailed minority of samples. Because prediction error is
only observable after the future arrives, the posterior responsibility of the
heavy component provides a *label* for the failure mode, not a usable online
feature; the corresponding online question - whether the failure mode is
predictable from causal features - is posed separately.

### C5. Dataset construction and annotation protocol

The only hand-verified ground truth in this work is 277 frames (3612 boxes)
reviewed frame by frame, and the protocol that produced it is itself a
contribution, because three of its rules are what make the labels usable.

1. **Vehicles before people, riders are not people.** A person riding a
   bicycle, motorcycle or tricycle, and a person in a driver's seat, is labelled
   only as the vehicle; only a pedestrian on foot is labelled as a person.
   Riders are additionally given their own class rather than deleted, because
   deleting them destroys evidence that a vehicle was occupied. Without this
   rule the person class is inflated by exactly the instances that a
   short-horizon predictor must treat as moving vehicles.
2. **A frozen class order with append-only extension.** The first nine class
   indices are never renumbered; every new class is appended. This keeps
   previously labelled frames valid when the schema grows, and it is why the
   project can carry four schemas (12-class base, 10-class merged, 9-class
   legacy, 7-class deployed) over one set of source labels.
3. **Asymmetric error preference.** Annotators are instructed to keep
   uncertain boxes rather than delete them. Missing a true target damages the
   trained model far more than leaving a false one, and a false detection can be
   removed later by a targeted review pass whereas a deletion cannot be
   recovered.

Two further mechanisms make the protocol scale beyond the team. First,
automatic pre-labelling is followed by a *measured* human correction pass whose
statistics are reported (how many boxes humans delete, add and re-class), so the
residual error of the automatic labels is known rather than assumed. Second,
annotation is distributed as self-contained offline batches: an image folder, a
`classes.txt`, a written rule sheet and a zero-installation browser tool, with
results returned as labels only. This decouples review throughput from the
project's own environment.

Two hygiene rules are non-negotiable. Supplemental samples mined for rare
classes must exclude every validation clip: an earlier version of this pipeline
mined from all clips and produced 20.9% train/validation leakage with
correspondingly inflated metrics. And static-structure candidates that pass the
temporal vote are still routed to human adjudication rather than being trusted,
because temporal stability cannot distinguish a bollard from a kerb.

## Path to deployment

The system is a batch processor, not a streaming service, and the honest
deployment path has four steps.

**1. Make it real-time.** End-to-end cost is 0.24 s per frame, i.e. about four
frames per second on one 8 GB GPU - enough for offline processing of recorded
clips, not for live monitoring. The dominant cost is the open-vocabulary static
channel, which the per-camera scene prior removes after the first pass; the
remaining cost is detection at resolution 1280. Live operation therefore
requires either a lower detection resolution on a subset of frames or a
detection-then-refine schedule, not a larger model.

**2. Calibrate once per camera.** The system currently reports pixels. With the
camera fixed - which is the normal case for installed traffic cameras - a
one-time calibration converts predictions into metres and hence into km/h, and
unlocks the ground-plane motion modelling that our uncalibrated attempt could
not exploit. This is the single highest-value deployment change because it adds
an output (metric speed) rather than improving an existing one.

**3. Close the compliance and privacy gaps.** The detection and association
stack is AGPL-3.0, so a closed-source delivery must replace it with a
permissive-licensed detector and a self-implemented tracker; this path was
assessed and is feasible. Recorded scenes contain faces and plates and must be
de-identified before any public release, and the third-party aerial dataset
must not be redistributed.

**4. Replace the self-referential ground truth.** Every prediction number in
this paper is measured against the same tracker's future detections, which
measures self-consistency rather than correctness. Independent human truth over
a bounded set of clips is the one input that can change the reported numbers in
either direction, and it is the only remaining route to a defensible accuracy
claim.

## Expected Contribution

1. A deployable monocular short-horizon prediction system that runs on a single
   8 GB consumer GPU and reports its own accuracy per horizon.
2. An explicit characterisation of the error *distribution* of short-horizon
   prediction: a two-scale mixture whose heavy component is 3-8x wider than its
   core yet contains a minority of samples, and which accounts for the majority
   of the variance.
3. A systematically negative map of the design space: eleven families of
   alternatives (constant acceleration, constant turn-rate, interacting multiple
   models, adaptive Kalman, ground-plane/BEV modelling, tiled inference, higher
   input resolution, larger backbones, alternative trackers, longer association
   buffers, stationary-class removal) are implemented and measured, and none
   improves the operating point.
