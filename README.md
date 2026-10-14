# CardioVision 3D — Cardiovascular Risk Visualization & Prediction

> Predicts overall coronary artery disease (CAD) status and per-vessel stenosis
> probability for the **LAD**, **LCX** and **RCA** from tabular clinical features,
> with TreeSHAP attributions, served to an interactive 3D anatomical dashboard.

---

## ⚠️ Clinical Safety Disclaimer

**FOR CLINICAL DECISION SUPPORT AND EDUCATIONAL PURPOSES ONLY.**
This system produces statistical risk estimates from a 303-patient research cohort.
It is **not** a diagnostic device and must **not** replace invasive coronary
angiography, diagnostic cardiac imaging, or physician judgement. Per-vessel
performance in particular (see LCX and RCA below) is modest and is reported honestly
rather than presented as diagnostic accuracy.

---

## Status

| Component | State |
| :--- | :--- |
| ML training pipeline | ✅ Rebuilt — nested CV, no leakage, calibrated |
| Backend API | ✅ Rebuilt — serves the trained models with real TreeSHAP |
| Test suites | ✅ 31 Python (backend + anatomy) · 45 frontend (parser + completeness) |
| 3D viewer + dashboard | ✅ Rebuilt — BodyParts3D anatomy, per-vessel risk mapping |
| Detection validation CLI | ✅ `ml/src/validate.py` |

---

## Dataset

**UCI Z-Alizadeh Sani** (`ml/data/z_alizadeh_sani.xlsx`) — 303 patients, 59 columns.

- 55 clinical input features across demographic, examination, ECG, laboratory and
  echocardiographic groups.
- 4 angiographic targets: `Cath` (overall CAD), `LAD`, `LCX`, `RCA`.
- No missing cells and no duplicate rows.
- `Exertional CP` is constant (`'N'` for all 303 patients) and is dropped
  automatically by a `VarianceThreshold` step.

**Target leakage prevention (requirement 1d).** `CAD`, `Cath`, `LAD`, `LCX` and `RCA`
are removed before the feature matrix is built, asserted absent in training, asserted
again when the API loads the artifacts, and stripped from inbound request payloads —
so even a caller who sends the answers cannot change a prediction. There is a test
for each of those layers.

A structural note worth knowing: **`Cath` equals `OR(LAD, LCX, RCA)` on 302 of 303
rows.** Vessel involvement splits 86 / 87 / 67 / 63 across 0, 1, 2 and 3 vessels.

---

## Evaluation

Nested stratified 5-fold cross-validation. The outer folds report performance; the
**inner** folds select each target's decision threshold (Youden's J), so the tuned
figures are never scored on the data that chose the threshold. All preprocessing is
fit inside each training fold.

### Ranking quality (threshold-independent)

| Target | Prevalence | ROC-AUC | PR-AUC | Brier |
| :--- | :---: | :---: | :---: | :---: |
| **Overall CAD** | 71.3% | **0.9136 ± 0.0537** | 0.9604 ± 0.0217 | 0.1069 ± 0.0416 |
| **LAD** | 58.4% | **0.8356 ± 0.0495** | 0.8756 ± 0.0453 | 0.1600 ± 0.0265 |
| **LCX** | 39.3% | **0.7421 ± 0.0715** | 0.6433 ± 0.0760 | 0.2050 ± 0.0278 |
| **RCA** | 37.6% | **0.6976 ± 0.0467** | 0.5794 ± 0.0826 | 0.2234 ± 0.0227 |

PR-AUC is the more informative curve for LCX and RCA, whose positive class is the
minority.

### Operating points

Both thresholds are reported so nothing looks cherry-picked.

| Target | Threshold | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CAD | 0.500 | 0.8546 | 0.8761 | 0.9304 | 0.9014 |
| CAD | **0.659** (tuned) | 0.8710 | 0.9213 | 0.8982 | **0.9090** |
| LAD | 0.500 | 0.7790 | 0.7969 | 0.8367 | 0.8156 |
| LAD | **0.529** (tuned) | 0.7823 | 0.8021 | 0.8421 | **0.8180** |
| LCX | 0.500 | 0.6963 | 0.6139 | 0.6388 | 0.6247 |
| LCX | **0.353** (tuned) | 0.6303 | 0.5210 | **0.7728** | 0.6208 |
| RCA | 0.500 | 0.6633 | 0.5939 | 0.4553 | 0.5022 |
| RCA | **0.229** (tuned) | 0.6108 | 0.4928 | **0.7451** | 0.5881 |

**Read the LCX and RCA rows carefully.** Thresholds are tuned for sensitivity, which
suits a decision-support tool that should not quietly miss disease, but the cost is
real: LCX precision falls to 0.52 and RCA to 0.49. At the tuned operating point
roughly half of the vessels flagged red are false alarms. RCA at ROC-AUC 0.70 is only
modestly better than chance and should be presented as a weak signal, not a finding.

Threshold tuning is what lifted vessel recall — LCX from 0.64 to 0.77 and RCA from
0.46 to 0.75 — not model changes. `class_weight="balanced"` is applied to **LCX only**:
under nested CV it helps LCX marginally (ROC-AUC 0.736 → 0.742) and makes RCA worse on
every metric, so it was dropped there. Those gaps all sit inside one standard
deviation, so treat them as weak preferences rather than results. The underlying
separability is what it is at N=303.

### Calibration

Calibration is a 1-D monotonic map fit on out-of-fold scores, with the method chosen
by cross-validating *on those scores* so the reported Brier is not scored on the data
that fit the map.

| Target | Method | Brier before → after |
| :--- | :--- | :--- |
| CAD | identity | 0.1067 → 0.1067 |
| LAD | sigmoid | 0.1601 → 0.1596 |
| LCX | sigmoid | 0.2049 → 0.2011 |
| RCA | sigmoid | 0.2233 → 0.2130 |

CAD keeps the identity map because LightGBM is already well calibrated there and no
candidate map improved the held-out Brier — adding one would be a moving part for
nothing.

### CAD coherence

Because `Cath` is nearly the disjunction of the three vessels, the dashboard shows the
directly-trained CAD head **and** a vessel-derived noisy-OR, `1 − Π(1 − pᵥ)`, plus
whether the two agree. The derived score is itself calibrated: the independence
assumption ranks well but inflates the raw scale, since real vessel stenoses are
correlated through shared risk factors.

| Metric | Value |
| :--- | :---: |
| Direct CAD head ROC-AUC | 0.9070 |
| Vessel-derived ROC-AUC | 0.9063 |
| Direct vs derived agreement | 89.8% |
| Derived mean before → after calibration | 0.769 → 0.713 |
| Observed CAD prevalence | 0.713 |

Figures (ROC, PR, reliability, confusion matrices, coherence scatter) are written to
`ml/results/` by `evaluate.py`.

---

## Verifying detection

`ml/src/validate.py` answers "is it actually finding the disease?". Every figure in
the default report is **out-of-fold** — each patient was scored by a model that never
saw them during training.

```bash
python ml/src/validate.py              # per-target detection report
python ml/src/validate.py --errors     # every patient the model gets wrong
python ml/src/validate.py --patient 6  # one patient, predicted vs actual
python ml/src/validate.py --live       # check a running API on all 303 rows
```

Held-out detection at the tuned thresholds:

| Target | Caught | False alarms | Sensitivity | Specificity | PPV | NPV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CAD | 195 / 216 | 18 | 90.3% | 79.3% | 91.5% | 76.7% |
| LAD | 146 / 177 | 34 | 82.5% | 73.0% | 81.1% | 74.8% |
| LCX | 95 / 119 | 84 | 79.8% | 54.3% | 53.1% | 80.6% |
| RCA | 84 / 114 | 89 | 73.7% | 52.9% | 48.6% | 76.9% |

**The CAD head is the trustworthy output.** The vessel heads are not equally good:
LCX and RCA buy their sensitivity with precision, so roughly half the vessels they
flag are false alarms, and **all three vessels are simultaneously correct in only
35.6% of patients**. The system over-calls vessel disease on average — mean predicted
vessel count 1.76 against an actual 1.35. Those numbers are surfaced in the
dashboard's model card rather than left in this file.

`--errors` is the useful one for diagnosing why: 20 of the 21 missed CAD patients had
no typical chest pain, and one (row 91) was missed by 0.001 of probability.

`--live` is what proves the *deployed* system rather than a notebook: it posts all 303
rows to the API and asserts each returned probability equals the offline pipeline's.

---

## Interpretability

Attributions come from LightGBM's native TreeSHAP (`pred_contrib=True`) — the same
Lundberg et al. algorithm `shap.TreeExplainer` wraps for LightGBM models — computed
by **the exact booster that served the prediction**. They are additive to machine
precision (~1e-15): contributions plus `shap_base_value` equal the raw log-odds
margin, and there is a test asserting it.

One honest caveat the UI must respect: SHAP is additive in **log-odds margin** space
while clinicians read probabilities. The API returns the calibrated probability as
`risk_score` and the margin-space attributions beside it. The map between them is
monotonic, so the ranking the bars explain is the ranking displayed — but the bars sum
to the margin, not to the percentage, and must be labelled as such.

---

## Getting data in

### Can a report supply the inputs? Mostly not, and here is the measurement

The obvious question about a form with 55 fields is whether a lab report could just
fill it in. Mean |SHAP| over all 303 patients, summed across the four models and
grouped by the document a field could come from:

| Source | Fields | Share of predictive weight |
| :--- | ---: | ---: |
| Lab panel (lipids, CBC, RFT, FBS, ESR) | 14 | 26.8% |
| Echo report (EF, RWMA, valve disease) | 3 | 18.1% |
| Demographics | 5 | 12.1% |
| ECG report | 7 | 5.5% |
| Vitals | 2 | 4.6% |
| **Symptoms** — clinical note only | 7 | **24.4%** |
| **History** — clinical note only | 12 | 8.4% |
| **Physical exam** — clinical note only | 5 | **0.0%** |

Documents can fill **31 of 55 fields, 67.2% of the weight**. Two findings follow, and
they shape the whole intake design:

1. **The strongest single driver cannot come from any report.** Typical Chest Pain has
   mean |SHAP| 1.26 — more than twice the next feature — and it is a symptom.
2. **The four models never split on 18 of the 55 inputs.** All five examination
   findings, plus EX-Smoker, Obesity, CRF, CVA, Airway disease, Thyroid Disease, CHF,
   Exertional CP, LowTH Ang, St Elevation, LVH, Poor R Progression, BBB. Their SHAP
   contribution is exactly zero: typing into them cannot change a prediction. They are
   collapsed behind a disclosure that says so rather than sitting beside Typical Chest
   Pain as though both mattered.

Subtracting the two leaves exactly **ten questions** that are both used by the models
and absent from every report — 25.6% of total weight, asked heaviest first:

```
Typical Chest Pain 15.20%   Atypical pain    1.02%   Nonanginal pain 0.54%
Diabetes            2.99%   Family history   1.01%   Dyslipidaemia   0.48%
Hypertension        1.63%   NYHA class       0.69%
Dyspnea             1.45%   Current smoker   0.60%
```

**Prediction is blocked until the first is answered.** Defaulting a field worth 15.2%
of the model means serving a confident percentage built on the cohort's most common
answer standing in for this patient's. The completeness meter reports covered *weight*,
not field count, for the same reason: "40 of 55 complete" sounds nearly done when the
missing one is the only one that matters.

The reproduction of these figures lives in `frontend/src/lib/completeness.ts`, and
`completeness.test.ts` checks them against `backend/app/models/metadata.json` so a
retrain cannot silently invalidate them.

### Reading a report (`src/lib/extract/`)

Drop a lab, echo or ECG report on the upload panel. **Nothing is transmitted** — the
PDF is parsed and any OCR runs in the page. For a medical record that is worth more
than a privacy policy, and it means the demo cannot fail on someone else's machine.

```
digital PDF  ->  pdf.js text layer  ->  the publisher's own characters, exact
scan / photo ->  rasterise -> tesseract.js  ->  a best guess
```

Most lab reports are generated as digital PDFs, so usually no OCR runs at all. The
review table says which path a document took, because "read from the text layer" and
"recognised from an image" deserve different levels of trust.

`readPdfTextLayer` rebuilds lines from item positions rather than concatenating in
order: "Sodium" and "139" are separate text items, and a report's whole meaning depends
on them staying on one row.

**Nothing is ever applied automatically.** Every candidate shows the verbatim line it
came from, so the parse can be checked against the page without trusting the parser.
Values outside the training range, and anything inferred from a drug name, start
switched off. Accepting a set starts a *new* patient rather than merging into whoever
was loaded, so a document's labs can never be combined with some other patient's
symptoms.

### Units are the hazard

The dataset is internally inconsistent, so the parser normalises per field and then
gates on the cohort's observed range. The range gate is what catches a unit error the
synonym matching missed — a white cell count of 7 or a platelet count of 210,000 is
surfaced, not used.

- **`WBC` is a raw count (3,700–18,000 cells/µL) but `PLT` is thousands (25–742
  ×10³/µL).** A CBC prints both in the same column with the same unit, so WBC needs
  multiplying by 1000 and PLT does not. Tested in both directions.
- **`BUN`, not urea.** Many labs report "Blood Urea"; urea mg/dL × 0.467 = BUN. The
  synonym list tries "blood urea nitrogen" first, or every real BUN would be converted
  too.
- **mmol/L is not interchangeable** and the factor differs per analyte: LDL/HDL ×38.67,
  TG ×88.57, glucose ×18.016, creatinine µmol/L ÷88.4.
- **`EF-TTE` tops out at 60** in this cohort, so "LVEF 65%" is outside the training
  range and is flagged rather than clamped silently.

Negation is checked in both directions, because a report negates on the left in prose
("no evidence of ST elevation") and on the right in a table ("ST elevation: absent").
It never looks across a newline — the previous row's "Normal" must not negate the next
analyte.

**Medications suggest, they do not fill.** A prescription lists drugs, not findings, but
metformin implies diabetes, a statin implies dyslipidaemia and an antihypertensive
implies hypertension. Those three arrive in the review table marked *inferred from
medication*, naming the drug, and start unaccepted.

**Handwriting is out of scope.** No OCR engine reads a handwritten prescription
reliably, and the upload panel says so rather than producing confident nonsense.

Offline OCR needs a one-time fetch; PDFs with a text layer never do:

```bash
python tools/fetch_ocr_assets.py          # ~18 MB into frontend/public/ocr/ (gitignored)
python tools/fetch_ocr_assets.py --check  # report status
```

### Browsing the research cohort

`GET /cohort` serves all 303 training patients with prev/next stepping, jump-to-number
and filters — including **"model got it wrong"**, which is 39 patients.

The figure listed beside each one is the **held-out** probability, from the
cross-validation fold where that patient was excluded. This matters: the served model
was fitted on every one of these rows, so its live output for them is memory as much as
inference. It agrees with angiography on 99.0% of them against a true held-out 90.3%.
Listing the live number would make the model look near-perfect by construction.

Loading a patient shows both, labelled — an `in-sample` tag on the live headline with
the held-out figure and the angiographic truth beside it. That is what
`python ml/src/validate.py --patient N` prints on the command line, brought into the UI.
Editing any field makes it a custom patient again.

```bash
python tools/build_cohort.py          # -> backend/app/models/cohort.json
python tools/build_cohort.py --check  # verify the export is current
```

It reads `ml/results/oof_predictions.npz`, so no retrain is needed, and it refuses to
write if the ground truth in that file disagrees with the dataset — which would pair a
held-out probability with the wrong patient.

---

## 3D visualization

### The source set, and what replaced it

The first build of this pipeline downloaded `partof_BP3D_4.0_obj_99.zip`, and two
consequences were visible on screen:

1. **That archive contains no myocardium.** Its only heart solids are the four chamber
   *cavities* — the blood pool — so the "heart" it drew was a set of blood-shaped blobs
   rather than cardiac muscle.
2. **Its arteries were too pitted to render at 99% polygon reduction**, so each was
   reduced to a ~40-point centreline and a synthetic tube was swept along it. The path
   came from the data, but every surface on screen was generated.

The current build reads the full-resolution **BodyParts3D heart part set**
(`heart5.0i`) from `heart-beat parts/`: **129 OBJ files, 93 FMA concepts, 271,706
vertices**, with real ventricular and atrial walls, individually named coronary
branches, valve leaflets and the conduction system. Both workarounds are gone — what
renders is the mesh itself, decimated but never reconstructed.

### Three things the files themselves tell you

Found by reading the OBJs rather than trusting any parts list. Each would have produced
a confidently wrong picture if taken on faith:

1. **Vertices are not welded.** Faces reference per-face duplicates, so a mesh that is
   one solid piece geometrically is a cloud of loose triangles topologically: merged
   raw, the LAD splits into **20 components whose largest holds 565 of 1518 vertices**.
   Welding at 1e-3 mm brings that to 3. Anything that walks edges — decimation,
   contraction, component counts — is wrong without it, and this was silently
   corrupting the old centrelines too.
2. **Multi-file structures are sequential segments, not alternatives.** LAD trunk
   `MM420 → MM424 → MM425` abut within **0.04 mm**, RCA `MM556 → MM436 → MM439` within
   0.12 mm, LCX `MM426 → MM635` within 0.22 mm. Merging per FMA gives one continuous
   artery — the thing that was impossible with the decimated archive.
3. **This donor is left-dominant.** The posterior interventricular branch is filed as
   `FMA79742 — trunk of posterior interventricular branch of the **left** coronary
   artery`, so the PDA belongs to the LCX territory here. In roughly 85% of people it
   arises from the RCA instead. The legend and the dashboard footer say so; grouping it
   silently would misrepresent both this model and the typical patient.

### What is in the scene

Each coronary territory is the trunk **plus its named branches**, so colouring a vessel
colours everything it supplies.

| Node | FMA concepts | Tris | Structure |
| :--- | :--- | ---: | :--- |
| `artery_LAD` | 74912, 3860, 3888, 3890, 3893 | 13,574 | anterior interventricular trunk + diagonal, 2nd/3rd anterior, septal |
| `artery_LCX` | 74923, 3902, 3914, 79742 | 20,272 | circumflex trunk + left marginal, 1st posterior ventricular, PDA |
| `artery_RCA` | 3802, 3807, 3815, 3818, 3823, 3829, 3837, 3851 | 13,764 | RCA trunk + conus, SA nodal, marginal, AV nodal, ventricular, atrial |
| `artery_left_main` | 4685 | 500 | left coronary stem — proximal to the bifurcation, so attributed to neither |
| myocardium ×5 | 84850, 9345, 7134, 83468, 49207, 49208 | 59,310 | LV free and septal walls, IV septum, RV inflow and outflow walls |
| atria ×2 | 9306, 84095, 9562, 49292, 9542, 9547, 49285, 49286, 84094, 49299 | 29,190 | both atrial wall sets with their auricles |
| great vessels ×2 | 23733, 15098, 15086 | 7,610 | ascending aorta with its bulb, pulmonary trunk |
| valves ×4 *(layer)* | 7242/7243/9498, 7238/7239/7240/85237, 7252/7253/7254/9502, 7247/7249/7250 | 37,711 | mitral, tricuspid, aortic, pulmonary |
| conduction ×3 *(layer)* | 9488/9489/83331, 9486, 9477/9478/9484/9480/9482/9483 | 13,107 | left and right bundle branches, SA/AV nodes with internodal tracts |

**144,220 triangles are drawn by default**; the valve and conduction layers are
user-toggled, for 195,038 loaded in total (3.8 MB GLB, uint16 indices).

**Decimation is set per node, because vertex clustering tears thin walls.** The
merged source meshes are watertight — zero boundary edges — so every hole in the
render came from this step: at a 0.35 keep-ratio the LAD picked up 280 boundary
edges (4.5% of its edges) and the ventricle walls showed visible slits with the
interior glinting through. Measured on the LV free wall, boundary edges by ratio:
0.8 → 74, 0.6 → 166, 0.5 → 167. So the arteries — the point of the view and the
thinnest things in it — are **not decimated at all**, the bulk organ is reduced
to 0.8, and the hardest reductions are reserved for the valve leaflets, which are
modelled at an absurd density (the mitral valve alone is 70,480 triangles raw).

### Feature → anatomy

The viewer offers to show the structure behind a finding. Only links the dataset
supports are offered, and each states its own limits:

| Feature | Structure | Specificity |
| :--- | :--- | :--- |
| `EF-TTE` < 50, `Region RWMA` > 0, `LVH` | left ventricular myocardium | whole ventricle |
| `VHD`, `Systolic Murmur`, `Diastolic Murmur` | the relevant valve group | which valve is unknown |
| `BBB = LBBB` / `RBBB` | left / right bundle branch | **exact** |

Two limits are stated in the UI itself, not only here:

- **`Region RWMA` is a 0–4 grade, not a location** (distribution 217/26/32/14/14). It
  records *how many* regions move abnormally, never *which*. So the highlight covers
  the whole left ventricle and the note says why — the 13 LV free-wall patches in the
  source could tile regions, but nothing in the dataset says which patch is abnormal.
- **`VHD` does not name a valve**, so all four are shown with the grade quoted.

Deliberately **not** mapped: `Q Wave`, `St Elevation`, `St Depression`, `Tinversion`,
`Poor R Progression`. In real ECG reading these localise to a myocardial territory, but
this dataset records presence or absence with no lead information, so lighting up a
wall for them would be invention. They stay in the SHAP panel, which claims nothing
anatomical.

### Stenosis as geometry, not just colour

Each artery carries **glTF morph target 0**, baked at build time and blended on the GPU
by predicted risk. A reader who cannot separate the status hues still sees a stenotic
vessel as a thinner one.

The displacement takes its **direction** from a Laplacian contraction toward the mesh's
own medial axis, and its **distance** from the vessel's measured local radius — half
the distance to the nearest oppositely-facing surface. Splitting the two matters:
contraction distance depends on how dense the mesh is, not on the anatomy, and driving
the magnitude from it collapsed thin segments through their own axis while leaving
thick proximal ones barely changed. Against a true median calibre of 0.59 mm (LAD),
0.75 mm (LCX) and 0.72 mm (RCA), 25 contraction passes already move the average LAD
vertex 1.23 radii — straight through the lumen and out the far side.

Clipped to 45% of the local radius, the shipped result on the real meshes is a
**29–40% calibre reduction** at full influence, with each vessel keeping **98.9–99.5%
of its extent** — it thins without reading as an artery retracting, which would be a
different and wrong claim. The radii the measurement recovers (0.59–0.75 mm median) are
the calibres a coronary angiogram reports, so the number means something outside this
file.

The narrowing is **uniform along the whole vessel**. The dataset labels a vessel
stenotic or not and never says where the lesion sits, so the shape change encodes
severity only and claims nothing about position.

This also replaced the geodesic centreline fit, which was the fragile step in the old
pipeline: on these meshes it returned a 47 mm "left coronary stem" against a true
length under 15 mm.

**Correspondence.** One normalisation transform is applied to every node and recorded
in `vessels.json`. The axes were derived from the data, not assumed: +x is the
patient's left (the LV free wall sits at +x, the right atrium at −x), −y is anterior
(the LAD running the anterior groove is more negative than the posterior LCX), +z is
superior (the heart sits at z ≈ 1175–1293 mm in a body whose origin is near the feet).
The transform's determinant is asserted to be +1, because a mirrored heart renders
perfectly happily and would put the LAD on the wrong side.

The displayed LAD *is* FMA74912 and its branches, so model output and anatomy match by
construction.

**Nine anatomical assertions run before the asset is written**, and the build refuses
to emit a file that fails any: RCA right of both left vessels; LAD anterior to LCX;
aorta superior to the mitral plane; left atrium superior to the left ventricle; LV left
of RV; pulmonary trunk anterior to aorta; left bundle left of right bundle; nodes
superior to both bundles. A wrong heart is still a plausible-looking heart, so this is
checked numerically rather than by eye.

### Rendering

Static model — no beat animation, no auto-rotation. `frameloop="demand"` means
three.js renders only when something changes (camera move, hover, new prediction)
instead of running a 60 fps loop, which is the main reason this stays responsive
without a discrete GPU. No postprocessing, `dpr` capped at 1.5.

**Solid by default.** The previous version drew the whole heart at opacity 0.12–0.22
with `depthWrite` off, which is why it read as a cloud rather than tissue — nothing in
the scene had a surface. Opacity is now only ever reduced by the explicit **X-ray**
control, and even then only for muscle, atrium and great vessels; arteries, valves and
bundles stay opaque, so x-ray reveals them instead of fading everything together.
Selecting a finding that points at an internal structure switches x-ray on, because a
highlight behind opaque muscle is drawn and still invisible.

One failure here is worth recording, because it is not the kind that degrades
gracefully: a mesh whose geometry carries `morphAttributes.position` but whose
`morphTargetInfluences` was never built makes three.js throw **during the draw call**,
which aborts the whole frame — so every mesh queued behind it disappears too. React
Three Fiber attaches geometry after constructing the Mesh, so three's constructor-time
`updateMorphTargets()` finds nothing and each such mesh must call it itself. One
unconverted artery was enough to make all three coronary trees vanish while the
myocardium in front of them kept drawing.

Tissue colour is deliberately **not** from the risk palette, and deliberately low
chroma: a saturated muscle red sits close enough to the High-risk hue that a vessel
stops reading as a separate thing, which is the whole job of this view. The status
palette and the SHAP diverging pair were both run through a palette validator
(lightness band, chroma floor, colour-vision separation, normal-vision floor, contrast
vs surface) in light and dark mode. Status colour never travels alone: every tier ships
with its label and the model's threshold.

### Shell and surfaces

The page is an app shell: a persistent rail carrying the brand, scroll-spy navigation
over the four sections, the loaded patient, and the light/dark switch. The rail
collapses to an 84px icon strip and remembers the choice — 84px because the theme
switch is a fixed 64px wide and still has to fit inside the padding.

Everything rides over a **fixed photographic backdrop** (`src/assets/background.jpg`,
174 KB). The image is filtered (`--backdrop-filter`) and then washed with a scrim
(`--scrim`) before anything is drawn on it; between them the photo keeps its warmth
and loses the contrast that would fight with the data. The surfaces above it are
translucent, which is what lets it read through the whole page — a three-step ladder
of `--page`, `--canvas`, `--surface`.

The backdrop is a **sibling** of the shell, not a child of it. As a child it painted
*over* the page, because a `position: fixed` element paints in the positioned-descendants
layer, above static content like the disclaimer — and the cards survived only because
`.canvas` carries a `backdrop-filter`, which quietly creates its own stacking context.
That made a required safety notice disappear while everything inside the canvas stayed
put.

Contrast over glass is **measured against the real image**, not assumed: the backdrop
is sampled, the filter chain applied numerically, and the scrim, canvas and surface
composited, so every ink/surface pair is checked against the worst pixel in the photo.
That caught two failures — the rail at its first opacity gave 2.37:1 for secondary text
and 1.14:1 for muted (it sits outside `.canvas`, so it gets none of that wash), and
`--ink-muted` fell to 2.74:1 on the cards. The rail went to 0.78 and `--ink-muted`
darkened to `#686a72`, which also fixes a pre-existing failure: on the old opaque white
it was only 3.43:1, under AA for the caption text it carries.

**`--surface` is load-bearing for the 3D viewer and is deliberately never retuned.**
The `<Canvas>` sets no background and react-three-fiber defaults to `alpha: true`, so
the heart renders onto a *transparent* canvas and what sits behind it is `--surface`
showing through the card. Five overlay panels are `color-mix`-ed against it at 90–94%,
and the tissue colours are fixed hex in `lib/anatomy.ts` that cannot adapt. So the
shell layers new tokens around that token rather than changing it, and `.viewer-pin`
re-declares it on the viewer's subtree so a later edit cannot silently alter the scene.

Decorative tints are reserved for cards that carry **no** risk meaning — cohort
position, input coverage, model quality. A tinted surface must never be confusable
with severity, so every risk figure stays on a neutral card with the status palette
doing the colour. Label text on a tint is `--ink-secondary`: `--ink-muted` measures
2.77–2.89:1 there and fails AA.

Two contrast defects were found and fixed while tokenising the palette. `#b07600`,
used as warning text at four sites and defined nowhere, is 3.86:1 on white — under AA
for the 11px text carrying it; `--warn-ink` is 4.61:1 at the same hue. And white on
`--accent` was 4.42:1 in light and **2.83:1 in dark**, hardcoded as `#fff` at six
sites; light darkens the accent to 5.35:1 and dark inverts to near-black ink on the
pale accent at 6.57:1.

### Dashboard

Beside the canvas: the CAD headline with its decision threshold marked in place and
the vessel-derived cross-check; three vessel cards that double as the 3D selector;
SHAP contributions as a diverging bar list and a waterfall from cohort baseline to the
patient's margin; physiological measurements paired with their contribution share; a
patient editor driven by `GET /schema`; and the model card reading `GET /metrics`.

SHAP panels are labelled as **log-odds margin space** — the bars sum to the model's
margin, not to the displayed percentage. The two are related by a monotonic
calibration map, so the ranking is identical but the units are not, and the UI says so
instead of implying the bars add up to the risk score.

---

## Architecture

```text
ml/
  data/z_alizadeh_sani.xlsx     # raw cohort
  src/train.py                  # nested CV, calibration, artifact export
  src/evaluate.py               # metrics tables + ROC/PR/calibration/confusion figures
  src/validate.py               # clinical detection report (see "Verifying detection")
  notebooks/                    # 01 EDA · 02 cross-validated training · 03 TreeSHAP
  results/                      # cv_metrics.json, oof_predictions.npz, *.png
heart-beat parts/               # BodyParts3D heart5.0i source OBJs (129 files)
tools/
  build_anatomy.py              # BodyParts3D parts -> anatomy.glb + manifest
backend/
  app/main.py                   # FastAPI routes
  app/services.py               # inference engine: loads artifacts, real TreeSHAP
  app/schemas.py                # response contract (single source of truth)
  app/core/config.py            # settings, CORS origins
  app/models/                   # serialized artifacts (written by train.py)
  tests/                        # 14 integrity tests
frontend/
  public/models/                # anatomy.glb, vessels.json (generated, committed)
  src/api/                      # typed client + contract mirroring schemas.py
  src/lib/risk.ts               # risk tiers and the validated status palette
  src/components/viewer/        # Canvas, heart shell, vessel tubes, legend
  src/components/dashboard/    # CAD summary, vessel cards, SHAP, physiology, model card
```

### Artifacts written by `train.py`

| File | Contents |
| :--- | :--- |
| `cardiac_models.joblib` | 4 sklearn `Pipeline`s (impute → scale/one-hot → variance → LGBM) |
| `calibration.joblib` | per-target 1-D calibrators, plus one for the derived CAD score |
| `thresholds.json` | tuned and default thresholds with the selection rule |
| `metadata.json` | feature order, dtypes, categorical levels, inference defaults |
| `demo_presets.json` | 4 real cohort rows covering 0/1/2/3-vessel disease |

Preprocessing lives **inside** each pipeline, so there is no separate preprocessor
artifact to drift out of sync.

---

## API

| Endpoint | Purpose |
| :--- | :--- |
| `GET /health` | reports whether the trained artifacts actually loaded |
| `GET /metrics` | CV metrics read from `ml/results/cv_metrics.json` on disk |
| `GET /presets` | 4 real patient rows with ground truth |
| `GET /cohort` | all 303 training patients: truth + **held-out** probabilities |
| `GET /cohort/{id}` | one patient's 55 features, by id (`ROW006`) or index (`6`) |
| `GET /schema` | feature order, categorical levels and defaults, to build a form |
| `POST /predict` | CAD + per-vessel risk, thresholds, TreeSHAP attributions |

`/metrics` reads from disk rather than returning literals, so the API cannot report
numbers that differ from what training measured.

`POST /predict` takes features keyed exactly as the dataset spells them, e.g.
`EF-TTE`, `Typical Chest Pain`, `Region RWMA`. Underscored aliases (`EF_TTE`) are
accepted for compatibility. Omitted features fall back to the training median/mode
and are listed back in `defaulted_features`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Age": 62, "Typical Chest Pain": 1, "EF-TTE": 40, "Region RWMA": 2}}'
```

---

## Quick start

**Prerequisites:** Python 3.11+ and Node 18+.

```bash
# 1. Train (writes artifacts into backend/app/models/)
pip install -r ml/requirements.txt
python ml/src/train.py

# 2. Figures and the evaluation report
python ml/src/evaluate.py

# 3. API
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
# docs at http://127.0.0.1:8000/docs

# 4. Integrity tests
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -v

# 5. Is it detecting disease correctly?
python ml/src/validate.py
python ml/src/validate.py --live        # with the API running

# 6. 3D anatomy assets (reads heart-beat parts/; no download)
python tools/build_anatomy.py
python tools/build_anatomy.py --report   # source statistics only
python -m pytest tools/test_build_anatomy.py

# 7. Cohort browser data, and optional offline OCR
python tools/build_cohort.py
python tools/fetch_ocr_assets.py        # optional: ~18 MB, only for scanned images

# 8. Dashboard
cd frontend
npm install
npm test                                # parser + completeness guards
npm run dev                             # http://localhost:5173
```

`npm install` resolves cleanly on React 19 / Vite 8 — the `--legacy-peer-deps` flag
an earlier version of this README suggested is **not** needed. Set `VITE_API_BASE` in
`frontend/.env` if the API is not on `http://127.0.0.1:8000` (see `.env.example`).

Run every command from the repository root. `train.py` resolves paths relative to its
own location and accepts `--data` / `--out-models` / `--out-results` overrides.

### Docker

Build from the repository root so the image can carry `ml/results/`:

```bash
docker build -f backend/Dockerfile -t cardiovision-api .
docker run -p 8000:8000 cardiovision-api
```

### Windows note

If `import` fails with *"An Application Control policy has blocked this file"*, Smart
App Control is blocking unsigned `.pyd` extension modules. It blocked both venv and
user-site installs on the development machine for some wheels. Workarounds: install
into a location your policy permits, or disable Smart App Control in Windows Security
→ App & browser control (note that it cannot be re-enabled afterwards without
reinstalling Windows). This is why the serving path depends on LightGBM's built-in
TreeSHAP rather than the `shap` package.

---

## Deployment

The two halves deploy separately, and they have to: the frontend is a static Vite
bundle, while the API loads LightGBM, scikit-learn, scipy and pandas to serve a
prediction. Installed unzipped that dependency set runs past Vercel's serverless
function size limit, so **the backend does not go on Vercel.** Put it on a host that
runs a long-lived container - Render, Railway, Fly.io, a VM - using
`backend/Dockerfile`, which already exists and builds from the repository root.

### Frontend on Vercel

`vercel.json` at the repository root configures the whole build, so importing the
repo needs no dashboard settings:

```json
{
  "buildCommand": "npm --prefix frontend run build",
  "installCommand": "npm --prefix frontend ci",
  "outputDirectory": "frontend/dist"
}
```

Set one environment variable in the Vercel project, for every environment you intend
to use:

| Variable | Value |
| :--- | :--- |
| `VITE_API_BASE` | the deployed API origin, e.g. `https://cardiovision-api.onrender.com` |

It is read at `src/api/client.ts:15` and **baked in at build time**, not fetched at
runtime - Vite inlines `import.meta.env` during the build. Changing it therefore
requires a redeploy, not just a restart.

Without it the bundle falls back to `http://127.0.0.1:8000`, and a browser on an
`https://` page will refuse the request as mixed content. The dashboard still loads;
every prediction fails.

### Then point the API back at the frontend

`cors_origins` in `backend/app/core/config.py` lists localhost only. The deployed
site is a different origin, so the browser will block it until that host sets:

```bash
CORS_ORIGINS='["https://your-project.vercel.app"]'
```

Vercel gives every preview deployment its own generated subdomain, so if you want
previews to work too, add the production domain you have assigned rather than
relying on the per-deployment URL.

### What is deliberately not deployed

`.vercelignore` keeps `heart-beat parts/` (28 MB of BodyParts3D source meshes),
`backend/`, `ml/` and `tools/` out of the upload. The meshes were already consumed
once by `tools/build_anatomy.py` to produce `frontend/public/models/anatomy.glb`,
which **is** committed - so a clone builds the 3D scene with no download step.

`frontend/public/ocr/` is gitignored (19 MB of third-party tesseract WASM and
language data). Its absence is handled rather than fatal: `hasLocalOcrAssets()` in
`src/lib/extract/readDocument.ts` probes for it and tesseract.js falls back to its
CDN, with the UI saying which is about to happen. Run `python
tools/fetch_ocr_assets.py` if you want OCR to stay offline.

**Do not add a catch-all SPA rewrite.** There is no router in this app - navigation
is scroll-based over one document - so nothing needs it, and a
`"/(.*)" -> "/index.html"` rule would actively break things: the OCR probe above is
a `HEAD` request that would start returning `200` with an HTML body, so the app
would believe local assets are present and then hand tesseract an HTML page instead
of a WASM binary.

---

## Reproducibility

- `random_state=42` throughout; `--seed` to change it.
- Dependency versions are pinned in both requirements files, and the ML pins are kept
  identical across them, because joblib artifacts are sensitive to scikit-learn and
  lightgbm minor versions. **Retrain after changing a pin.**
- `python -m pytest backend/tests` asserts the properties that matter: no leakage, the
  served probability equals `pipeline.predict_proba` on the same row, TreeSHAP
  additivity, threshold consistency, and that `/metrics` carries no hardcoded numbers.

---

## Credits

- **Dataset:** Z-Alizadeh Sani CAD dataset, UCI Machine Learning Repository.
- **Anatomical meshes:** BodyParts3D, © The Database Center for Life Science,
  licensed under **CC Attribution 4.0 International**. Attribution appears in the
  dashboard footer, in `frontend/public/models/ATTRIBUTION.txt`, and in the
  `anatomy.glb` asset's own copyright field.
  Source: the BodyParts3D heart part set, compatibility version `heart5.0i`, 129 OBJ
  files supplied locally in `heart-beat parts/` — see
  [lifesciencedb.jp/bp3d](https://lifesciencedb.jp/bp3d/) and
  [dbarchive.biosciencedbc.jp](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html).

  **Worth confirming before you publish.** These files carry **no embedded licence
  notice of their own** — their headers record only File ID, Representation ID, Concept
  ID, English name, bounds and volume — so the credit above rests on the BodyParts3D
  licence page, which states CC Attribution 4.0 International (page last updated
  2025-02-27). Note that the separately distributed PART-OF archive *does* carry an
  older embedded notice reading *"CC Attribution-Share Alike 2.1 Japan"*. If ShareAlike
  were to apply to this set too it would oblige derivative works to be licensed alike,
  so confirm with DBCLS if that distinction matters for how you release this.
  `frontend/public/models/ATTRIBUTION.txt` records the provenance of every structure,
  down to the source filename.
