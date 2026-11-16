<div align="center">

# Computer Vision Final Project

# Squat Form Analyzer

[![GitHub](https://img.shields.io/badge/github-yeyorg/proyecto--vision-blue?logo=github)](https://github.com/yeyorg/proyecto-vision)

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD)](https://xgboost.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-30-3FB950?logo=pytest&logoColor=white)](tests/)

*[Leer en español](README.es.md)*

</div>

> **English summary**
>
> Computer vision tool that evaluates squat technique from an uploaded video. YOLOv8-pose detects 17 body keypoints per frame, which are turned into biomechanical angles for the knees, hips, back and left-right symmetry, so that each execution receives a score from 0 to 100 against ranges from the biomechanics literature together with specific corrective feedback. The team also recorded and labeled its own video dataset to train an XGBoost classifier on the angle features, and the solution was delivered as a Streamlit app supported by unit tests.
>
> **My role:** As project lead, I defined the approach and divided the tasks across the team, reviewed each contribution and kept track of changes across branches, and I also built the XGBoost training notebook on the extracted angle features.
>
> *Original repository: [yeyorg/proyecto-vision](https://github.com/yeyorg/proyecto-vision).*

> Analyze your squat technique by uploading a video:
> YOLOv8-pose detects keypoints, computes biomechanical angles
> and evaluates them against optimal ranges with interpretable rules.

---

## Table of contents

- [Description](#description)
- [Recent changes](#recent-changes)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Useful commands](#useful-commands)
- [Useful commands (Windows without make)](#useful-commands-windows-without-make)
- [Model](#model)
- [Extending the project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Description

This project analyzes squat form in video using **YOLOv8-pose**
to detect 17 body keypoints, computes specific **biomechanical angles**
(knee, hip, back, symmetry) and evaluates them against optimal ranges
based on the biomechanics literature.

Unlike purely ML-based approaches that need hundreds of
labeled videos, this classifier uses **interpretable rules** that
work from the very first moment. Each metric has specific feedback
("knee angle: 125 -- you did not reach parallel").

### What it evaluates

| Criterion | Weight | What it measures |
| --- | --- | --- |
| Depth | 2x | Minimum knee angle, ideal ~90 |
| Back angle | 1x | Torso inclination relative to vertical |
| Knee tracking | 1x | Horizontal displacement of the knee over the ankle |
| Symmetry | 1x | Difference between the left and right sides |
| Stability (range) | 0.5x | Full range of motion |

---

## Recent changes

### Removed: real-time mode and webcam detection

All live analysis functionality was removed to simplify the project
to a single flow: **upload video → analyze**.

| What was removed | Why |
|---|---|
| `realtime.py` (443 lines) | OpenCV window with webcam, phase detection and live scoring |
| `SquatDetector` + `SquatPhase` | State machine that classified phases (standing/descending/bottom/ascending), used only by `realtime.py` |
| Webcam in `app.py` | "Webcam (foto)" option and "Tiempo Real" buttons in the sidebar |
| `make realtime` | Makefile target |

The project now focuses exclusively on analyzing **recorded videos**
(uploaded by the user), without depending on the camera or on real-time processing.

### Current flow

```
You upload an MP4 video → YOLOv8-pose detects keypoints →
biomechanical angles are computed per frame →
statistics are aggregated (mean, std, min, max) →
SquatFormClassifier evaluates with rules →
score 0-100 + personalized feedback
```

---

## Architecture

```
     video (uploaded MP4)
           |
           v
 +------------------+
 |  PoseExtractor   |  YOLOv8-pose -> 17 COCO keypoints
 +------------------+
           |
           v
 +------------------+
 |  get_squat_angles |  Computes angles per frame
 +------------------+   (knee, hip, back, symmetry)
           |
           v
 +------------------+
 | aggregate_video   |  Statistics: mean, std, min, max
 | _features         |
 +------------------+
           |
           v
 +------------------+
 | SquatForm         |  Biomechanical rules -> score + feedback
 | Classifier        |
 +------------------+
           |
           v
    score 0-100 + tips
```

### Modules

| File | Responsibility |
| --- | --- |
| `src/pose_extractor.py` | YOLOv8-pose wrapper. Processes frames and videos. |
| `src/angle_utils.py` | Geometric calculations: angles, aggregated features, explanations. |
| `src/squat_classifier.py` | Rule-based classifier. `predict()` / `predict_proba()` interface. |
| `app.py` | Streamlit application with video upload and visualization. |
| `squat_form.ipynb` | Interactive notebook for exploration and testing. |
| `tests/test_classifier.py` | Unit tests of the classifier with synthetic data. |

---

## Evaluation rules

The `SquatFormClassifier` evaluates **5 criteria** with different weights.
Each criterion uses a different statistic (`min`, `mean`, `max`) according to
what makes biomechanical sense for that metric.

The **overall score** is a weighted average:

```
overall = sum(score_i * weight_i) / sum(weight_i)
```

A score >= 60 = good form, < 60 = needs work.

### 1. Depth — weight 2x

Uses the **MINIMUM** knee angle reached throughout the video
(`knee_angle_min`). The minimum is used because the deepest moment
of the squat is what defines whether you reached parallel or not.

| Minimum angle | Score | Meaning |
|---|---|---|
| ~90° (ideal) | 100 | Perfect parallel |
| 70°–110° | 40–100 | Within the acceptable range |
| < 70° | 0–40 | Hyperflexion / butt wink |
| > 110° | 0–40 | You did not reach parallel |

### 2. Back angle — weight 1x

Uses the **AVERAGE** torso inclination (`back_angle_mean`).
The average is used because the inclination should be consistent
throughout the exercise; isolated peaks are not representative.

| Inclination | Score | Meaning |
|---|---|---|
| ~35° (ideal) | 100 | Normal range for a high-bar squat |
| 15°–50° | 40–100 | Acceptable |
| < 15° | 0–40 | Too upright, loss of mechanical advantage |
| > 50° | 0–40 | Lower-back risk, excessive inclination |

### 3. Knee tracking (knee over ankle) — weight 1x

Uses the **ABSOLUTE MAXIMUM** horizontal displacement
(`knee_toe_x_max` and `knee_toe_x_min`). The peak value is used because
a single time the knee goes too far forward is already an
injury risk, which the average would not capture.

| Max displacement | Score | Meaning |
|---|---|---|
| < 40 px | 70–100 | Controlled |
| 40–60 px | 30–70 | Went a little too far |
| > 60 px | 0–30 | Knee at risk |

### 4. Symmetry — weight 1x

Uses the **AVERAGE** difference between legs
(`knee_symmetry_mean`, `hip_symmetry_mean`). A sustained difference
indicates muscle imbalance or compensation.

| Asymmetry | Score | Meaning |
|---|---|---|
| < 5° | ~95 | Balanced movement |
| 5°–12° | 70–95 | Mild asymmetry |
| > 12° | 0–70 | Significant imbalance |

### 5. Stability (range of motion) — weight 0.5x

Uses the knee's **RANGE** of motion
(`knee_angle_max - knee_angle_min`). It measures whether you are doing the
full range of a squat or only half a repetition.

| Range | Score | Meaning |
|---|---|---|
| > 60° | 100 | Full squat |
| 40°–60° | 70–100 | Acceptable range |
| 20°–40° | 30–70 | Half squat |
| < 20° | 0–30 | Almost no flexion |

### Concrete example

For a video where:
- Minimum knee angle: 88° → **Depth = 95**
- Average back angle: 32° → **Back = 85**
- Max displacement: 25 px → **Knee = 80**
- Average asymmetry: 4° → **Sym = 90**
- Range of motion: 65° → **Stability = 100**

```
overall = (95×2 + 85×1 + 80×1 + 90×1 + 100×0.5) / (2 + 1 + 1 + 1 + 0.5)
       = (190 + 85 + 80 + 90 + 50) / 5.5
       = 495 / 5.5
       = 90  ✅ Good form
```

---

## Requirements

- **Python** >= 3.12
- **uv** >= 0.5 (install it with `pip install uv` or from <https://docs.astral.sh/uv/>)
- Optional but recommended: **Jupyter** for the notebook
- Optional: **make** (Linux/macOS) or **Chocolatey** (Windows)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yeyorg/proyecto-vision.git
cd proyecto-vision

# 2. Create a virtual environment and install dependencies
uv venv
uv sync
```

Done. There is no need to activate the venv -- `uv run` uses it automagically.
If you prefer to activate it: `.venv\Scripts\activate` (Windows) or
`source .venv/bin/activate` (Linux/macOS).

---

## Usage

### Streamlit app

```bash
uv run streamlit run app.py
```

It opens in the browser. Upload an MP4 video to analyze.

### Notebook

```bash
uv run jupyter notebook squat_form.ipynb
```

Interactive exploration: try the pose extraction, compute angles, evaluate
synthetic squats.

### Tests

```bash
uv run --group dev pytest tests/ -v
```

Verifies that the rule-based classifier works correctly
using synthetic data of good and bad squats.

---

## Useful commands

```bash
make help       # Shows all available commands
make run        # Starts the Streamlit app
make notebook   # Opens the notebook
make test       # Runs the classifier unit tests
make check      # Checks that everything works
make clean      # Cleans generated files
make deep-clean # Cleans everything including .venv
```

---

## Useful commands (Windows without make)

```powershell
# Start the app
uv run streamlit run app.py

# Run tests
uv run --group dev pytest tests/ -v

# Verify the installation
uv run python -c "from src.pose_extractor import PoseExtractor; print('OK')"

# Reinstall dependencies if pyproject.toml changes
uv sync

# Add a new dependency
uv add <package>

# Clean up
Remove-Item -Recurse -Force __pycache__, src\__pycache__, .ipynb_checkpoints -ErrorAction SilentlyContinue
```

---

## Model

The saved model (`models/squat_form_model.pkl`) is a `SquatFormClassifier`
that implements the scikit-learn interface:

```python
import joblib

model = joblib.load("models/squat_form_model.pkl")

# features: dict with angle statistics
prediccion = model.predict([features])       # 0 = good, 1 = bad
probabilidad = model.predict_proba([features])  # [[good_prob, bad_prob]]
```

### Why rules instead of XGBoost?

The original notebook (`exercise_form.ipynb`) uses XGBoost trained on labeled
data. The problem: **without data, there is no model**. Our approach:

| Advantage | Biomechanical rules | XGBoost |
| --- | --- | --- |
| Works without data | YES | NO |
| Interpretable feedback | "knee at 125" | "class 1" |
| Adjustable | Edit thresholds | Retrain |
| Size | 0.3 KB | ~MB |

**Next step:** If you collect labeled videos, the features from
`angle_utils.py` can be used directly to train an XGBoost. The `predict()`
interface is identical -- not a single line of the app changes.

---

## Extending the project

### Add a new exercise (e.g., deadlift)

1. In `angle_utils.py`, create `get_deadlift_angles()`
2. In `squat_classifier.py`, create `_score_deadlift_*()` or a new classifier
3. In `app.py`, add an exercise selector

### Adjust thresholds

```python
from src.squat_classifier import SquatFormClassifier

clf = SquatFormClassifier(thresholds={
    "knee_angle_min_range": (75.0, 105.0),  # stricter
    "good_form_threshold": 70.0,            # higher cutoff
})
```

### Train XGBoost with real data

```python
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load features from your labeled videos
# X = np.array([...])  # features of each video
# y = np.array([...])  # 0 = good, 1 = bad

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = XGBClassifier()
# model.fit(X_train, y_train)
# joblib.dump(model, "models/xgboost_model.pkl")
```

---

## Troubleshooting

### "No module named 'ultralytics'"

```bash
uv sync
```

If the package is not in `pyproject.toml`:

```bash
uv add ultralytics
```

### "No se detectaron personas en el video" (no people detected in the video)

- Make sure you are facing the camera or in profile
- The video must have good lighting
- Try the notebook to see whether YOLO detects keypoints

### The bad-form score is too high/low

Adjust the thresholds in `squat_classifier.py`:

```python
DEFAULT_THRESHOLDS = {
    "good_form_threshold": 60.0,  # raise to 70 to be stricter
    "knee_angle_min_range": (70.0, 110.0),
    "back_angle_mean_range": (15.0, 50.0),
}
```

Run `uv run --group dev pytest tests/ -v` to verify that the classifier works correctly.

---

## License

Project based on the Science Buddies notebook
"[Using AI to Detect Proper Exercise Form](https://www.sciencebuddies.org/science-fair-projects/project-ideas/ArtificialIntelligence_p027/artificial-intelligence/exercise_form)".
For personal and educational use.

---

*Last updated: 2026-06-02*
