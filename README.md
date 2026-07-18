## R50F25 — Multimodal Stroke Severity Detection 

**Using Vision Transformer (ViT) and Wav2Vec2 + LSTM**

Department of Computer Science — DHA Suffa University
Final Year Project (FYP)

**Live Dashboard:** https://fyp-mssd-final-iatxf7nwdfhni3vmm9kuyp.streamlit.app/ *(it's our deployed `streamlit.app` URL)*

---

## Overview

This project is a two-stage, multimodal clinical decision-support pipeline for stroke assessment. It first screens brain imaging (CT/MRI) for the presence of a stroke, and — only if a stroke is detected — analyzes a patient's voice recording to grade the **severity** of the stroke as **Low**, **Medium**, or **High**. An interactive Streamlit dashboard reports real evaluation results (confusion matrices, accuracy, per-class metrics) alongside an explainability layer that helps a clinician understand *why* the model made a given prediction.

## Objectives

- Leverage a **Vision Transformer (ViT)** to accurately classify stroke vs. non-stroke cases from medical imaging, enabling rapid preliminary diagnosis.
- Utilize **Wav2Vec2 + LSTM** to analyze patient voice recordings and classify stroke severity into Low, Medium, or High categories.
- Integrate the visual and audio diagnostic pathways into a unified system to assist clinicians with faster, more reliable stroke severity evaluation.

## Methodology — Multimodal Fusion

The system runs two independent branches whose outputs are combined through a simple fusion rule: **the severity branch only executes if the imaging branch detects a stroke.**

```
                 ┌────────────────────┐        ┌─────────────────────┐
                 │   Branch A — ViT   │        │  Branch B — LSTM     │
                 │       (MRI)        │        │     (Voice)          │
                 └────────────────────┘        └─────────────────────┘
                           │                              │
      MRI input → preprocessing              Voice input → MFCC / wav2vec2 features
      (resize, normalize, patch)             (resample, trim, extract features)
                           │                              │
              ViT-B/16 encoder                    LSTM encoder
        (12 layers, multi-head attention)     (hidden-128, 4-gate mechanism)
                           │                              │
        Output: Stroke / Non-Stroke               Output: Severity class
                                                    (Low / Medium / High)
                           │                              │
                           └───────────────┬──────────────┘
                                            │
                                     Fusion logic
                        (if stroke detected → run severity module)
                                            │
                                  Final combined output
                            (Diagnosis + Severity Report)
```

## Two-Stage Pipeline

| Stage | Model | Task | Input |
|---|---|---|---|
| **Stage 1** | Vision Transformer (ViT) | Stroke / Non-stroke classification | CT / MRI scan |
| **Stage 2** | Wav2Vec2 + LSTM | Severity classification (Low / Medium / High) | Voice recording (only runs if Stage 1 is stroke-positive) |

## Results

| Metric | Stage 1 — ViT | Stage 2 — Wav2Vec2 + LSTM | Combined Multimodal |
|---|---|---|---|
| **Accuracy** | 71.4% | 44.4% | 67.5% |
| **Sensitivity (stroke recall)** | ~75% | — | — |
| **Confusion Matrix** | TP 33, TN 56, FP 24, FN 11 | See dashboard (3×3, Low/Medium/High) | — |

- The ViT model's accuracy is derived from a confusion matrix of 124 test images (56 true non-stroke, 33 true stroke correctly classified, with 24 false positives and 11 false negatives).
- The severity model was evaluated on a substantially larger speech test set and reaches its best validation accuracy (44.4%) at early stopping; validation loss plateaus early while validation accuracy stays low and noisy — indicating the model is struggling to learn a strong signal rather than classically overfitting. Class imbalance and low Medium-class recall are flagged as key areas for future improvement.
- The **combined multimodal accuracy (67.5%)** reflects the effect of chaining both stages: only stroke-positive cases from Stage 1 are passed on to Stage 2 for severity grading.

## Interactive Dashboard

The project includes a Streamlit dashboard (`app.py`) built for the FYP defense, with the following tabs:

- **Overview** — key stats, accuracy gauges, and both confusion matrices at a glance.
- **Architecture** — visual breakdown of the ViT and Wav2Vec2 + LSTM pipelines, plus dataset snapshot.
- **Stage 1 — ViT** — full classification report, confusion matrix, Sankey flow of true→predicted classes, and a patch-attention explainability panel (with an uploader to drop in a real Grad-CAM export).
- **Stage 2 — Severity** — classification report, confusion matrix (counts and normalized recall), training curves, and acoustic feature-importance chart (with an uploader for real SHAP / permutation importance results).
- **Explainability** — a radar chart of per-class precision/recall/F1, a prediction-confidence distribution, and a walkthrough of an example end-to-end decision.
- **Combined Pipeline** — an end-to-end accuracy estimate and a simulated patient funnel (how many of 1,000 patients are correctly screened and graded, given an assumed stroke prevalence).

Two kinds of visuals appear throughout the dashboard, and are labeled accordingly:
- 🟢 **Derived from evaluation data** — computed directly from the real confusion matrices / training logs.
- 🟡 **Illustrative example** — a conceptual demonstration of what an explainability technique (e.g. Grad-CAM, SHAP) would show, clearly marked as a placeholder until replaced with real computed output via the sidebar uploaders.

### Running the dashboard locally

```bash
pip install streamlit pandas numpy plotly
streamlit run app.py
```

## Tools & Tech Stack

- **Python**
- **PyTorch**
- **TensorFlow**
- **Scikit-learn**
- **Streamlit** (dashboard / deployment)
- **Plotly** (interactive charts)
- **Pandas / NumPy**
-  **keras**

## Project Structure

```
.
├── app.py                 # Streamlit dashboard (results, architecture, explainability)
├── notebooks/              # ViT and wav2vec2+LSTM training notebooks
├── data/                   # Imaging and speech datasets (not included — see dataset notes)
└── README.md
```

## Disclaimer

This system is a research prototype developed for academic purposes as part of a Final Year Project. It is **not** a certified medical device and is not intended for clinical use or real patient diagnosis. Illustrative explainability visuals in the dashboard are conceptual placeholders and should not be interpreted as verified model outputs unless explicitly labeled "Derived from evaluation data."

## Team

**Group Leader**
Tanzila Khalid (DS221012)

**Members**
- Fiza Anwar Khetani (DS221017)
- Nimarta Kumari (DS221007)
- Aashir Ahmed (CS221049)

**Supervisor:** Soohan Abbasi
**Co-Supervisor:** Hafsa Asif

Department of Computer Science, DHA Suffa University
