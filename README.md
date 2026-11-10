
# VLM-Based Image Restoration

### Instruction-Guided, Degradation-Aware Image Restoration with DA-CLIP

A project on restoring degraded images using Vision-Language Models (VLMs). The pipeline first **identifies which degradations are present**, restores the image with **DA-CLIP** for each detected type, and then **combines** the results into a single clean output. **AutoDIR** is studied as a related all-in-one restoration framework.

| Metric (DIV2K, Kaggle, 800 images) | Value |
|---|---|
| PSNR | ~31 dB |
| SSIM | ~0.9 |
| LPIPS | ~0.07 |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Models and Frameworks](#3-models-and-frameworks)
4. [Project Workflow](#4-project-workflow)
5. [Degradation Types](#5-degradation-types)
6. [Dataset](#6-dataset)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results](#8-results)
9. [Proposed Improvements over DA-CLIP](#9-proposed-improvements-over-da-clip)
10. [Technologies Used](#10-technologies-used)
11. [Repository Structure](#11-repository-structure)
12. [Running the Project](#12-running-the-project)
13. [Limitations](#13-limitations)
14. [Future Work](#14-future-work)
15. [Team Members](#15-team-members)
16. [References](#16-references)
17. [Acknowledgements](#17-acknowledgements)

---

## 1. Project Overview

Images captured in the real world often suffer from several problems at once: noise, motion blur, haze, low light, rain, snow, shadows or JPEG compression. Most restoration models are trained for one task, so they fail when the damage is unknown or mixed.

This project uses Vision-Language Models to understand *what is wrong* with an image and then restore it. It builds on **DA-CLIP** (degradation-aware CLIP) and studies **AutoDIR** for comparison.

## 2. Objectives

- Explore Vision-Language Models for image degradation understanding.
- Detect which degradation types are present in an image.
- Restore degraded images using degradation-aware conditioning.
- Handle images with more than one degradation.
- Evaluate restoration quality with PSNR, SSIM and LPIPS.

## 3. Models and Frameworks

### DA-CLIP

DA-CLIP adapts a pretrained CLIP model for restoration in two stages:

1. **Degradation-aware encoder.** CLIP stays frozen. A trainable *controller* looks at the degraded image and produces a **content embedding** (what is in the scene) and a **degradation embedding** (what is wrong with it). Both are trained with a contrastive loss against text embeddings.
2. **Restoration network.** The content embedding is injected through cross-attention and the degradation embedding through a prompt module. The original paper uses the IR-SDE diffusion model as the restorer.

Repository: https://github.com/Algolzw/daclip-uir

### AutoDIR

AutoDIR is an all-in-one restoration framework. A vision-language stage detects the dominant degradation, and a latent-diffusion stage restores it step by step. It is used here as a related method, and its degradation classification and restoration stages were explored through custom DIV2K-based data preparation and fine-tuning experiments.

Repository: https://github.com/jiangyitong/AutoDIR

## 4. Project Workflow

```
Degraded image
      │
      ▼
┌───────────────────────────┐
│ 1. Degradation-type       │   Which degradations are present?
│    finder                 │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 2. DA-CLIP restoration    │   One restoration pass per detected type
│    (per degradation)      │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 3. Combine                │   Merge the per-type outputs
└─────────────┬─────────────┘
              ▼
         Restored image
              ▼
   Evaluation (PSNR, SSIM, LPIPS)
```

1. **Input image:** a degraded image, possibly with several degradations.
2. **Degradation-type finder:** a model predicts which degradation types are present.
3. **DA-CLIP per type:** each detected degradation is handled by DA-CLIP with the matching degradation condition.
4. **Combine:** the per-type results are merged into one output.
5. **Evaluation:** the output is compared with the clean reference using image quality metrics.

## 5. Degradation Types

The experiments use the ten DA-CLIP degradation types:

| | | |
|---|---|---|
| Motion blur | Haze | JPEG compression |
| Low light | Noise | Raindrop |
| Rain | Shadow | Snow |
| Missing regions (uncompleted) | | |

## 6. Dataset

### DIV2K

DIV2K provides high-resolution clean images used as ground truth.

The experiments involve:
- High-resolution ground-truth images (800 images, Kaggle version).
- Synthetic generation of degraded versions for each degradation type.
- Cropping and resizing.
- Paired degraded and clean images.
- Text descriptions of each degradation (for example "a noisy photo") for conditioning.

Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

The dataset is not included in this repository.

## 7. Evaluation Metrics

| Metric | Direction | Meaning |
|---|---|---|
| **PSNR** | higher is better | Pixel-level closeness to the clean image, in dB |
| **SSIM** | higher is better (0 to 1) | Similarity of brightness, contrast and structure |
| **LPIPS** | lower is better | Perceptual distance using deep features |

PSNR rewards pixel accuracy and LPIPS rewards realism. The two can disagree, so all three are reported.

## 8. Results

Evaluated on the DIV2K dataset from Kaggle (800 images):

| PSNR | SSIM | LPIPS |
|---|---|---|
| ~31 dB | ~0.9 | ~0.07 |

An earlier prototype that combined a CLIP detector with a conditioned NAFNet restorer averaged about 27 dB PSNR across the ten degradation types. The two numbers come from different test protocols, so they are not directly comparable.

## 9. Proposed Improvements over DA-CLIP

Standard DA-CLIP has three limits: degradation is **categorical** (no severity), its embeddings are **global** (no spatial guidance for local damage such as rain or shadow), and its **diffusion restorer** trades some PSNR for realism.

The following changes target these limits. They build on published ideas, and the contribution is their combination for multi-degradation, selective restoration.

| Idea | What it adds |
|---|---|
| Per-type detection before restoration | Each degradation is handled by a matching condition instead of a single global one. |
| Severity-aware conditioning | A head predicts severity (noise level, JPEG quality, blur length). The labels come free from the synthetic generator. |
| Multi-scale spatial conditioning | Patch-level features are injected at several U-Net levels, so local damage is handled locally. |
| Learned combine step | A small network replaces a fixed merge rule and reduces error build-up. |
| Regression restorer trained with L1 | Optimises PSNR directly. The diffusion model can stay as an optional texture step. |

> Items beyond per-type detection are proposed and should be validated with ablations (add one block at a time and report PSNR, SSIM and LPIPS).

## 10. Technologies Used

- Python
- PyTorch
- OpenAI CLIP / OpenCLIP
- DA-CLIP
- AutoDIR
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-image
- LPIPS
- Jupyter Notebook
- Kaggle

## 11. Repository Structure

```text
VLM-project/
│
├── notebooks/
│   ├── new-da-clip.ipynb
│   ├── daclip (3).ipynb
│   └── autodir.ipynb
│
├── results/
│   ├── input_images/
│   ├── restored_images/
│   └── comparisons/
│
├── README.md
└── requirements.txt
```

The notebooks contain model setup, data preparation, restoration experiments and evaluation.

## 12. Running the Project

Clone the repository:

```bash
git clone https://github.com/pratik5200/VLM-project.git
cd VLM-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then:

1. Download DIV2K, or attach a **DIV2K High Resolution Images** dataset on Kaggle (full-size `.png` files).
2. Open the notebooks in Jupyter, VS Code or Kaggle. On Kaggle, set the accelerator to **GPU** and **Internet** to **On**.
3. Set the dataset and checkpoint paths before running.
4. Run the cells from top to bottom.

For long runs on Kaggle, use **Save Version → Save & Run All (Commit)** so that training continues in the background and checkpoints are kept.

## 13. Limitations

- Heavy haze and missing regions are the hardest cases and may look smoother than the rest.
- A wrong degradation type sends the image to the wrong restorer and lowers PSNR directly.
- Results are measured on synthetic degradations. Real degraded images (for example Rain100L, SOTS, LOL) should be tested separately.
- AutoDIR was trained on fewer degradation types than DA-CLIP, so it may not handle every one of the ten types here.

## 14. Future Work

- Add severity-aware and multi-scale conditioning, then run ablations.
- Add an instruction layer so users can type requests such as "improve quality" or "remove the rain".
- Choose the removal order per image, since the order affects the result.
- Add an optional diffusion refinement step for more realistic texture.
- Compare against PromptIR, InstructIR and AutoDIR on the same validation images.
- Prepare the work for submission to a conference or journal on all-in-one restoration.

## 15. Team Members

| S. No. | Name |
|---|---|
| 1 | Rajat Panchal |
| 2 | Shashikaant Paandey |
| 3 | Pratik Jaisal |
| 4 | Vardan Singh |

## 16. References

1. Luo, Z., Gustafsson, F. K., Zhao, Z., Sjölund, J., Schön, T. B. *Controlling Vision-Language Models for Multi-Task Image Restoration.* ICLR 2024. arXiv:2310.01018.
2. Potlapalli, V., Zamir, S. W., Khan, S., Khan, F. S. *PromptIR: Prompting for All-in-One Blind Image Restoration.* NeurIPS 2023. arXiv:2306.13090.
3. Conde, M. V., Geigle, G., Timofte, R. *InstructIR: High-Quality Image Restoration Following Human Instructions.* ECCV 2024. arXiv:2401.16468.

## 17. Acknowledgements

We acknowledge the original researchers and contributors of DA-CLIP, AutoDIR and the DIV2K dataset for providing the foundational models, frameworks and resources used in this project.

---

**Project Domain:** Artificial Intelligence and Machine Learning | Computer Vision | Vision-Language Models | Image Restoration
