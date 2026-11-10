# VLM-Based Image Restoration

### Image Restoration Using Vision-Language Models

A project exploring image restoration using Vision-Language Models (VLMs), focusing on degradation-aware image understanding and restoration using **DA-CLIP** and **AutoDIR**.

## 1. Project Overview

Images captured in real-world environments often suffer from different types of degradation, including noise, motion blur, haze, low illumination, rain, JPEG compression and low resolution.

This project explores Vision-Language Models for image degradation understanding and image restoration.

It investigates DA-CLIP for degradation-aware visual-language representations and AutoDIR for image restoration.

## 2. Objectives

- Explore Vision-Language Models for image degradation understanding.
- Investigate degradation-aware visual and textual representations.
- Restore degraded images using image restoration frameworks.
- Experiment with different degradation types and restoration prompts.
- Evaluate restoration quality using quantitative image quality metrics.

## 3. Models and Frameworks

### DA-CLIP

DA-CLIP is a degradation-aware adaptation of CLIP designed for understanding image degradations.

It uses visual and textual representations to capture degradation-related information and supports image restoration tasks through degradation-aware conditioning.

Repository: https://github.com/Algolzw/daclip-uir

### AutoDIR

AutoDIR is an image restoration framework that explores degradation understanding and restoration.

Our experiments investigate its degradation classification and restoration stages, along with custom DIV2K-based data preparation and fine-tuning experiments.

Repository: https://github.com/jiangyitong/AutoDIR

## 4. Project Workflow

1. **Input Image:** Provide a degraded image as input.
2. **Degradation Understanding:** Explore degradation-aware visual and textual representations.
3. **Prompt Conditioning:** Use textual descriptions to represent degradation and guide restoration experiments.
4. **Image Restoration:** Apply the restoration framework to generate an enhanced image.
5. **Evaluation:** Compare degraded and restored images using image quality metrics.

## 5. Degradation Types

The experiments explore several degradation categories:

- Motion Blur
- Haze
- JPEG Compression
- Low Light
- Noise
- Raindrops
- Rain
- Shadows
- Snow
- Low Resolution

Synthetic degradation generation is also explored using the DIV2K dataset.

## 6. Dataset

### DIV2K Dataset

DIV2K is used for image restoration experiments.

The experiments involve:
- High-resolution ground-truth images.
- Generation of degraded image samples.
- Image cropping and resizing.
- Creation of paired degraded and clean images.
- Textual degradation descriptions for experimental conditioning.

Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

The dataset is not included in this repository.

## 7. Evaluation Metrics

- **PSNR:** Measures reconstruction quality. Higher values generally indicate better reconstruction fidelity.
- **SSIM:** Measures structural similarity between restored and reference images.
- **LPIPS:** Measures perceptual similarity using deep feature representations. Lower values indicate greater perceptual similarity.

## 8. Technologies Used

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

## 9. Repository Structure

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

The notebooks contain the experimental implementation, model setup, data preparation, restoration experiments and evaluation procedures.

## 10. Running the Project

Clone the repository:

```bash
git clone https://github.com/pratik5200/VLM-project.git
cd VLM-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and configure the DIV2K dataset.

Open the notebooks using Jupyter Notebook, VS Code or Kaggle.

Configure the dataset and model checkpoint paths before execution.

## 11. Team Members

| S. No. | Name |
|---|---|
| 1 | Rajat Panchal |
| 2 | Shashikaant Paandey |
| 3 | Pratik Jaisal |
| 4 | Vardan Singh |

## 12. Acknowledgements

We acknowledge the original researchers and contributors of DA-CLIP, AutoDIR and the DIV2K dataset for providing the foundational models, frameworks and resources used in this project.

---

**Project Domain:** Artificial Intelligence and Machine Learning | Computer Vision | Vision-Language Models | Image Restoration
