
# Interformer: Interpretable Large Time Series Model For Concept Drift Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based_Systems-B31B1B.svg)](#) > **🔔 Important Note to Reviewers (Knowledge-Based Systems):**
> Welcome to the official repository for **Interformer**. Please note that while our codebase builds upon the robust LTSM pretrain-finetune data pipeline initially introduced by [Timer](https://github.com/thuml/Large-Time-Series-Model) to ensure standardized pretraining on UTSD, **this repository contains the novel and distinct implementation of Interformer**. 
> 
> Specifically, the core architectural innovations—including the **Interpretable Encoder (Intercoder)** and the **Residual-Focused Cross-Attention Decoder**—are entirely original to this work. These custom modules are explicitly defined and can be found in the `models/` and `layers/` directories.

## 📖 Overview

Data stream analysis faces severe challenges from **concept drift**, where the underlying data distribution shifts over time, degrading model performance. Many existing Large Time Series Models (LTSMs) lack the structural interpretability required to explicitly decouple informative drift from stochastic noise.

**Interformer** addresses this by introducing a pretrain-finetune architecture explicitly designed for interpretable stream forecasting. By leveraging the Universal Time Series Dataset (UTSD), Interformer learns universal temporal representations and adapts them to highly non-stationary downstream tasks.

### ✨ Key Contributions
* **Intercoder (Interpretable Encoder):** Forces the time series into a strict "Season-Trend-Residual" format. It utilizes truncated Fourier series for seasonality and polynomial functions for the trend envelope, acting as a structural bottleneck to adaptively filter high-frequency uninterpretable noise.
* **Residual-Focused Decoder:** Forecasts the isolated residual component by integrating historical residuals (via cross-attention) and horizon-specific residuals (via self-attention).
* **Robustness against Concept Drift:** Achieves consistently lower Mean Absolute Percentage Error (MAPE) across abrupt, gradual, recurrent, and sudden drift scenarios compared to state-of-the-art baselines.

---

## 🏗️ Architecture

![Interformer Architecture](methodology.png)  


---

## 🚀 Getting Started

### 1. Environment Setup

For optimal performance and dependency resolution, we recommend utilizing `micromamba` for environment management and `uv` for lightning-fast package installation.

```bash
# Create and activate the environment
micromamba create -n interformer python=3.10 -c conda-forge
micromamba activate interformer
pip install -r requirements.txt

```



---



## 📊 Main Results Summary

Interformer demonstrates superior zero-shot generalization and conceptual robustness on six real-world datasets, effectively reducing the average MSE by **10.44%** relative to the best competing LTSM model. Most notably, on profoundly stochastic datasets such as **DeepMIMO** and **ECL**, the Intercoder maintains exceptional structural stability where conventional online learning methods explicitly fail to converge.

---

## 🤝 Acknowledgements

We sincerely thank the authors of [Timer](https://github.com/thuml/Large-Time-Series-Model) for providing the foundational data-loading framework and UTSD curation. Their open-source contributions significantly accelerated the development of this research.

## 📝 Citation

If you find this framework or the proposed Extended Temporal Concept Drift (ETCD) formulations useful for your research, please consider citing our paper:

```text
@article{lin2026interformer,
  title={Interformer: Interpretable Large Time Series Model For Concept Drift Adaptation},
  author={Lin, Borong and Jin, Nanlin and Zhu, Xiaohui and Grasso, Floriana},
  journal={Knowledge-Based Systems},
  year={2026},
  publisher={Elsevier}
}
```