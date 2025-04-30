# 🧠 Evaluating Uncertainty Estimation in High-Stakes AI

## 🔍 Project Overview

This project explores how **uncertainty estimation techniques** can improve the reliability and interpretability of AI models deployed in **high-stakes domains** like **healthcare** and **finance**.

Despite strong performance, traditional deep learning models tend to be **overconfident**, especially in ambiguous or rare scenarios. We aim to address this by applying three powerful uncertainty-aware methods:

- **Monte Carlo Dropout (MC Dropout)**
- **Bayesian Neural Networks (BNNs)**
- **Deep Ensembles**

These techniques are evaluated on two critical real-world datasets:

- **ISIC 2018 Skin Cancer Classification** (medical imaging)
- **Credit Card Fraud Detection** (highly imbalanced tabular data)

---

## 🎯 Objectives

- Assess model calibration using **Expected Calibration Error (ECE)** and **AUROC**
- Compare how each uncertainty estimation technique improves model **confidence alignment**
- Explore **selective prediction** and **deferral strategies** for safer decision-making

---

## 📊 Datasets

| Dataset | Description | Domain |
|--------|-------------|--------|
| [ISIC 2018](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) | Over 10,000 dermoscopic images of skin lesions across 9 classes | Healthcare |
| [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 transactions with only 492 fraud cases (extreme imbalance) | Finance |

---

## 🧪 Methods & Models

### 1. Baseline Models
- **ResNet-50** and **EfficientNet** for image classification
- **Random Forest** and **XGBoost** for tabular data

### 2. Uncertainty Estimation Techniques
| Technique | Description |
|----------|-------------|
| MC Dropout | Adds randomness at inference by enabling dropout |
| Bayesian NN | Models weights as probability distributions |
| Deep Ensembles | Averages predictions from independently trained models |

---

## 📈 Key Results

| Dataset | Model | AUROC ↑ | ECE ↓ |
|--------|-------|---------|--------|
| ISIC | Deep Ensemble | **0.91** | **0.06** |
| ISIC | MC Dropout | 0.88 | 0.18 |
| ISIC | BNN | 0.89 | 0.09 |
| Fraud | Deep Ensemble | **0.981** | **0.0067** |
| Fraud | MC Dropout | 0.972 | 0.0108 |
| Fraud | BNN | 0.969 | 0.0115 |

> ✅ **Deep Ensembles** consistently offered the best trade-off between accuracy and calibration.

---

## ⚙️ Implementation Highlights

- Applied **SMOTE** and **undersampling** to handle class imbalance in fraud data
- Used **transfer learning** for CNN-based models (ImageNet-pretrained weights)
- Created **reliability diagrams** to visualize calibration
- Designed models to **abstain on low-confidence predictions**

---

## 💡 Conclusion

Incorporating uncertainty estimation improves both performance and safety in AI systems used for critical decisions. These techniques can guide when to defer to human expertise, making AI more **trustworthy** and **deployable** in real-world scenarios.

---

## 📁 Project Structure

- credit_card_fraud_detection.ipynb
- isic_skin_cancer_detection.ipynb
