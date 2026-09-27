# Tri-Hybrid XAI-IDS

## Explainable Intrusion Detection System Using Machine Learning, Deep Learning, Meta-Fusion and Multi-Explainer XAI

---

## 1. Project Overview

Tri-Hybrid XAI-IDS is an explainable intrusion detection system designed to detect and analyze malicious network traffic using a combination of:

1. Machine Learning ensemble models
2. Deep Learning sequence modeling
3. Meta-learning based fusion
4. Explainable AI techniques
5. Cross-explainer/model agreement
6. Interactive Streamlit visualization

The system combines multiple modeling approaches rather than relying on a single classifier.

The primary objective is not only to classify network traffic but also to provide interpretable explanations for the predictions.

---

# 2. Core Architecture

The system follows a multi-branch architecture:

```text
                         Network Traffic
                                |
                                v
                    +-----------------------+
                    | Data Preprocessing    |
                    | Cleaning / Encoding   |
                    | Scaling / Balancing   |
                    +-----------+-----------+
                                |
                +---------------+---------------+
                |                               |
                v                               v
       +------------------+             +----------------------+
       | ML Branch        |             | DL Branch            |
       |                  |             |                      |
       | Random Forest    |             | CNN                  |
       | SVM              |             | LSTM                 |
       | XGBoost          |             | Attention            |
       +--------+---------+             +----------+-----------+
                |                                  |
                | ML probabilities                 | DL probabilities
                +----------------+-----------------+
                                 |
                                 v
                       +--------------------+
                       | Meta-Learner       |
                       | Logistic Regression|
                       +---------+----------+
                                 |
                                 v
                         Final Prediction
                                 |
                +----------------+----------------+
                |                |                |
                v                v                v
              SHAP             LIME           Attention
                \                |                /
                 \               |               /
                  +--------------+--------------+
                                 |
                                 v
                       XAI / Agreement Layer
                                 |
                                 v
                       Streamlit Dashboard