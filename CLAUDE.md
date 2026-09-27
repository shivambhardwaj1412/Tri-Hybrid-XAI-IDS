\# Tri-Hybrid-XAI-IDS — Claude Project Instructions



\## Project Overview



This project is a software-based Explainable AI Intrusion Detection System (XAI-IDS).



The system combines:



1\. Machine Learning ensemble

&#x20;  - Random Forest

&#x20;  - SVM

&#x20;  - XGBoost



2\. Deep Learning branch

&#x20;  - CNN

&#x20;  - LSTM

&#x20;  - Attention mechanism



3\. Meta-level fusion

&#x20;  - Logistic Regression meta-learner

&#x20;  - Combines ML and DL probability outputs



4\. Explainable AI

&#x20;  - SHAP

&#x20;  - LIME

&#x20;  - Attention-based explanations

&#x20;  - Cross-explainer/model agreement



5\. Dashboard

&#x20;  - Streamlit



6\. Supported datasets

&#x20;  - CICIDS2017

&#x20;  - NSL-KDD

&#x20;  - UNSW-NB15



\---



\## Important Project Rule



Do NOT fabricate results, metrics, datasets, model performance, experiments, screenshots, or implementation details.



When writing the thesis, paper, documentation, or analysis:



\- Use the actual project files.

\- Use actual result files whenever available.

\- Use actual training logs and experiment outputs.

\- Clearly distinguish implemented functionality from planned functionality.

\- If evidence is unavailable, explicitly state that it is unavailable.

\- Never invent a value just to complete a table or paragraph.



\---



\## Dataset Isolation



Each dataset must remain isolated.



Expected structure:



data/

\- processed\_cicids2017/

\- processed\_nsl\_kdd/

\- processed\_unsw\_nb15/



models/

\- cicids2017/

\- nsl\_kdd/

\- unsw\_nb15/



results/

\- cicids2017/

\- nsl\_kdd/

\- unsw\_nb15/



Do not mix artifacts between datasets.



\---



\## Current Configuration



The main configuration is controlled through:



model\_config.yaml



Important parameters include:



\- sequence window size: 10

\- sequence stride: 1

\- ML models:

&#x20; - Random Forest

&#x20; - SVM

&#x20; - XGBoost

\- DL:

&#x20; - CNN

&#x20; - LSTM

&#x20; - Attention

\- meta learner:

&#x20; - Logistic Regression

\- XAI:

&#x20; - SHAP

&#x20; - LIME

&#x20; - Attention

\- agreement:

&#x20; - Jaccard top-k

\- federated configuration:

&#x20; - 4 nodes

&#x20; - 2 local epochs

&#x20; - 5 rounds



Always inspect the actual configuration file before describing implementation details.



\---



\## Dataset-Specific Results



\### CICIDS2017



Actual experiments include:



\- ML ensemble

\- chronological ML training

\- CNN-LSTM-Attention DL branch

\- meta fusion

\- XAI

\- Streamlit inference/dashboard



Known verified metrics include:



ML-only accuracy:

0.9982986389111289



DL-only accuracy:

0.9777822257806245



Fused accuracy:

0.9982986389111289



Fused macro-F1:

0.597601694086235



Do not assume that fusion improves every metric.



\---



\### NSL-KDD



The project contains:



\- 23-class classification

\- ML ensemble

\- CNN-LSTM-Attention DL branch

\- global class-space meta learner

\- XAI

\- dataset-specific inference



Known approximate verified results:



ML accuracy:

\~0.9967



DL accuracy:

\~0.8955



Fused accuracy:

\~0.9962



Fused macro-F1:

\~0.5189



When exact values are available in result files, use those instead of these approximate values.



\---



\### UNSW-NB15



The project contains:



\- 10-class classification

\- categorical encoding

\- chronological splitting

\- SMOTE-Tomek

\- SHAP-based feature selection

\- CNN-LSTM-Attention

\- meta fusion

\- XAI



Important:



UNSW-NB15 fusion currently does NOT outperform the ML-only branch.



Verified full endpoint-target evaluation:



ML-only accuracy:

0.7536035897593509



DL-only accuracy:

0.5005685834588315



Fused accuracy:

0.668008728524449



Fused macro-F1:

0.4571874223893305



These results must be reported honestly.



Do NOT use earlier restricted-population/purity-filtered UNSW results as final performance results.



\---



\## Architecture



The intended high-level flow is:



Dataset

→ preprocessing

→ feature engineering / encoding

→ chronological split

→ ML branch

→ DL branch

→ meta fusion

→ prediction

→ XAI explanations

→ agreement analysis

→ Streamlit dashboard



For the ML branch:



Input

→ preprocessing

→ SHAP feature selection where enabled

→ Random Forest

→ SVM

→ XGBoost

→ probability ensemble



For the DL branch:



Input sequence

→ CNN

→ LSTM

→ Attention

→ class probabilities



For fusion:



ML probabilities + DL probabilities

→ meta learner

→ final prediction



\---



\## Inference



The inference pipeline is dataset-aware.



Important model artifacts include:



\- ml\_ensemble\_chrono.joblib

\- ml\_ensemble.joblib

\- dl\_model\_chrono.keras

\- dl\_model.keras

\- dl\_model.h5

\- meta\_learner.joblib



The inference system should select artifacts from the appropriate dataset directory.



Never load a model from another dataset directory simply because the filename matches.



\---



\## Thesis Instructions



The thesis should follow the provided CGU thesis/report format.



Use the base research paper as a reference for:



\- scientific presentation

\- figure style

\- table organization

\- architecture presentation

\- result presentation

\- confusion matrices

\- performance comparison



Do NOT copy the base paper's wording, claims, results, or experimental conclusions.



The thesis must describe the actual implementation of this project.



Recommended thesis visuals include:



\- overall system architecture

\- complete workflow

\- preprocessing workflow

\- ML architecture

\- CNN-LSTM-Attention architecture

\- meta-fusion architecture

\- XAI architecture

\- dataset class distributions

\- SHAP feature importance

\- confusion matrices

\- ML vs DL vs fusion comparison

\- training curves where actual histories exist

\- XAI comparison

\- agreement analysis

\- Streamlit dashboard screenshots



Only create a figure when its content can be supported by the project.



\---



\## Coding Rules



When modifying the project:



1\. Preserve existing functionality unless the task explicitly requires changing it.

2\. Prefer complete replacement files when making substantial changes.

3\. Keep dataset-specific paths isolated.

4\. Do not hard-code one dataset into a generic pipeline.

5\. Keep configuration in YAML/config files where appropriate.

6\. Do not silently change experimental methodology.

7\. Do not overwrite verified result files without explicit instruction.

8\. Test modified Python files where practical.

9\. Report errors honestly.



\---



\## Results Rules



Before reporting a metric:



1\. Check the actual result file.

2\. Identify the dataset.

3\. Identify the evaluation population.

4\. Identify whether the result is ML, DL, or fusion.

5\. Check whether the experiment is comparable with other datasets.

6\. Do not mix experimental versions.



If two experiments use different evaluation populations or methodologies, clearly label them as different experiments.



\---



\## Important UNSW-NB15 Warning



Several UNSW experiments were conducted during development.



Some experiments used restricted or filtered evaluation populations and produced apparently higher performance.



Those results are NOT the final comparable results.



Use the full endpoint-target evaluation for the final UNSW-NB15 thesis results unless a newer verified experiment explicitly replaces it.



\---



\## Large Files



Large datasets and trained model binaries should generally remain outside GitHub unless Git LFS or another appropriate storage mechanism is intentionally configured.



Do not modify .gitignore merely to force large datasets/models into the repository.



The repository should primarily contain:



\- source code

\- configuration

\- documentation

\- thesis material

\- result CSV files

\- plots

\- small supporting files



\---



\## Working Style



When asked to modify or analyze the project:



1\. Inspect the relevant files first.

2\. Understand the current implementation.

3\. Make the smallest necessary change.

4\. Preserve existing behavior.

5\. Verify the change.

6\. Clearly report what changed.



For thesis work, prioritize evidence from the actual project over assumptions or generic descriptions.

