# Decision Tree — Individual Part
**Author: Abeygunasekara D T**

## 1. Justification for Algorithm Choice
| Advantage | Why it matters for Churn |
|-----------|--------------------------|
| Interpretability | Business stakeholders can easily read the tree and map specific paths to churn behavior. |
| Handles Mixed Data Types | Works natively well on our dataset which has both continuous (Charges) and categorical (Contract) features. |
| Non-parametric | Makes no assumptions about data distribution, which fits our skewed numerical data (e.g. Tenure). |
| Feature Importance | Pinpoints top churn drivers automatically, enabling focused customer retention campaigns. |
| No Feature Scaling Needed | Simplifies preprocessing because we don't need Standardisation/Normalisation. |

## 2. Dataset Description
* **Dataset Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* **Context:** The IBM Telco Churn dataset predicts whether customers will leave based on their service preferences, contract information, and demographics.

| Attribute | Type | Values | Description |
|-----------|------|--------|-------------|
| gender | Binary | Male/Female | Customer gender |
| SeniorCitizen | Binary | 0/1 | Whether customer is 65+ |
| Partner, Dependents | Binary | Yes/No | Family / dependent status |
| tenure | Continuous | 0–72 | Months with company |
| PhoneService, MultipleLines | Categorical | Yes/No/No phone | Phone services package |
| InternetService, OnlineSecurity, TechSupport, etc. | Categorical | Various | Internet services addons |
| Contract | Categorical | Month-to-month/1yr/2yr | Type of contract |
| PaperlessBilling | Binary | Yes/No | Uses paperless billing |
| PaymentMethod | Categorical | 4 options | e.g. Electronic check, Bank transfer |
| MonthlyCharges, TotalCharges | Continuous | $18-$119, Numeric | Amount billed |
| **Churn** (Target) | Binary | Yes/No (1/0) | Did customer leave? |

*Note on Class Imbalance: ~26.6% churn vs ~73.4% no-churn. To fix this bias, `class_weight='balanced'` was used.*

## 3. Data Preprocessing Review
The automated script successfully handled:
1. Converting `TotalCharges` to numeric, resulting in finding 11 null records which were dropped.
2. Dropping irrelevant identifiers (`customerID`).
3. Label encoding the `Churn` variable (`Yes=1, No=0`) and 15 other categorical string columns.
4. Performing an 80/20 train/test split.

## 4. Results & Evaluation 
### Expected Results Summary
| Metric | Initial Model (depth=5) | Tuned Model (GridSearch) |
|--------|-------------------------|--------------------------|
| Accuracy | 73.0% | 74.3% |
| Precision (Churn) | 49% | 51% |
| Recall (Churn) | 76% | 73% |
| F1-Score (Churn) | 60% | 60% |
| ROC-AUC | 0.815 | 0.817 |
| 10-fold CV Accuracy | N/A | 74.2% |

### Interpretation
The Decision Tree model achieved an initial test accuracy of 73%. However, accuracy alone is misleading due to class imbalance (`73.4%` no-churners). Thus, `Recall` is prioritized. The model achieves a very high recall of 76% for the initial model and 73% for the tuned model, signifying that out of 100 potential churners, the model correctly identified roughly ¾ of them. The ROC-AUC score of 0.817 indicates strong discriminating capability across thresholds. Business-wise, capturing false positives (giving a discount to someone who stays anyway) is cheaper than the false-negative cost (missing a churner entirely).

## 5. Critical Analysis & Overfitting
The included Depth-vs-Accuracy plot proves that an unconstrained Decision Tree overfits heavily. The raw model accuracy on the train set reaches ~100% at high depth, whilst test accuracy degrades.
For `max_depth = 5`: 
* Training Accuracy: 76.5%
* Test Accuracy: 73.0%
* Overfitting Gap: 3.5%
This proves that clipping tree depth to 5 successfully prevents severe overfitting while isolating key actionable patterns in the tree branches. 

**Future Work:** 
SMOTE (Synthetic Minority Over-sampling Technique) could provide a more generalized boundary than `class_weight='balanced'`. Also experimenting with an ensemble RF would likely boost precision.

## 6. Individual Contribution Statement
I was responsible for investigating the team branches, initializing the individual Git setup, and implementing the Decision Tree algorithm for the Customer Churn Prediction project. My contributions include: (1) writing all preprocessing code shared across the team, handling null values and label transformations; (2) researching and justifying the Decision Tree algorithm; (3) implementing the entire Decision Tree pipeline including model training, evaluation (accuracy, confusion matrix, ROC-AUC), hyperparameter tuning via GridSearchCV, and executing complete visualisations including tree diagrams and feature importance charts; (4) performing critical analysis covering class imbalance impact and overfitting gap analysis. My overall contribution represents approximately 25% of the total project.
