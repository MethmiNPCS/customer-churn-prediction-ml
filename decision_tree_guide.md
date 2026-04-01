# Decision Tree Model Guide

This guide explains everything you need to know about your code, prepares you for the Viva, and shows you how to connect your machine learning model to a React/HTML frontend.

---

## 1. Full Code Understanding

Here is a breakdown of what exactly happens in your `decision_tree_model.py` script:

1. **Preprocessing (Steps 1-3):** You load the raw data, drop the meaningless `customerID`, and drop rows where `TotalCharges` is missing. You then use `LabelEncoder()` to seamlessly turn all text columns (like "Month-to-month") into numbers ($0, 1, 2$) so the algorithm can understand them natively.
2. **Train/Test Split:** You reserve 20% of the dataset as unseen "Test" data (`X_test`), preserving the exact churn ratio using `stratify=y`.
3. **The Base Model (Steps 4-5):** You train a basic `DecisionTreeClassifier`. Critically, you use `class_weight='balanced'` to force the model to pay extra attention to the minority `Churn=1` class, avoiding the pitfall where the model just predicts "No Churn" for everyone. 
4. **Visualisations (Steps 6-9):** You generate the Confusion Matrix (showing True Positives vs False Negatives), the ROC-AUC Curve (measuring distinction capability), the tree diagram, and Feature Importances (which columns drive Churn).
5. **Overfitting Analysis:** The code loops depths from 1 to 20 to prove that an unrestricted decision tree mathematically overfits to 100% training accuracy while failing on test data.
6. **Hyperparameter Tuning (Step 10):** Instead of guessing the best tree settings, `GridSearchCV` tries **90** different combinations of `max_depth`, `min_samples_split`, and `criterion` using 5-Fold Cross-Validation, finding the ultimate mathematically perfect parameters to prevent overfitting while maximising ROC-AUC score.

---

## 2. Viva Questions & Cheat Sheet

These are the most likely questions your examiner will ask you during the Viva presentation:

**Q1: Why did you use `class_weight='balanced'`?**
*Answer:* Because the dataset suffers from class imbalance (approx 73% No Churn vs 27% Churn). If I didn't balance the weights, the Decision Tree would be heavily biased towards predicting "No Churn" for everyone just to achieve 73% fake accuracy. Balancing it penalises the model heavily for missing an actual churner, which improves our Recall.

**Q2: Your accuracy is around 74%, why isn't it higher?**
*Answer:* Total Accuracy is a misleading metric here. The most important metric to a telecom business is **Recall for Churners** ( catching people before they leave ). By sacrificing a little total accuracy using balanced weights, we boosted our Recall to ~73%, meaning we successfully identified roughly 3 out of every 4 actual churners. Missing a churner costs real revenue, so this trade-off is completely intentional. 

**Q3: How does the Decision Tree algorithm choose where to split?**
*Answer:* It calculates the 'Gini impurity' (or Entropy) for every possible feature. Gini impurity measures the probability of misclassifying a random element in that node. It picks the feature that results in the purest child nodes (the lowest Gini impurity).

**Q4: What is overfitting and how did you resolve it?**
*Answer:* Overfitting is when the model grows too deep and basically memorises the training data, performing terribly on unseen data. I proved this with the Depth-vs-Accuracy plot. To fix it, I used `GridSearchCV` to automatically find the best `max_depth` (pruning the tree) and `min_samples_split` to stop the tree from growing unnecessary branches.

---

## 3. How to Connect the Model to a Frontend

To use this model in a real frontend web application, you need to transition from "training" the model to "serving" the model. 

### Step 1: Export (Save) the Trained Model
First, you must save your trained `best_dt` model into a `.pkl` file (a pickled binary file). I have updated your python script to automatically do this! (See `decision_tree_model.pkl`).

### Step 2: Create a Backend API (FastAPI / Flask)
You cannot connect a Frontend (React/HTML) directly to a `.pkl` file. You need a Python server to load the `.pkl` file and listen for API requests. 

Create a file called `app.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# 1. Load the model during server startup
model = joblib.load("decision_tree_model.pkl")

# 2. Define exactly what the frontend should send you
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: int # 0=Month-to-month, 1=One year, 2=Two year
    # ... add the other 15 scaled features here

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert JSON to a format sklearn understands (DataFrame)
    input_df = pd.DataFrame([data.dict()])
    
    # Run the model!
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0][1])
    }
```
*Run it using: `uvicorn app:app --reload`*

### Step 3: Call the API from your Frontend (React/Vanilla JS)
On your website, you can take user input from a form, and `fetch()` the prediction from your python backend API:

```javascript
async function checkCustomerChurn() {
    const customer = {
        tenure: 12,
        MonthlyCharges: 85.50,
        TotalCharges: 1020.00,
        Contract: 0 // Month-to-month
        // ... include remaining properties
    };

    const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(customer)
    });

    const result = await response.json();
    
    if (result.churn_prediction === 1) {
        alert("WARNING: This customer has a " + (result.churn_probability * 100) + "% chance of churning!");
    } else {
        alert("Safe! Customer is likely to stay.");
    }
}
```
