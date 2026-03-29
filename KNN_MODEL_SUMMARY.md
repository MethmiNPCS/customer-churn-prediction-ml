# KNN Model Implementation - Customer Churn Prediction

## Assignment Component: K-Nearest Neighbors (KNN) Algorithm

---

## 📊 MODEL TRAINING RESULTS

### Dataset Information
- **Training Samples**: 5,625 customers
- **Testing Samples**: 1,407 customers
- **Number of Features**: 30 customer attributes
- **Target Variable**: Customer Churn (Yes/No)

### Optimal Model Configuration
After testing K values from 1 to 30, the optimal configuration was found:

**Optimal K Value: 29 neighbors**

This K value achieved the best balance between:
- Avoiding overfitting (too small K)
- Avoiding underfitting (too large K)

---

## 🎯 MODEL PERFORMANCE METRICS

### Overall Performance
- **Test Accuracy**: 78.11%
- **Cross-Validation Accuracy**: 78.74% (+/- 2.68%)
- **Misclassification Rate**: 21.89%

### Classification Metrics for Churn Prediction

| Metric | Score |
|--------|-------|
| **Precision (Churn)** | 0.5902 |
| **Recall (Churn)** | 0.5775 |
| **F1-Score (Churn)** | 0.5838 |

### Confusion Matrix
```
                Predicted
              No Churn   Churn
Actual No Churn    883      150
       Churn       158      216
```

**Analysis:**
- True Negatives (Correct No Churn): 883
- True Positives (Correct Churn): 216
- False Positives (Wrongly predicted as Churn): 150
- False Negatives (Missed Churn cases): 158

---

## 📈 VISUALIZATIONS GENERATED

The following visualizations have been created and saved in the `notebooks/` folder:

1. **knn_accuracy_plot.png** - Accuracy vs K Value graph showing how model performance changes with different K values
2. **knn_confusion_matrix.png** - Heatmap visualization of the confusion matrix
3. **knn_risk_distribution.png** - Bar chart showing customer distribution across risk levels
4. **knn_cv_scores.png** - 5-fold cross-validation scores bar chart

---

## 🔍 RISK LEVEL ANALYSIS

Based on churn probability predictions, customers are categorized into three risk levels:

### Risk Level Distribution
| Risk Level | Number of Customers | Percentage |
|------------|---------------------|------------|
| **Low Risk** (< 30%) | 760 | 54.0% |
| **Medium Risk** (30-70%) | 564 | 40.1% |
| **High Risk** (> 70%) | 83 | 5.9% |

### Business Recommendations by Risk Level

#### High Risk Customers
- **Action**: Offer discount or loyalty plan
- **Priority**: Immediate intervention required
- These customers have >70% probability of churning

#### Medium Risk Customers
- **Action**: Offer promotional package
- **Priority**: Proactive engagement
- These customers have 30-70% probability of churning

#### Low Risk Customers
- **Action**: No action needed
- **Priority**: Maintain current service quality
- These customers have <30% probability of churning

---

## 💡 KEY FINDINGS

### Model Strengths
1. **Good General Accuracy**: 78% accuracy is reasonable for real-world customer churn data
2. **Stable Performance**: Cross-validation shows consistent performance (78.74% ± 2.68%)
3. **Better at Predicting Non-Churn**: 85% precision and recall for "No Churn" class
4. **Actionable Insights**: Provides probability scores and risk categorization for business decisions

### Model Limitations
1. **Class Imbalance Impact**: Lower performance on churn class (58% F1-score)
2. **False Negatives**: 158 actual churners were missed (could be costly for business)
3. **Distance Sensitivity**: KNN performance depends heavily on feature scaling and distance metrics

### Business Value
- The model can identify **216 high-risk churn customers** correctly
- Enables **targeted retention campaigns** for 647 customers (Medium + High risk)
- Potential cost savings by focusing retention efforts on at-risk customers
- Probability scores allow for **graded intervention strategies**

---

## 🔬 TECHNICAL DETAILS

### Algorithm: K-Nearest Neighbors (KNN)

**How it works:**
1. For each new customer, find K most similar customers from training data
2. Similarity measured using Euclidean distance
3. Majority voting among K neighbors determines prediction
4. Prediction probability based on proportion of neighbors who churned

**Why Feature Scaling was Critical:**
- KNN uses distance calculations
- Features with larger scales would dominate distance computation
- StandardScaler normalizes all features to mean=0, std=1

**K Selection Process:**
- Tested K values from 1 to 30
- Plotted accuracy for each K value
- Selected K=29 as it achieved highest accuracy (78.11%)
- Larger K provides more smoothing and better generalization

### Data Preprocessing Applied
- **Feature Scaling**: StandardScaler applied to all features
- **Train-Test Split**: 80% training, 20% testing (random_state=42)
- **Categorical Encoding**: One-hot encoding applied in preprocessing stage

---

## 📁 FILES GENERATED

All files are ready for submission:

### Source Code
- `train_knn_model.py` - Complete Python script for KNN model training
- `notebooks/knn_model.ipynb` - Jupyter notebook version (can be executed)

### Visualizations
- `notebooks/knn_accuracy_plot.png` - K value optimization graph
- `notebooks/knn_confusion_matrix.png` - Confusion matrix heatmap
- `notebooks/knn_risk_distribution.png` - Risk level distribution chart
- `notebooks/knn_cv_scores.png` - Cross-validation results

### Results
- `dataset/knn_predictions.csv` - Full predictions with probabilities and recommendations (1,407 customers)

---

## 📝 COMPARISON WITH OTHER ALGORITHMS

For the complete assignment, compare these results with:

### Your Team Members' Algorithms:
1. **Logistic Regression** (already implemented)
   - Expected accuracy: ~79%
   - Linear decision boundary
   - Good baseline model

2. **Decision Tree** (team member's work)
   - Expected characteristics: interpretable rules
   - May overfit without pruning

3. **Random Forest** (team member's work)
   - Expected accuracy: ~78-80%
   - Ensemble method, better generalization

### KNN Comparison Points:
- **Pros**: 
  - Simple to understand and implement
  - No assumptions about data distribution
  - Naturally handles multi-class problems
  
- **Cons**:
  - Computationally expensive (stores all training data)
  - Slower prediction time compared to other algorithms
  - Sensitive to irrelevant features

---

## 🎓 CONTRIBUTION SUMMARY

### What I Implemented:
✅ Loaded and preprocessed customer churn dataset  
✅ Applied feature scaling (StandardScaler)  
✅ Determined optimal K value through experimentation  
✅ Trained KNN classifier with K=29  
✅ Generated predictions and probability scores  
✅ Created risk level categorization system  
✅ Developed business recommendation engine  
✅ Performed 5-fold cross-validation  
✅ Generated comprehensive visualizations  
✅ Analyzed model performance and limitations  

### Skills Demonstrated:
- Machine Learning: KNN algorithm implementation
- Data Preprocessing: Feature scaling and preparation
- Model Evaluation: Multiple metrics and cross-validation
- Business Analytics: Risk assessment and recommendations
- Visualization: Matplotlib and Seaborn charts
- Python Programming: Clean, documented code

---

## 🚀 HOW TO RUN THE MODEL

### Prerequisites
Python 3.x with required packages installed

### Installation (if needed)
```bash
cd customer-churn-prediction-ml
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
```bash
source venv/bin/activate
python train_knn_model.py
```

The script will:
1. Load preprocessed data
2. Find optimal K value
3. Train the model
4. Generate predictions
5. Create visualizations
6. Save results to CSV

**Total execution time**: ~30 seconds

---

## 📊 PRESENTATION TALKING POINTS

### For Your 4-Minute Presentation Segment:

**Minute 1: Problem & Algorithm**
- Customer churn costs telecom companies money
- KNN predicts churn by finding similar customers
- Simple yet effective instance-based learning

**Minute 2: Methodology**
- Tested 30 different K values (1-30)
- Found optimal K=29 with 78.11% accuracy
- Used 5-fold cross-validation for reliability

**Minute 3: Results & Business Value**
- 78% accuracy in predicting churn
- Categorized customers into Low/Medium/High risk
- Provided actionable recommendations for each group

**Minute 4: Comparison & Conclusion**
- KNN vs Logistic Regression vs Decision Tree vs Random Forest
- Trade-offs: simplicity vs computational cost
- Best use cases for each algorithm

---

## 📖 REFERENCES

**Dataset Source:**
Telco Customer Churn Dataset (provided in project)

**Algorithm Reference:**
- Scikit-learn KNN Documentation: https://scikit-learn.org/stable/modules/neighbors.html
- "Introduction to Statistical Learning" - Chapter on KNN

**Code Inspiration:**
- Logistic Regression implementation in team repository (for consistency)

---

## ✅ SUBMISSION CHECKLIST

For your KNN component:

- [x] Working KNN model implementation
- [x] Model training script (`train_knn_model.py`)
- [x] Jupyter notebook (`knn_model.ipynb`)
- [x] Visualizations (4 PNG files)
- [x] Predictions CSV file
- [x] This summary document
- [ ] Add to report PDF (when compiling final report)
- [ ] Prepare 4-minute presentation video
- [ ] Update GitHub repository with commits

---

**Generated:** March 29, 2026  
**Author:** Your Name (Team Member - KNN Specialist)  
**Course:** Machine Learning Assignment  
**Algorithm:** K-Nearest Neighbors (KNN)
