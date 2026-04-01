# DECISION TREE — Customer Churn Prediction


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import os
import joblib

# Set output directory for plots (works in both .py and .ipynb)
try:
    output_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    output_dir = os.getcwd()

# ── STEP 1: Load & Inspect Data ─────────────────────────────
# Path relative to script location
dataset_path = os.path.join(os.path.dirname(output_dir), 'dataset', 'telco_churn.csv')
print(f'Loading dataset from: {dataset_path}')

df = pd.read_csv(dataset_path)
print(f'Dataset shape: {df.shape}')
print(df.head())

# ── 1a. Identify Categorical Columns before dropping ────────
# The original script mentions encoding all remaining categorical columns
# To reproduce the EDA correlation map exactly with all features, we need numeric values.

# ── STEP 2: Preprocess ──────────────────────────────────────
# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('Nulls before drop:', df.isnull().sum().sum())
df.dropna(inplace=True)
print('Shape after drop:', df.shape)

# Drop Irrelevant Column
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode Target Variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode all remaining categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ── Exploratory Data Analysis (EDA) ─────────────────────────
print('Churn distribution:')
print(df['Churn'].value_counts(normalize=True) * 100)

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
print("Saved correlation_heatmap.png")
# plt.show() # Not showing interactively during script execution

# ── STEP 3: Feature / Target Split ──────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# ── STEP 4: Train Decision Tree (Initial) ───────────────────
dt_model = DecisionTreeClassifier(
    criterion='gini',          # Split quality measure
    max_depth=5,               # Limit depth to prevent overfitting
    min_samples_split=10,      # Min samples to split a node
    min_samples_leaf=5,        # Min samples in leaf
    class_weight='balanced',   # Handle class imbalance
    random_state=42            # Reproducibility
)
dt_model.fit(X_train, y_train)

# ── STEP 5: Evaluate ────────────────────────────────────────
y_pred = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f'Accuracy:  {acc:.4f}')
print(f'ROC-AUC:   {auc:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Churn','Churn']))

# ── STEP 6: Confusion Matrix Plot ───────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Decision Tree — Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dt_confusion_matrix.png'), dpi=150)
print("Saved dt_confusion_matrix.png")

# ── STEP 7: Visualise Decision Tree ─────────────────────────
plt.figure(figsize=(20, 10))
plot_tree(dt_model,
          feature_names=X.columns,
          class_names=['No Churn', 'Churn'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree — Customer Churn Prediction', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decision_tree_visual.png'), dpi=150, bbox_inches='tight')
print("Saved decision_tree_visual.png")

# ── STEP 8: Feature Importance ──────────────────────────────
importances = pd.Series(dt_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances_sorted.head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Feature Importances — Decision Tree')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dt_feature_importance.png'), dpi=150)
print("Saved dt_feature_importance.png")

# ── STEP 9: ROC Curve ───────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0,1],[0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Decision Tree')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dt_roc_curve.png'), dpi=150)
print("Saved dt_roc_curve.png")

# ── OVERFITTING ANALYSIS (DEPTH VS ACCURACY) ────────────────
print("\nPerforming Overfitting Analysis (Depth vs Accuracy)...")
depths = range(1, 21)
train_scores, test_scores = [], []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, class_weight='balanced',
                                 random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  clf.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, label='Train Accuracy', marker='o')
plt.plot(depths, test_scores,  label='Test Accuracy',  marker='s')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Depth vs Accuracy (Overfitting Analysis)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dt_depth_analysis.png'), dpi=150)
print("Saved dt_depth_analysis.png")

train_acc = accuracy_score(y_train, dt_model.predict(X_train))
test_acc  = accuracy_score(y_test,  dt_model.predict(X_test))
print(f'\nAt max_depth=5:')
print(f'Training Accuracy: {train_acc:.4f}')
print(f'Test Accuracy:     {test_acc:.4f}')
print(f'Overfitting Gap:   {train_acc - test_acc:.4f}')

# ── STEP 10: Hyperparameter Tuning with GridSearchCV ────────
print("\nRunning GridSearchCV for Hyperparameter Tuning...")
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
}
grid_search = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,           # 5-fold cross-validation
    scoring='roc_auc',
    n_jobs=-1,      # Use all CPU cores
    verbose=1
)
grid_search.fit(X_train, y_train)

print('Best Parameters:', grid_search.best_params_)
print('Best CV AUC:    ', round(grid_search.best_score_, 4))

# ── STEP 11: Evaluate Tuned Model ───────────────────────────
best_dt = grid_search.best_estimator_
y_pred_tuned = best_dt.predict(X_test)
y_prob_tuned = best_dt.predict_proba(X_test)[:, 1]

print('\nTuned Model Accuracy:', round(accuracy_score(y_test, y_pred_tuned), 4))
print('Tuned Model AUC:     ', round(roc_auc_score(y_test, y_prob_tuned), 4))
print('\nTuned Model Classification Report:')
print(classification_report(y_test, y_pred_tuned, target_names=['No Churn','Churn']))

# ── STEP 12: Cross-Validation Score ─────────────────────────
cv_scores = cross_val_score(best_dt, X, y, cv=10, scoring='accuracy')
print(f'\n10-fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

# ── STEP 13: Export Model for Frontend Integration ──────────
model_path = os.path.join(output_dir, 'decision_tree_model.pkl')
joblib.dump(best_dt, model_path)
print(f'\nModel successfully saved to {model_path} for Frontend Integration.')

print("\nScript completed successfully.")
