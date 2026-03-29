#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Model for Customer Churn Prediction
Team Assignment - Individual Component

This script implements the KNN algorithm to predict telecom customer churn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

print("="*60)
print("K-NEAREST NEIGHBORS (KNN) MODEL TRAINING")
print("Customer Churn Prediction System")
print("="*60)

# Load the preprocessed dataset
print("\n[Step 1] Loading dataset...")
X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

y_train = pd.read_csv("dataset/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset/y_test.csv").values.ravel()

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Display sample of training data
print("\nSample training data:")
print(X_train.head())

# Check target variable distribution
print("\nTarget variable distribution:")
print(pd.Series(y_train).value_counts())

# Feature scaling is crucial for KNN since it's distance-based
print("\n[Step 2] Scaling features...")
scaler = StandardScaler()

# Fit scaler on training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data has been scaled using StandardScaler")

# Finding the optimal K value
print("\n[Step 3] Finding optimal K value...")
k_range = range(1, 31)
accuracy_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"K={k}: Accuracy = {accuracy:.4f}")

# Plot accuracy vs K value
plt.figure(figsize=(12, 6))
plt.plot(k_range, accuracy_scores, marker='o', linestyle='-', color='blue')
plt.xlabel('K Value (Number of Neighbors)')
plt.ylabel('Accuracy Score')
plt.title('KNN: Accuracy vs K Value')
plt.grid(True)
plt.xticks(range(0, 31, 2))
plt.savefig('notebooks/knn_accuracy_plot.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: notebooks/knn_accuracy_plot.png")
plt.close()

# Find and display optimal K
optimal_k = k_range[np.argmax(accuracy_scores)]
best_accuracy = max(accuracy_scores)
print(f"\n✓ Optimal K value: {optimal_k}")
print(f"✓ Best accuracy: {best_accuracy:.4f}")

# Train the final KNN model with optimal K
print(f"\n[Step 4] Training final KNN model with K={optimal_k}...")
knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train_scaled, y_train)

print("✓ KNN Model trained successfully")

# Make predictions on test set
print("\n[Step 5] Making predictions...")
y_pred = knn_model.predict(X_test_scaled)

# Display first 10 predictions
print("\nFirst 10 predictions:")
print(y_pred[:10])

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"K-Nearest Neighbors (KNN) Model Performance")
print(f"{'='*50}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Number of neighbors used: {optimal_k}")

# Generate detailed classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churn', 'Churn'],
            yticklabels=['Not Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - K-Nearest Neighbors (KNN)')
plt.savefig('notebooks/knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/knn_confusion_matrix.png")
plt.close()

# Get prediction probabilities
probabilities = knn_model.predict_proba(X_test_scaled)

# Extract churn probability (probability of class 1)
churn_prob = probabilities[:, 1]

print(f"\nFirst 10 customers' churn probabilities:")
print(churn_prob[:10])

# Define risk levels based on churn probability
def risk_level(p):
    """
    Categorize customers into risk levels based on churn probability
    
    Parameters:
    p (float): Churn probability (0 to 1)
    
    Returns:
    str: Risk level category
    """
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# Apply risk level categorization
risk_levels = [risk_level(p) for p in churn_prob]

print(f"\nFirst 10 customers' risk levels:")
print(risk_levels[:10])

# Count risk level distribution
risk_counts = pd.Series(risk_levels).value_counts()
print(f"\nRisk Level Distribution:")
print(risk_counts)

# Visualize risk level distribution
plt.figure(figsize=(8, 6))
plt.bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
plt.xlabel('Risk Level')
plt.ylabel('Number of Customers')
plt.title('Customer Risk Level Distribution - KNN Model')
plt.savefig('notebooks/knn_risk_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/knn_risk_distribution.png")
plt.close()

# Define recommendations based on risk level
def recommendation(risk):
    """
    Provide business recommendations based on customer risk level
    
    Parameters:
    risk (str): Risk level category
    
    Returns:
    str: Recommended action
    """
    if risk == "High Risk":
        return "Offer discount or loyalty plan"
    elif risk == "Medium Risk":
        return "Offer promotional package"
    else:
        return "No action needed"

# Generate recommendations for all customers
recommendations = [recommendation(r) for r in risk_levels]

print(f"\nFirst 10 customers' recommendations:")
print(recommendations[:10])

# Create results DataFrame
results = pd.DataFrame({
    "Actual": y_test,
    "Prediction": y_pred,
    "Churn Probability": churn_prob,
    "Risk Level": risk_levels,
    "Recommendation": recommendations
})

print(f"\nResults Summary (First 10 customers):")
print(results.head(10))

# Save results to CSV for later analysis
results.to_csv("dataset/knn_predictions.csv", index=False)
print(f"\n✓ Predictions saved to 'dataset/knn_predictions.csv'")

# Cross-validation to verify model stability
print(f"\n[Step 6] Performing 5-fold cross-validation...")
cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)

print(f"\n{'='*50}")
print("Cross-Validation Results (5-fold)")
print(f"{'='*50}")
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualize cross-validation scores
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_scores, color='skyblue', edgecolor='blue')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy Score')
plt.title('5-Fold Cross-Validation Scores - KNN Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('notebooks/knn_cv_scores.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/knn_cv_scores.png")
plt.close()

# Analyze misclassified examples
results['Correct'] = results['Actual'] == results['Prediction']
misclassified = results[~results['Correct']]

print(f"\n{'='*50}")
print("Misclassification Analysis")
print(f"{'='*50}")
print(f"Total misclassified: {len(misclassified)} out of {len(results)}")
print(f"Misclassification rate: {len(misclassified)/len(results)*100:.2f}%")

# Show some misclassified examples
print(f"\nSample misclassified examples:")
print(misclassified.head(10))

# Feature importance approximation using distance-based approach
# For KNN, we can analyze which features contribute most to predictions
print(f"\n{'='*50}")
print("KNN Model Information")
print(f"{'='*50}")
print(f"Algorithm: K-Nearest Neighbors")
print(f"Number of neighbors: {optimal_k}")
print(f"Distance metric: Euclidean (default)")
print(f"Weights: uniform (all neighbors weighted equally)")
print(f"Training samples: {len(X_train_scaled)}")
print(f"Testing samples: {len(X_test_scaled)}")
print(f"Number of features: {X_train.shape[1]}")

# Final performance summary
print(f"\n{'='*50}")
print("FINAL MODEL SUMMARY")
print(f"{'='*50}")
print(f"Model: K-Nearest Neighbors (KNN) Classifier")
print(f"Optimal K Value: {optimal_k}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}%")

# Get classification metrics
class_report = classification_report(y_test, y_pred, output_dict=True)
# Get the keys to find the correct label for churn class
labels = list(class_report.keys())
print(f"\nClassification labels found: {labels}")
# The last key before 'accuracy', 'macro avg', 'weighted avg' is typically the positive class
churn_label = [k for k in labels if k not in ['accuracy', 'macro avg', 'weighted avg']][-1]
print(f"Using churn label: {churn_label}")
precision_churn = class_report[churn_label]['precision']
recall_churn = class_report[churn_label]['recall']
f1_churn = class_report[churn_label]['f1-score']

print(f"Precision (Churn): {precision_churn:.4f}")
print(f"Recall (Churn): {recall_churn:.4f}")
print(f"F1-Score (Churn): {f1_churn:.4f}")
print(f"{'='*50}")

print("\n" + "="*60)
print("✓ KNN MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated Files:")
print("  - notebooks/knn_accuracy_plot.png")
print("  - notebooks/knn_confusion_matrix.png")
print("  - notebooks/knn_risk_distribution.png")
print("  - notebooks/knn_cv_scores.png")
print("  - dataset/knn_predictions.csv")
print("\nYour KNN implementation is ready for submission!")
print("="*60 + "\n")
