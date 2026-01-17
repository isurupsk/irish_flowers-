# Iris Flower Classification Project
# මල් වර්ග වර්ගීකරණ ව්‍යාපෘතිය

# Step 1: Import කරන්න ඕන libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Dataset එක load කරන්න
print("=" * 50)
print("Dataset Loading (දත්ත ගැනීම)")
print("=" * 50)

iris = load_iris()
X = iris.data  # Features (විශේෂාංග) - sepal length, sepal width, petal length, petal width

# print(f"\nFeatures: {X}")
y = iris.target  # Target (ඉලක්කය) - flower type (0, 1, 2)
print(f"\nFeatures: {y}")
# DataFrame එකක් හදන්න - easy visualization වලට
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples (මුළු නියැදි): {len(df)}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Step 3: Data Exploration (දත්ත විශ්ලේෂණය)
print("\n" + "=" * 50)
print("Data Exploration")
print("=" * 50)

print("\nBasic Statistics (මූලික සංඛ්‍යාන):")
print(df.describe())

print("\nSpecies Distribution (මල් වර්ග ව්‍යාප්තිය):")
print(df['species_name'].value_counts())

print("\nMissing Values (නැති අගයන්):")
print(df.isnull().sum())

# Step 4: Data Visualization (දත්ත නිරූපණය)
print("\n" + "=" * 50)
print("Creating Visualizations (චිත්‍ර නිර්මාණය)")
print("=" * 50)

# Pairplot - Features එකිනෙකට සංසන්දනය කිරීම
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species_name', markers=['o', 's', 'D'])
plt.suptitle('Iris Dataset - Feature Relationships', y=1.02)
plt.tight_layout()
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
print("✓ Pairplot saved as 'iris_pairplot.png'")

# Correlation Matrix (සහසම්බන්ධතා න්‍යාසය)
plt.figure(figsize=(10, 8))
correlation = df.iloc[:, :-2].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Correlation matrix saved as 'iris_correlation.png'")

# Step 5: Data Preparation (දත්ත සකස් කිරීම)
print("\n" + "=" * 50)
print("Data Preparation for Training")
print("=" * 50)

# Train-Test Split (පුහුණු-පරීක්ෂණ බෙදීම)
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples (පුහුණු නියැදි): {len(X_train)}")
print(f"Testing samples (පරීක්ෂණ නියැදි): {len(X_test)}")

# Feature Scaling (විශේෂාංග පරිමාණකරණය)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Data scaled successfully")

# Step 6: Model Training (ආකෘති පුහුණු කිරීම)
print("\n" + "=" * 50)
print("Training Multiple Models (බහු ආකෘති පුහුණු කිරීම)")
print("=" * 50)

models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Model train කරන්න
    model.fit(X_train_scaled, y_train)
    
    # Predictions (අනාවැකි)
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy (නිරවද්‍යතාවය)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"✓ {name} Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed Report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Results Comparison (ප්‍රතිඵල සංසන්දනය)
print("\n" + "=" * 50)
print("Model Comparison (ආකෘති සංසන්දනය)")
print("=" * 50)

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")

# Best Model
best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with {results[best_model] * 100:.2f}% accuracy")

# Visualize Model Comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [acc * 100 for acc in results.values()], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.xlabel('Models (ආකෘති)')
plt.ylabel('Accuracy % (නිරවද්‍යතාවය)')
plt.title('Model Performance Comparison')
plt.ylim(90, 100)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc * 100 + 0.5, f'{acc * 100:.2f}%', ha='center')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison chart saved as 'model_comparison.png'")

# Step 8: Confusion Matrix (ව්‍යාකූලතා න්‍යාසය)
print("\n" + "=" * 50)
print("Confusion Matrix for Best Model")
print("=" * 50)

best_model_obj = models[best_model]
y_pred_best = best_model_obj.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted (අනාවැකි කළ)')
plt.ylabel('Actual (සැබෑ)')
plt.title(f'Confusion Matrix - {best_model}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# Step 9: Make Predictions (අලුත් අනාවැකි කිරීම)
print("\n" + "=" * 50)
print("Making New Predictions (අලුත් අනාවැකි)")
print("=" * 50)

# Example: නව මලක් predict කරන්න
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sample measurements
new_flower_scaled = scaler.transform(new_flower)
prediction = best_model_obj.predict(new_flower_scaled)
predicted_species = iris.target_names[prediction[0]]

print(f"New Flower Measurements: {new_flower[0]}")
print(f"Predicted Species (අනාවැකි කළ මල් වර්ගය): {predicted_species}")

print("\n" + "=" * 50)
print("Project Completed Successfully! ✨")
print("=" * 50)