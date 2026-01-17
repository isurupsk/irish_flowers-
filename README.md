A comprehensive machine learning project that classifies iris flowers into three species using multiple classification algorithms.
📋 Table of Contents

Overview
Dataset
Features
Installation
Usage
Models Used
Results
Visualizations
Project Structure
Requirements
Contributing
License

🎯 Overview
This project demonstrates the application of supervised machine learning algorithms to classify iris flowers based on their physical measurements. The project compares three different classification models and identifies the best performer for this dataset.
Key Objectives:

Load and explore the famous Iris dataset
Perform data visualization and analysis
Train multiple classification models
Compare model performances
Make predictions on new data

🌺 Dataset
The project uses the classic Iris Dataset, which includes:

150 samples of iris flowers
4 features: sepal length, sepal width, petal length, petal width (all in cm)
3 target classes:

Setosa
Versicolor
Virginica



Each class contains 50 samples, making it a balanced dataset ideal for classification tasks.
✨ Features

Data Exploration: Statistical analysis and distribution visualization
Multiple Models: Comparison of KNN, Decision Tree, and SVM algorithms
Feature Scaling: StandardScaler for optimal model performance
Comprehensive Visualization: Pairplots, correlation matrices, and confusion matrices
Bilingual Comments: Code documentation in both English and Sinhala (සිංහල)
Model Comparison: Performance metrics and accuracy visualization
Prediction Capability: Make predictions on new flower measurements

🔧 Installation
Prerequisites

Python 3.7 or higher
pip package manager

Setup

Clone the repository:

bashgit clone <repository-url>
cd iris-classification

Install required packages:

bashpip install numpy pandas matplotlib seaborn scikit-learn
Or use the requirements file:
bashpip install -r requirements.txt
🚀 Usage
Run the main script:
bashpython iris_classification.py
The script will:

Load and explore the dataset
Generate visualization plots (saved as PNG files)
Train three different models
Display accuracy metrics and classification reports
Save comparison charts and confusion matrix
Make a sample prediction

🤖 Models Used
1. K-Nearest Neighbors (KNN)

Parameters: n_neighbors=3
Description: Classifies based on the 3 nearest training examples

2. Decision Tree

Parameters: random_state=42
Description: Creates a tree-like model of decisions

3. Support Vector Machine (SVM)

Parameters: kernel='rbf', random_state=42
Description: Finds optimal hyperplane for classification

📊 Results
The project outputs detailed performance metrics including:

Accuracy scores for each model
Classification reports with precision, recall, and F1-score
Confusion matrices showing prediction accuracy per class
Visual comparisons of model performance

Expected accuracy range: 90-100% depending on the train-test split.
