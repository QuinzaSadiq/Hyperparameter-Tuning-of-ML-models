# views.py
from django.shortcuts import render
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings

def classify_emails(request):
    # Load and prepare the data
    df = pd.read_csv("emails.csv")  # Update with the actual path
    df.fillna(0, inplace=True)

    X = df.iloc[:, 1:3001]
    Y = df.iloc[:, -1].values

    # Split the dataset
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)

    # Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }
    rfc = RandomForestClassifier()
    grid_rfc = GridSearchCV(rfc, param_grid, cv=5)
    grid_rfc.fit(train_x, train_y)

    best_params_grid = grid_rfc.best_params_

    # Create and save the heatmap
    results_grid = pd.DataFrame(grid_rfc.cv_results_)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        results_grid.pivot_table(
            index='param_n_estimators', 
            columns='param_max_depth', 
            values='mean_test_score'
        ), 
        annot=True, 
        cmap='YlGnBu'
    )
    plt.title('Grid Search Results')
    plt.xlabel('Max Depth')
    plt.ylabel('Number of Estimators')

    # Define the path for saving the heatmap
    image_dir = os.path.join(settings.BASE_DIR, 'static', 'images')
    image_path = os.path.join(image_dir, 'grid_search_heatmap.png')

    # Check if the directory exists, if not, create it
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    plt.savefig(image_path)
    plt.close()  # Close the plot to free up memory

    # Random Search
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 6, None],
        "max_features": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "bootstrap": [True, False]
    }
    random_rfc = RandomizedSearchCV(rfc, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
    random_rfc.fit(train_x, train_y)

    best_params_random = random_rfc.best_params_

    # Predictions with Grid Search
    best_rfc_grid = grid_rfc.best_estimator_
    y_pred_grid = best_rfc_grid.predict(test_x)

    # Predictions with Random Search
    best_rfc_random = random_rfc.best_estimator_
    y_pred_random = best_rfc_random.predict(test_x)

    # Prepare results for HTML
    results = {
        'best_params_grid': best_params_grid,
        'accuracy_grid': accuracy_score(y_pred_grid, test_y),
        'classification_report_grid': classification_report(test_y, y_pred_grid, output_dict=True),
        'best_params_random': best_params_random,
        'accuracy_random': accuracy_score(y_pred_random, test_y),
        'classification_report_random': classification_report(test_y, y_pred_random, output_dict=True),
        'image_url': '/static/images/grid_search_heatmap.png',  # Path for the saved image
    }

    return render(request, 'email_ml_app/results.html', results)
