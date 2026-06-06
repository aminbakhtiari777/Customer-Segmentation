# Customer Segmentation using K-Means Clustering

## Project Overview

This project applies K-Means Clustering to segment mall customers based on their purchasing behavior.

The goal is to discover hidden customer groups without using any target labels and provide business insights for marketing and customer relationship management.

## Dataset

Mall Customers Dataset

Features used for clustering:

* Annual Income (k$)
* Spending Score (1-100)

Additional columns:

* CustomerID
* Gender
* Age

## Project Steps

1. Data Loading and Exploration
2. Missing Value Check
3. Duplicate Check
4. Feature Selection
5. Data Scaling using StandardScaler
6. Elbow Method for Optimal K Selection
7. K-Means Model Training
8. Cluster Analysis
9. Business Interpretation
10. Model Saving with Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Joblib

## K-Means Clustering

The Elbow Method was used to determine the optimal number of clusters.

Selected number of clusters:

```text
K = 5
```

## Cluster Summary

### Cluster 0

Regular customers with average income and average spending behavior.

### Cluster 1

High-income customers with high spending behavior.
These customers can be considered VIP customers.

### Cluster 2

Low-income customers with high spending behavior.
Loyal and highly engaged customers.

### Cluster 3

High-income customers with low spending behavior.
Potential customers for targeted marketing campaigns.

### Cluster 4

Low-income customers with low spending behavior.

## Business Insights

The most valuable opportunities were identified in:

* High Income / Low Spending customers
* Low Income / High Spending customers

These groups can be targeted using personalized marketing strategies and loyalty programs.

## Saved Artifacts

* kmeans_customer_segmentation_model.joblib
* customer_segmentation_scaler.joblib

## Learning Outcomes

This project covers:

* Unsupervised Learning
* K-Means Clustering
* Feature Scaling
* Elbow Method
* Inertia
* Cluster Centers
* Customer Segmentation
* Business-Oriented Data Analysis
