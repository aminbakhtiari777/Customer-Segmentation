# Model Card — Customer Segmentation Platform

## Summary

This project groups mall customers using standardized annual income and spending score. A five-cluster K-Means model is trained with a fixed random seed and multiple initializations. Raw cluster IDs are translated into stable business-facing names based on cluster geometry.

## Dataset

The included educational dataset contains 200 records with customer ID, gender, age, annual income, and spending score. Only annual income and spending score are used for clustering. Customer ID is never used as a feature, and gender and age are retained only for aggregate reporting.

## Evaluation

Because clustering has no ground-truth target, the project reports intrinsic metrics:

- inertia for within-cluster compactness;
- silhouette score for cohesion versus separation;
- Davies-Bouldin score for cluster overlap.

These metrics describe geometry, not business impact. A real deployment also requires campaign experiments, stability analysis over time, and stakeholder validation.

## Intended use

- educational unsupervised-learning experiments;
- exploratory cohort analysis;
- demonstration of reproducible clustering and inference artifacts;
- starting point for marketing experiments with human review.

## Limitations

The dataset is small, old, and not representative of a complete customer population. K-Means assumes roughly spherical clusters and is sensitive to feature choice. Segment names are interpretations rather than facts about individuals. Do not use the model for eligibility, credit, employment, pricing, or other high-impact decisions.

## Monitoring requirements

Track input ranges, missing values, cluster size drift, centroid movement, assignment stability, and measured campaign outcomes. Retrain only after reviewing whether the original five-segment business interpretation remains useful.

