### README: K-Means Clustering Project on Facebook Live sellers in Thailand, UCI ML Repo data.


#### Project Overview
This project demonstrates the implementation of the K-Means Clustering algorithm, an unsupervised machine learning technique, to classify data into clusters. The notebook covers the essential steps involved in preparing the data, applying the K-Means algorithm, and evaluating the results.

---

#### Detailed Steps

1. **Introduction**
   - The project aims to classify data using K-Means Clustering, an algorithm that partitions data into `k` clusters, where each data point belongs to the cluster with the nearest mean.

2. **Data Preparation**
   - Load the dataset into a Pandas DataFrame.
   - Inspect the data structure, handle missing values, and normalize the features to ensure all data points are on a comparable scale.

3. **Choosing the Number of Clusters (k)**
   - Apply the Elbow Method by plotting the Within-Cluster Sum of Squares (WCSS) for different values of `k`.
   - The plot helps in identifying the optimal `k` where the WCSS begins to decrease at a slower rate, forming an "elbow."

4. **K-Means Clustering with Different k Values**
   - **For `k=2`:**
     - Fit the K-Means model.
     - Evaluate the model using inertia (a measure of how internally coherent the clusters are) and accuracy.
     - Accuracy obtained: 1%.
   - **For `k=4`:**
     - Fit the K-Means model.
     - Evaluate the model using the same metrics.
     - Accuracy obtained: 62%.

5. **Results and Conclusion**
   - The Elbow Method suggested that `k=2` could be a good starting point.
   - However, the accuracy with `k=2` was very low (1%).
   - Increasing `k` to 4 significantly improved the accuracy to 62%.
   - Therefore, the optimal number of clusters for this dataset is concluded to be 4.

---

#### Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, sklearn

---

#### How to Run

1. Ensure you have the required libraries installed.
2. Load the Jupyter Notebook and run each cell sequentially.
3. Observe the output and visualizations for each section.

---

#### Conclusion
This project provides a step-by-step implementation of K-Means Clustering, demonstrating how to choose the optimal number of clusters and evaluate the model's performance. The project concludes that using `k=4` clusters yields the best results for the given dataset.
