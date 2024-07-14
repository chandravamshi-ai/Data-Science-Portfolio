# Identifying the Optimal Neighborhood in Toronto for a New Gym

## Introduction
This project aims to identify the best neighborhood in Toronto to open a new gym based on demographic, geographic, and venue information. 
[Complete code](https://github.com/chandravamshi-ai/Data-Science-Portfolio/blob/main/Machine%20Learning/Clustering/Identifying%20the%20Optimal%20Neighborhood%20in%20Toronto%20for%20a%20New%20Gym/toronto-neighborhoods-clustering.ipynb)

## Data
The dataset used for this analysis is available on [Kaggle](https://www.kaggle.com/datasets/youssef19/toronto-neighborhoods-inforamtion).


![Datapoints on map](https://github.com/chandravamshi-ai/Data-Science-Portfolio/blob/main/Machine%20Learning/Clustering/Identifying%20the%20Optimal%20Neighborhood%20in%20Toronto%20for%20a%20New%20Gym/images/data%20points%20on%20map.png))

## Methodology 

### Data Preprocessing
Data normalization was performed using min-max normalization to ensure all features were on a similar scale. This step is crucial for k-means clustering, which relies on distance measurements. The min-max normalization formula is:
```
(feature - min(feature)) / (max(feature) - min(feature))
```
Geographical and neighborhood data were excluded from the normalization process as they are used by the clustering algorithm.

### K-Means Clustering
The optimal number of clusters (k) was determined using the elbow method, which involves plotting the average distance to cluster centers for different values of k and identifying the point where the rate of decrease sharply changes. The ideal number of clusters was found to be 3.
![Optimal K](https://github.com/chandravamshi-ai/Data-Science-Portfolio/blob/main/Machine%20Learning/Clustering/Identifying%20the%20Optimal%20Neighborhood%20in%20Toronto%20for%20a%20New%20Gym/images/Optimal%20no.of%20Clusters.png)

## Results 
The neighborhoods were categorized into three clusters as visualized in the figure below. The red cluster represents the first cluster, violet the second cluster, and green the third cluster.

![Result](https://github.com/chandravamshi-ai/Data-Science-Portfolio/blob/main/Machine%20Learning/Clustering/Identifying%20the%20Optimal%20Neighborhood%20in%20Toronto%20for%20a%20New%20Gym/images/K%20means%20clustered.png)

## Conclusion 
By analyzing demographic and venue data from Foursquare API, neighborhoods in Toronto were grouped into three clusters using k-means clustering. A significant finding was that the number of gyms is correlated with the number of venues.

- **Third Cluster**: Neighborhoods with a high number of venues and gyms.
    - **Optimal Choice**: Trinity-Bellwoods neighborhood.
- **First Cluster**: Neighborhoods with a large population but fewer gyms and a moderate number of venues.
    - **Optimal Choice**: Church-Yonge Corridor neighborhood, which has a comparable number of venues to Trinity-Bellwoods but double the population, making it a prime location for a new gym.
