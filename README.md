# Clustering Network Traffic for Cybersecurity Analysis Using CICIDS2017 Dataset

## [LINK TO GITHUB REPOSITORY](https://github.com/Karan-D-Software/Machine-Learning-Network-Security)

This project aims to perform clustering analysis on the CICIDS2017 dataset to identify patterns and group similar types of network traffic. The objective is to detect and categorize different types of network activities, which can help in identifying normal and potentially malicious behaviors. Through this analysis, we hope to gain insights into the characteristics of various network traffic clusters and enhance our understanding of cybersecurity threats.

## Dataset
We use the CICIDS2017 dataset, which includes various types of network traffic, such as normal traffic, DoS, DDoS, and other types of attacks. The dataset is publicly available on Kaggle. [Link to the CICIDS2017 Dataset on Kaggle](https://www.kaggle.com/datasets/sweety18/cicids2017-full-dataset)

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Analysis](#problem-analysis)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [About the Data and Initial Data Cleaning](#about-the-data-and-initial-data-cleaning)
    - [Data Distribution Analysis](#data-distribution-analysis)
    - [Correlations Analysis](#correlations-analysis)
    - [Missing Values Analysis](#missing-values-analysis)
    - [Outlier Analysis](#outlier-analysis)
    - [Final Data Cleaning](#final-data-cleaning)
4. [Model Training and Evaluation](#model-training-and-evaluation)
    - [Clustering Models](#clustering-models-eg-k-means-dbscan-hierarchical-clustering)
5. [Results and Discussion](#results-and-discussion)
6. [References](#references)

## Introduction
In this project, we aim to perform clustering analysis on the CICIDS2017 dataset to identify patterns and group similar types of network traffic. The CICIDS2017 dataset, provided by the Canadian Institute for Cybersecurity (CIC), includes a diverse range of network traffic data, encompassing normal and various attack activities. By leveraging unsupervised learning techniques, we seek to enhance the understanding of network behaviours and identify potential security threats, contributing valuable insights to cybersecurity.

## Problem Analysis
### What is the Problem and Its Impact on Industry?

The problem we are addressing is detecting and categorizing various types of network traffic, including potentially malicious activities, within large datasets. Cybersecurity is a critical concern for industries worldwide, as cyber-attacks can lead to significant financial losses, data breaches, and reputational damage. Traditional methods of detecting malicious activities often need help with network traffic data's vast and evolving nature. Clustering analysis offers a way to identify unusual patterns and group similar types of traffic, providing an automated method to enhance threat detection and improve network security measures.

### Machine Learning Model and Rationale
For this project, we have carefully selected two powerful clustering models: K-means clustering and DBSCAN (Density-Based Spatial Clustering of Applications with Noise). K-means, known for its simplicity and efficiency, excels in partitioning data into distinct clusters based on similarity, a key feature for identifying patterns in large datasets. DBSCAN, on the other hand, is a champion in handling noisy data and finding arbitrarily shaped clusters, making it a potent tool for identifying outliers and unusual network behaviours. These models, when used in tandem, provide a robust framework for analyzing the CICIDS2017 dataset, instilling confidence in the effectiveness of our approach.

### Expected Outcome
The expected outcome of this project is to develop a comprehensive clustering analysis of the CICIDS2017 dataset that successfully identifies and categorizes different types of network traffic. The results should highlight distinct clusters representing normal and malicious activities, providing valuable insights into network behaviours. This analysis aims to enhance the understanding of network security threats and support the development of more effective cybersecurity measures. By uncovering hidden patterns in the data, we aim to contribute to the broader effort of improving network security in various industries.

## Exploratory Data Analysis (EDA)