
# Mall Customer Segmentation (K-Means & DBSCAN)

## 📌 Project Overview

This project applies **unsupervised machine learning** to the **Mall Customers dataset** (from Kaggle).
The goal is to segment customers into meaningful groups based on:

* **Annual Income**
* **Spending Score**
* (Optional) **Age**

Clustering helps businesses understand customer behavior and target specific groups with personalized marketing strategies.

---

## ⚙️ Steps in the Project

1. **Data Preprocessing**

   * Load dataset (200 mall customers)
   * Select relevant features (income, spending score, age)
   * Apply scaling

2. **Exploratory Data Analysis**

   * Summary statistics
   * Visual exploration of features

3. **K-Means Clustering**

   * Find optimal number of clusters using **Elbow Method**
   * Fit model with best k
   * Visualize customer segments

4. **DBSCAN Clustering** (comparison)

---

## 📊 Results

* Clear segmentation of customer groups
* Visual plots of spending vs income with cluster centroids

---

## 📂 Files in Repository

* `Mall_Customers.csv` → Dataset
* `mall_customer_segmentation.ipynb` → Jupyter Notebook with full analysis
* `README.md` → Project documentation

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
git clone https://github.com/mariammothmman/customer-segmentation.git
   ```
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open the notebook:

   ```bash
   jupyter notebook mall_customer_segmentation.ipynb
   ```

---

## 📌 Dataset Source

[Kaggle – Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

---
