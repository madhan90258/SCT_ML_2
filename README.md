# SCT_ML_2
# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project applies **K-Means clustering** to segment mall customers based on their **Age**, **Annual Income**, and **Spending Score**. The goal is to identify distinct customer groups for targeted marketing strategies and better decision-making.

---

## 📊 Dataset

- **Source**: `Mall_Customers.csv`
- **Columns**:
  - `CustomerID`: Unique customer identifier
  - `Gender`: Male or Female (not used in clustering)
  - `Age`: Customer's age
  - `Annual Income (k$)`: Income in thousands
  - `Spending Score (1-100)`: Score assigned by the mall based on customer behavior

---

## 🧠 Clustering Approach

### Features Used:
- Age
- Annual Income (k$)
- Spending Score (1–100)

### Preprocessing:
- Standardization using `StandardScaler`

### Steps:
1. Load and prepare the dataset
2. Use the **Elbow Method** to determine optimal clusters (k = 5)
3. Apply **KMeans Clustering**
4. Visualize the clusters in **3D**
5. Save the results to `Mall_Customers_Clustered.csv`

---

## 📈 Visualizations

- 📌 **Elbow Curve** to determine the optimal `k`
- 🧭 **3D Scatter Plot** showing cluster separation across Age, Income, and Spending

---

## 📁 Files

| File                         | Description                          |
|------------------------------|--------------------------------------|
| `Mall_Customers.csv`         | Input dataset                        |
| `Mall_Customers_Clustered.csv` | Dataset with added `Cluster` column |
| `customer_segmentation.py`   | Final clustering script              |
| `README.md`                  | Project overview (this file)         |

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install pandas matplotlib scikit-learn
