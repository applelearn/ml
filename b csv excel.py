import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

os.makedirs("datasets/housing", exist_ok=True)
california = fetch_california_housing(as_frame=True)
housing = california.frame

housing.to_csv("datasets/housing/california.csv", index=False)
housing.to_excel("datasets/housing/california.xlsx", index=False)

for file, title in [("california.csv", "CSV"), ("california.xlsx", "Excel")]:
    print(f"\n=== Reading from {title} ===")
    df = pd.read_csv if file.endswith(".csv") else pd.read_excel
    data = df(f"datasets/housing/{file}")
    print(data.head(), "\n")
    print(data.info(), "\n")
    print(data.describe(), "\n")
    data.hist()
    plt.suptitle(f"{title} Dataset - Default Histogram")
    plt.show()
    data.hist(bins=50, figsize=(20, 15))
    plt.suptitle(f"{title} Dataset - Custom Histogram")
    plt.show()
