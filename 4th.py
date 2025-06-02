import pandas as pd
import os
import matplotlib.pyplot as plt
import urllib
import urllib.request

file_path = os.path.join("datasets", "housing", "housing.csv")
housing = pd.read_csv(file_path)
print("CSV Data:")
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())
housing.hist()
housing.hist(bins=50, figsize=(20, 15))
plt.suptitle("CSV Data Histograms")
plt.show()

file_path = os.path.join("datasets", "housing", "housing.xlsx")
housing = pd.read_excel(r'C:\Users\keert\OneDrive\Desktop\ml lab\datasets\housing\housing.xlsx')
print("Excel Data:")
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())
housing.hist()
housing.hist(bins=50, figsize=(20, 15))
plt.suptitle("Excel Data Histograms")
plt.show()
