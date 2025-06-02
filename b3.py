import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import fetch_california_housing
import numpy as np

california = fetch_california_housing()
housing = pd.DataFrame(california.data, columns=california.feature_names)
housing["median_house_value"] = california.target

np.random.seed(42)
housing["ocean_proximity"] = np.random.choice(["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"], size=len(housing))

print("######################################################################")
print("Printing housing dataset information\n")
print(housing.info())

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

print("\nPrinting the content of statistics_ variable after executing fit function\n")
print(imputer.statistics_)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

print("#########################################################################")
print("Printing housing dataset information after transformation\n")
print(housing_tr.info())
print("\nTransformed housing dataset contents (first 5 rows)\n")
print(housing_tr.head())

housing_cat = housing[["ocean_proximity"]]
print("\nFirst 10 instances of the 'ocean_proximity' column:\n")
print(housing_cat.head(10))

cat_encoder = OneHotEncoder()
housing_cat_1Hot = cat_encoder.fit_transform(housing_cat)

print("\nOne-hot encoded array of 'ocean_proximity':\n")
print(housing_cat_1Hot.toarray()[:10])

print("\nCategories of OneHotEncoder:\n")
print(cat_encoder.categories_)

new_housing = housing.drop(['ocean_proximity'], axis=1)
print("\nHousing dataset (without 'ocean_proximity') for scaling:\n")
print(new_housing.head())

scalar = StandardScaler()
scaled_data = scalar.fit_transform(new_housing)

scaled_df = pd.DataFrame(scaled_data, columns=new_housing.columns)

print("\nTop 5 rows of the scaled dataset:\n")
print(scaled_df.head())
