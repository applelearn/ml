import os
import pandas as pd
import matplotlib.pyplot as plt

file_path = os.path.join("datasets", "housing", "housing.csv")
housing = pd.read_csv(file_path)

housing["median_income"].hist()
print("content of housing.csv")
print(housing)

housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,
    label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.show()

print("correlation value of 9 columns in housing dataset")
print("*\n")
corr_matrix = housing.corr(method="pearson", numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

pd.plotting.scatter_matrix(housing)
plt.show()

attribute = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.plotting.scatter_matrix(housing[attribute])
plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(method="pearson", numeric_only=True)
print("correlation value of 12 columns in housing dataset after engineering 3 new columns")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("******************************************************************\n")

pd.plotting.scatter_matrix(housing)
plt.show()
