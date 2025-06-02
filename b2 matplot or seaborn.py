import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
housing = pd.DataFrame(california.data, columns=california.feature_names)
housing["median_house_value"] = california.target

housing["MedInc"].hist()
plt.title("Histogram of Median Income")
plt.xlabel("Median Income")
plt.ylabel("Frequency")
plt.show()

print("Content of California housing dataset:")
print(housing)

housing.plot(kind="scatter", x="Longitude", y="Latitude")
plt.title("Scatter Plot: Longitude vs Latitude")
plt.show()

housing.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1)
plt.title("Scatter Plot: Longitude vs Latitude (alpha=0.1)")
plt.show()

housing.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    alpha=0.4,
    s=housing["Population"] / 100,
    label="Population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.legend()
plt.title("California Housing Map Colored by Value")
plt.show()

print("\nCorrelation value of columns in California housing dataset:")
corr_matrix = housing.corr(method="pearson", numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

pd.plotting.scatter_matrix(housing, figsize=(12, 8))
plt.suptitle("Scatter Matrix (All Features)")
plt.show()

attributes = ["median_house_value", "MedInc", "AveRooms", "HouseAge"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(10, 8))
plt.suptitle("Scatter Matrix (Selected Features)")
plt.show()

housing.plot(kind="scatter", x="MedInc", y="median_house_value", alpha=0.1)
plt.title("Median Income vs House Value")
plt.show()

housing["rooms_per_household"] = housing["AveRooms"] / housing["AveOccup"]
housing["bedrooms_per_room"] = housing["AveBedrms"] / housing["AveRooms"]
housing["population_per_household"] = housing["Population"] / housing["AveOccup"]

print("\nCorrelation value after engineering 3 new columns:")
corr_matrix = housing.corr(method="pearson", numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("******************************************************************\n")

final_attributes = attributes + [
    "rooms_per_household", "bedrooms_per_room", "population_per_household"
]
pd.plotting.scatter_matrix(housing[final_attributes], figsize=(12, 10))
plt.suptitle("Final Scatter Matrix with Engineered Features")
plt.show()
