from sklearn.linear_model import LinearRegression
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = os.path.join("datasets", "housing", "housing.csv")
housing = pd.read_csv(file_path)

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

print("\n Printing housing dataset information after transformation\n")
print(housing_tr.info())

housing_tr['income_cat'] = pd.cut(
    x=housing_tr['median_income'],
    bins=[0., 1.5, 3., 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

housing_tr = housing_tr.drop("median_income", axis=1)
print(housing_tr.info())

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X=housing_tr, y=housing_tr['income_cat']):
    strat_train_set = housing_tr.loc[train_index]
    strat_test_set = housing_tr.loc[test_index]

strat_train_set_labels = strat_train_set["median_house_value"].copy()
strat_train_set = strat_train_set.drop("median_house_value", axis=1)
strat_test_set_labels = strat_test_set["median_house_value"].copy()
strat_test_set = strat_test_set.drop("median_house_value", axis=1)

lin_reg = LinearRegression()
lin_reg.fit(X=strat_train_set, y=strat_train_set_labels)

housing_prediction = lin_reg.predict(strat_test_set)

lin_mse = mean_squared_error(strat_test_set_labels, housing_prediction)
lin_mse = np.sqrt(lin_mse)
print("PMSE for linear regression", lin_mse)
