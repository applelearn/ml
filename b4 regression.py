from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

california = fetch_california_housing(as_frame=True)
housing = california.frame

imputer = SimpleImputer(strategy="median")
housing_num = housing.copy()
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

print("Printing housing dataset information after transformation\n")
print(housing_tr.info())

housing_tr['income_cat'] = pd.cut(
    housing_tr['MedInc'],
    bins=[0., 1.5, 3., 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

print(housing_tr.info())

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X=housing_tr, y=housing_tr['income_cat']):
    strat_train_set = housing_tr.loc[train_index]
    strat_test_set = housing_tr.loc[test_index]

strat_train_labels = strat_train_set["MedHouseVal"].copy()
strat_train_set = strat_train_set.drop("MedHouseVal", axis=1)
strat_test_labels = strat_test_set["MedHouseVal"].copy()
strat_test_set = strat_test_set.drop("MedHouseVal", axis=1)

lin_reg = LinearRegression()
lin_reg.fit(X=strat_train_set.drop("income_cat", axis=1), y=strat_train_labels)

housing_predictions = lin_reg.predict(strat_test_set.drop("income_cat", axis=1))

lin_mse = mean_squared_error(strat_test_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(f"PMSE for linear regression {lin_rmse}")
