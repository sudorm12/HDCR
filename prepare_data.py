import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

applications = pd.read_csv('application_train.csv', index_col="SK_ID_CURR")
apps_clean = applications.copy()

# TODO: identify encoded columns, change to categorical

# change y/n columns to boolean
yn_map = {'Y': 1,
          'N': 0}

yn_cols = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

apps_clean[yn_cols] = apps_clean[yn_cols].replace(yn_map)

# current home information
# TODO: pick one statistic for each home info category

# external data source
# TODO: impute with quantile of other scores

# age of car if owned
# TODO: impute car age with average, if client owns a car

# occupation type
# TODO: replace with variable indicating occupation undefined

# credit bureau requests
# TODO: impute all with zero, except past year with one

# annuity amount
# TODO: infer from amt_credit and amt_goods_price
