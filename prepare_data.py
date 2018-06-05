import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

applications = pd.read_csv('application_train.csv', index_col="SK_ID_CURR")

# TODO: identify encoded columns, change to categorical
# TODO: change y/n columns to boolean

# current home information
# TODO: pick one statistic for each home info category

# external data source
# TODO: impute with quantile of other scores

# age of car if owned
# TODO: impute car age with average, if client owns a car