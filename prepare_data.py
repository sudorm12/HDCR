import pandas as pd

applications = pd.read_csv('application_train.csv', index_col="SK_ID_CURR")
apps_clean = applications.copy()

# change y/n columns to boolean
yn_map = {'Y': 1,
          'N': 0}

yn_cols = ['FLAG_OWN_CAR', 
           'FLAG_OWN_REALTY']

apps_clean[yn_cols] = apps_clean[yn_cols].replace(yn_map)

# identify encoded columns, change to categorical
cat_cols = ['NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START',
            'ORGANIZATION_TYPE',
            'FONDKAPREMONT_MODE',
            'HOUSETYPE_MODE',
            'TOTALAREA_MODE',
            'WALLSMATERIAL_MODE',
            'EMERGENCYSTATE_MODE']

cat_na_count = applications[cat_cols].isna().sum(axis=0)
cat_na_map = {cat: 'Unspecified' for cat in cat_na_count[cat_na_count > 0].index}
apps_clean = apps_clean.fillna(value=cat_na_map)
cat_labels = {}

for cat_col in cat_cols:
    cat_labels[cat_col] = apps_clean[cat_col].unique()
    apps_clean[cat_col] = pd.Categorical(apps_clean[cat_col], categories=cat_labels[cat_col])

# current home information
# TODO: pick one statistic for each home info category

# external data source
# TODO: impute with quantile of other scores
ext_src_cols = ['EXT_SOURCE_' + str(i + 1) for i in range(3)]

# age of car if owned
# TODO: impute car age with average, if client owns a car

# credit bureau requests
# TODO: impute all with zero, except past year with one

# annuity amount
# TODO: infer from amt_credit and amt_goods_price
