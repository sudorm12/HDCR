import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from soft_impute import SoftImpute
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

applications = pd.read_csv('application_train.csv', index_col="SK_ID_CURR")
apps_clean = applications.copy()

# TODO: track rows with high number of na values to compare at end

# change y/n columns to boolean
yn_map = {'Y': 1,
          'N': 0}

yn_cols = ['FLAG_OWN_CAR', 
           'FLAG_OWN_REALTY']

apps_clean[yn_cols] = apps_clean[yn_cols].replace(yn_map)

# identify encoded columns, fill na with unspecified and change to categorical
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

# use smart imputation and pca on current home information
stat_suffixes = ['_AVG', '_MEDI', '_MODE']
stat_cols = [col[:-4] for col in applications.columns[applications.columns.str.contains(stat_suffixes[0])]]
all_stat_cols = [st + sf for st, sf in itertools.product(stat_cols, stat_suffixes)]

X = applications[all_stat_cols].values

clf = SoftImpute()
clf.fit(X)
apps_clean[all_stat_cols] = clf.predict(X)

# convert categorical home info columns to one-hot encoded
stat_cat_cols = ['FONDKAPREMONT_MODE',
                 'HOUSETYPE_MODE',
                 'WALLSMATERIAL_MODE',
                 'EMERGENCYSTATE_MODE']

full_home_stats = apps_clean[all_stat_cols].join(pd.get_dummies(apps_clean[stat_cat_cols]))

# TODO: may want to set a selectable threshold for explained variance
pca_components = 15
cols = ['CURR_HOME_' + str(pca_col) for pca_col in range(pca_components)]
st_pca = PCA(n_components=pca_components)

home_stats_pca = pd.DataFrame(st_pca.fit_transform(full_home_stats), index=full_home_stats.index, columns=cols)
# use st_pca.fit on out of sample data

apps_clean = apps_clean.join(home_stats_pca)
apps_clean = apps_clean.drop(all_stat_cols, axis=1)
apps_clean = apps_clean.drop(stat_cat_cols, axis=1)

# age of car if owned
# TODO: impute car age with average, if client owns a car

# impute all credit bureau requests with zero, except past year with one
app_credit_cols = apps_clean.columns[apps_clean.columns.str.contains('AMT_REQ_CREDIT_BUREAU')]
apps_clean['AMT_REQ_CREDIT_BUREAU_YEAR'] = apps_clean['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(1)
apps_clean[app_credit_cols] = apps_clean[app_credit_cols].fillna(0)

# infer goods price and annuity amount from credit using linear regression
# TODO: clean up to allow fit on out of sample data
amt_lr_rows = apps_clean['AMT_GOODS_PRICE'].notna()
x = apps_clean.loc[amt_lr_rows]['AMT_CREDIT']
y = apps_clean.loc[amt_lr_rows]['AMT_GOODS_PRICE']

lr = LinearRegression()
lr.fit(x.values.reshape(-1, 1), y)

amt_fill_rows = apps_clean['AMT_GOODS_PRICE'].isna()
x = apps_clean.loc[amt_fill_rows]['AMT_CREDIT']
y = lr.predict(x.values.reshape(-1, 1))

apps_clean.loc[amt_fill_rows, 'AMT_GOODS_PRICE'] = y

amt_lr_rows = apps_clean['AMT_ANNUITY'].notna()
x = apps_clean.loc[amt_lr_rows][['AMT_CREDIT', 'AMT_GOODS_PRICE']]
y = apps_clean.loc[amt_lr_rows]['AMT_ANNUITY']

lr = LinearRegression()
lr.fit(x.values, y)

amt_fill_rows = apps_clean['AMT_ANNUITY'].isna()
x = apps_clean.loc[amt_fill_rows][['AMT_CREDIT', 'AMT_GOODS_PRICE']]
y = lr.predict(x.values)

apps_clean.loc[amt_fill_rows, 'AMT_ANNUITY'] = y

# basic mean imputation of remaining na values
mean_imp_cols = apps_clean.columns[apps_clean.isna().sum(axis=0) > 0]
mean_imp_means = apps_clean[mean_imp_cols].mean()
apps_clean[mean_imp_cols] = apps_clean[mean_imp_cols].fillna(mean_imp_means)

# scale columns with numerical data
num_cols = []

apps_clean[num_cols] = apps_clean[num_cols].fillna(apps_clean[num_cols].mean())

scaler = StandardScaler()
apps_clean[num_cols] = scaler.fit_transform(apps_clean[num_cols])
# use scaler.fit on out-of-sample data

# read in credit bureau data
bureau = pd.read_csv('bureau.csv')
bureau_clean = bureau.copy()

# convert categorical columns to Categorical dtype
cat_cols = ['CREDIT_ACTIVE',
            'CREDIT_CURRENCY',
            'CREDIT_TYPE']

# TODO: find a nicer way to share the cat_labels dict
cat_labels = {}

for cat_col in cat_cols:
    cat_labels[cat_col] = bureau_clean[cat_col].unique()
    bureau_clean[cat_col] = pd.Categorical(bureau_clean[cat_col], categories=cat_labels[cat_col])

# count the number of each type of loan for each applicant
credit_types = bureau_clean.groupby(['SK_ID_CURR', 'CREDIT_TYPE']).size().unstack().fillna(0)

# sum up the numerical data for each applicant for a simple summary
bureau_summary = bureau.groupby('SK_ID_CURR').sum()
