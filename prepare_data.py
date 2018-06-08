import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from soft_impute import SoftImpute
from sklearn.decomposition import PCA

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

# use smart imputation and pca on current home information
stat_suffixes = ['_AVG', '_MEDI', '_MODE']
stat_cols = [col[:-4] for col in applications.columns[applications.columns.str.contains(stat_suffixes[0])]]
all_stat_cols = [st + sf for st, sf in itertools.product(stat_cols, stat_suffixes)]

X = applications[all_stat_cols].values

clf = SoftImpute()
clf.fit(X)
imputed = clf.predict(X)

apps_clean[all_stat_cols] = imputed

stat_cat_cols = ['FONDKAPREMONT_MODE',
                 'HOUSETYPE_MODE',
                 'WALLSMATERIAL_MODE',
                 'EMERGENCYSTATE_MODE']

full_home_stats = apps_clean[all_stat_cols].join(pd.get_dummies(apps_clean[stat_cat_cols]))
full_home_stats.head().T

# TODO: may want to set a selectable threshold for explained variance
pca_components = 15
cols = ['CURR_HOME_' + str(pca_col) for pca_col in range(pca_components)]
st_pca = PCA(n_components=pca_components)

home_stats_pca = pd.DataFrame(st_pca.fit_transform(full_home_stats), index=full_home_stats.index, columns=cols)

apps_clean = apps_clean.join(home_stats_pca)
apps_clean = apps_clean.drop(all_stat_cols, axis=1)
apps_clean = apps_clean.drop(stat_cat_cols, axis=1)

# TODO: replace categorical home information with one hot encoding

# external data source
# TODO: impute with quantile of other scores
ext_src_cols = ['EXT_SOURCE_' + str(i) for i in range(1, 4)]

# age of car if owned
# TODO: impute car age with average, if client owns a car

# credit bureau requests
# TODO: impute all with zero, except past year with one

# annuity amount
# TODO: infer from amt_credit and amt_goods_price

# scale columns with numerical data
num_cols = []

apps_clean[num_cols] = apps_clean[num_cols].fillna(apps_clean[num_cols].mean())

scaler = StandardScaler()
apps_clean[num_cols] = scaler.fit_transform(apps_clean[num_cols])
# just use scaler.fit on out-of-sample data
