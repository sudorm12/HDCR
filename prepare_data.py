import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from soft_impute import SoftImpute
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class HCDALoader:
    def __init__(self, data_dir='data'):
        self._curr_home_imputer = SoftImpute()
        self._amt_gp_lr = LinearRegression()
        self._amt_an_lr = LinearRegression()
        self._st_pca = None
        self._num_scaler = StandardScaler()

        self._applications = pd.read_csv('{}/application_train.csv'.format(data_dir), index_col="SK_ID_CURR")
        self._bureau_summary = self.read_bureau()
        self._previous_summary = self.read_previous_application()

    def load_train(self):
        # load each of the available data tables
        applications_train = self.read_applications_train()
        
        # join the dataframes together and fill nas with zeros
        joined = applications_train.join(self._bureau_summary, rsuffix='_BUREAU').join(self._previous_summary, rsuffix='_PREVIOUS')
        full_data = joined.combine_first(joined.select_dtypes(include=[np.number]).fillna(0))

        # TODO: scale numeric columns before returning
        return full_data

    def read_applications_train(self, split_index=None):
        if split_index is None:
            apps_clean = self._applications.copy()
        else:
            apps_clean = self._applications.iloc[split_index].copy()

        # TODO: track rows with high number of na values to compare at end

        # change y/n columns to boolean
        yn_cols = ['FLAG_OWN_CAR', 
                   'FLAG_OWN_REALTY']

        apps_clean[yn_cols] = self._yn_cols_to_boolean(apps_clean, yn_cols) #apps_clean[yn_cols].replace(yn_map)

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

        apps_clean = self._cat_data(apps_clean, cat_cols)

        # use smart imputation and pca on current home information
        stat_suffixes = ['_AVG', '_MEDI', '_MODE']
        stat_cols = [col[:-4] for col in apps_clean.columns[apps_clean.columns.str.contains(stat_suffixes[0])]]
        all_stat_cols = [st + sf for st, sf in itertools.product(stat_cols, stat_suffixes)]

        X = apps_clean[all_stat_cols].values

        self._curr_home_imputer.fit(X)
        apps_clean[all_stat_cols] = self._curr_home_imputer.predict(X)

        # convert categorical home info columns to one-hot encoded
        stat_cat_cols = ['FONDKAPREMONT_MODE',
                         'HOUSETYPE_MODE',
                         'WALLSMATERIAL_MODE',
                         'EMERGENCYSTATE_MODE']

        full_home_stats = apps_clean[all_stat_cols].join(pd.get_dummies(apps_clean[stat_cat_cols]))

        # TODO: may want to set a selectable threshold for explained variance
        pca_components = 15
        cols = ['CURR_HOME_' + str(pca_col) for pca_col in range(pca_components)]
        self._st_pca = PCA(n_components=pca_components)

        home_stats_pca = pd.DataFrame(self._st_pca.fit_transform(full_home_stats), index=full_home_stats.index, columns=cols)
        # use self._st_pca.fit on out of sample data

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

        self._amt_gp_lr.fit(x.values.reshape(-1, 1), y)

        amt_fill_rows = apps_clean['AMT_GOODS_PRICE'].isna()
        x = apps_clean.loc[amt_fill_rows]['AMT_CREDIT']
        y = self._amt_gp_lr.predict(x.values.reshape(-1, 1))

        apps_clean.loc[amt_fill_rows, 'AMT_GOODS_PRICE'] = y

        amt_lr_rows = apps_clean['AMT_ANNUITY'].notna()
        x = apps_clean.loc[amt_lr_rows][['AMT_CREDIT', 'AMT_GOODS_PRICE']]
        y = apps_clean.loc[amt_lr_rows]['AMT_ANNUITY']

        self._amt_an_lr.fit(x.values, y)

        amt_fill_rows = apps_clean['AMT_ANNUITY'].isna()
        x = apps_clean.loc[amt_fill_rows][['AMT_CREDIT', 'AMT_GOODS_PRICE']]
        y = self._amt_an_lr.predict(x.values)

        apps_clean.loc[amt_fill_rows, 'AMT_ANNUITY'] = y

        # basic mean imputation of remaining na values
        mean_imp_cols = apps_clean.columns[apps_clean.isna().sum(axis=0) > 0]
        mean_imp_means = apps_clean[mean_imp_cols].mean()
        apps_clean[mean_imp_cols] = apps_clean[mean_imp_cols].fillna(mean_imp_means)

        # scale columns with numerical data
        # TODO: find a way to automatically detect numeric columns
        # df.select_dtypes
        num_cols = []

        #apps_clean[num_cols] = apps_clean[num_cols].fillna(apps_clean[num_cols].mean())

        #apps_clean[num_cols] = self._num_scaler.fit_transform(apps_clean[num_cols])
        # use scaler.fit on out-of-sample data

        return apps_clean

    def read_bureau(self):
        # read in credit bureau data
        bureau = pd.read_csv('{}/bureau.csv'.format(data_dir))
        bureau_clean = bureau.copy()

        # convert categorical columns to Categorical dtype
        cat_cols = ['CREDIT_ACTIVE',
                    'CREDIT_CURRENCY',
                    'CREDIT_TYPE']

        # TODO: find a nicer way to share the cat_labels dict
        bureau_clean = self._cat_data(bureau_clean, cat_cols)

        # sum up the numerical data for each applicant for a simple summary
        bureau_num_summary = bureau.groupby('SK_ID_CURR').sum()

        # count the number of each type of loan for each applicant
        credit_types = bureau_clean.groupby(['SK_ID_CURR', 'CREDIT_TYPE']).size().unstack().fillna(0)

        # count the number of open and closed loans for each applicant
        credit_active = bureau_clean.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE']).size().unstack().fillna(0)

        # combine the summary dataframes
        bureau_summary = bureau_num_summary.join([credit_active, credit_types]).fillna(0)
        return bureau_summary

    def read_previous_application(self):
        previous_application = pd.read_csv('{}/previous_application.csv'.format(dat_dir))
        previous_clean = previous_application.copy()

        # convert categorical columns to Categorical dtype
        cat_cols = ['NAME_CONTRACT_TYPE',
                    'WEEKDAY_APPR_PROCESS_START',
                    'NAME_CASH_LOAN_PURPOSE',
                    'NAME_CONTRACT_STATUS',
                    'NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON',
                    'NAME_TYPE_SUITE',
                    'NAME_CLIENT_TYPE',
                    'NAME_GOODS_CATEGORY',
                    'NAME_PORTFOLIO',
                    'NAME_PRODUCT_TYPE',
                    'CHANNEL_TYPE',
                    'NAME_SELLER_INDUSTRY',
                    'NAME_YIELD_GROUP',
                    'PRODUCT_COMBINATION']

        previous_clean = self._cat_data(previous_clean, cat_cols)

        # create numerical and categorical summaries and join them together
        prev_app_num_summary = previous_clean.groupby('SK_ID_CURR').sum()
        prev_app_cat_summary = pd.get_dummies(previous_clean[cat_cols + ['SK_ID_CURR']]).groupby('SK_ID_CURR').sum()
        prev_app_summary = prev_app_num_summary.join(prev_app_cat_summary)
        return prev_app_summary

    def applications_train_index(self):
        return self._applications.index

    def _yn_cols_to_boolean(self, df, cols):
        yn_map = {'Y': 1,
                'N': 0}
        return df[cols].replace(yn_map)

    def _cat_data(self, df, cols):
        df_clean = df.copy()
        cat_na_count = df_clean[cols].isna().sum(axis=0)
        cat_na_map = {cat: 'Unspecified' for cat in cat_na_count[cat_na_count > 0].index}
        df_clean = df_clean.fillna(value=cat_na_map)

        cat_labels = {}
        for cat_col in cols:
            cat_labels[cat_col] = df_clean[cat_col].unique()
            df_clean[cat_col] = pd.Categorical(df_clean[cat_col], categories=cat_labels[cat_col])

        return df_clean
