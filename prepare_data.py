import pandas as pd
import numpy as np
import logging
import itertools
from sklearn.preprocessing import StandardScaler
from soft_impute import SoftImpute
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.sparse import csr_matrix
from loader import DataLoader


class HCDRDataLoader(DataLoader):
    def __init__(self, cc_tmax=25, bureau_tmax=25, pos_tmax=25, data_dir='data', load_time_series=True):
        super().__init__()
        logging.debug('Initializing data loader')

        self._data_dir = data_dir
        self._cc_tmax = cc_tmax
        self._bureau_tmax = bureau_tmax
        self._pos_tmax = pos_tmax

        self._curr_home_imputer = SoftImpute()
        self._amt_gp_lr = LinearRegression()
        self._amt_an_lr = LinearRegression()
        self._st_pca = None
        self._num_scaler = StandardScaler()

        self._mean_imp_cols = None
        self._mean_imp_means = None

        self._applications = pd.read_csv('{}/application_train.csv'.format(data_dir), index_col="SK_ID_CURR")
        self._applications_test = pd.read_csv('{}/application_test.csv'.format(data_dir), index_col="SK_ID_CURR")
        self.pca_all_home_stats()
        self._bureau_summary = self.read_bureau()
        self._previous_summary = self.read_previous_application()
        self._bureau_balance_summary = self.bureau_balance_summary()
        self._cc_balance_summary = self.cc_balance_summary()
        self._pos_cash_summary = self.pos_cash_summary()

        self._input_shape = None
        self._load_time_series = load_time_series

    def get_index(self):
        return self._applications.index

    def get_test_index(self):
        return self._applications_test.index

    def load_train_data(self, split_index=None, fit_transform=True, load_time_series=None):
        if load_time_series is None:
            load_time_series = self._load_time_series

        # load each of the available data tables
        applications = self.read_applications(split_index, fit_transform=fit_transform)
        joined_train = (applications
                        .join(self._bureau_summary, rsuffix='_BUREAU')
                        .join(self._previous_summary, rsuffix='_PREVIOUS')
                        .join(self._bureau_balance_summary, rsuffix='_BUREAU_BALANCE')
                        .join(self._cc_balance_summary, rsuffix='_CC_BALANCE')
                        .join(self._pos_cash_summary, rsuffix='_POS_CASH'))

        full_data_train = joined_train.combine_first(joined_train.select_dtypes(include=[np.number]).fillna(0))

        # split into features and target
        meta_data_train = full_data_train.drop('TARGET', axis=1)
        target_train = full_data_train['TARGET']

        # scale to zero mean and unit variance
        meta_data_train = self._num_scaler.fit_transform(meta_data_train.loc[:, meta_data_train.dtypes == np.number])
        meta_data_shape = tuple([meta_data_train.shape[1]])

        if load_time_series:
            cc_data_train = self.read_credit_card_balance(self._applications.index.values[split_index])
            bureau_data_train = self.read_bureau_balance(self._applications.index.values[split_index])
            pos_cash_data_train = self.read_pos_cash(self._applications.index.values[split_index])

            ts_data_shape = [tuple([self._cc_tmax, int(cc_data_train.shape[1] / self._cc_tmax)]),
                             tuple([self._bureau_tmax, int(bureau_data_train.shape[1] / self._bureau_tmax)]),
                             tuple([self._pos_tmax, int(pos_cash_data_train.shape[1] / self._pos_tmax)])]

            data_train = [meta_data_train, cc_data_train, bureau_data_train, pos_cash_data_train]
            self._input_shape = [meta_data_shape, *ts_data_shape]
        else:
            data_train = meta_data_train
            self._input_shape = meta_data_shape

        logging.debug(self._input_shape)

        return data_train, target_train

    def load_test_data(self, load_time_series=True):
        # load each of the available data tables
        applications = self.read_applications(split_index=None, fit_transform=False, test_data=True)
        joined_train = (applications
                        .join(self._bureau_summary, rsuffix='_BUREAU')
                        .join(self._previous_summary, rsuffix='_PREVIOUS'))
        meta_data_train = joined_train.combine_first(joined_train.select_dtypes(include=[np.number]).fillna(0))

        # scale to zero mean and unit variance
        meta_data_train = self._num_scaler.transform(meta_data_train.loc[:, meta_data_train.dtypes == np.number])
        meta_data_shape = tuple([meta_data_train.shape[1]])

        if load_time_series:
            cc_data_train = self.read_credit_card_balance(self.get_test_index().values)
            bureau_data_train = self.read_bureau_balance(self.get_test_index().values)
            pos_cash_data_train = self.read_pos_cash(self.get_test_index().values)

            ts_data_shape = [tuple([self._cc_tmax, int(cc_data_train.shape[1] / self._cc_tmax)]),
                             tuple([self._bureau_tmax, int(bureau_data_train.shape[1] / self._bureau_tmax)]),
                             tuple([self._pos_tmax, int(pos_cash_data_train.shape[1] / self._pos_tmax)])]

            data_train = [meta_data_train, cc_data_train, bureau_data_train, pos_cash_data_train]
            self._input_shape = [meta_data_shape, *ts_data_shape]
        else:
            data_train = meta_data_train
            self._input_shape = meta_data_shape

        # determine input shapes
        logging.debug(self._input_shape)

        return data_train

    def get_input_shape(self):
        return self._input_shape

    def read_applications(self, split_index=None, fit_transform=True, test_data=False):
        logging.debug('Preparing applications data...')
        if test_data:
            apps_clean = self._applications_test.copy()
        else:
            apps_clean = self._applications.copy()

        if split_index is not None:
            apps_clean = apps_clean.iloc[split_index]

        # track rows with high number of na values
        apps_clean['NA_COLS'] = apps_clean.isna().sum(axis=1)

        # change y/n columns to boolean
        yn_cols = ['FLAG_OWN_CAR', 
                   'FLAG_OWN_REALTY']

        apps_clean[yn_cols] = self._yn_cols_to_boolean(apps_clean, yn_cols)

        # identify encoded columns, fill na with unspecified and change to categorical
        apps_clean = self._cat_data_dummies(apps_clean)

        # impute all credit bureau requests with zero, except past year with one
        app_credit_cols = apps_clean.columns[apps_clean.columns.str.contains('AMT_REQ_CREDIT_BUREAU')]
        apps_clean['AMT_REQ_CREDIT_BUREAU_YEAR'] = apps_clean['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(1)
        apps_clean[app_credit_cols] = apps_clean[app_credit_cols].fillna(0)

        logging.debug('Performing linear regression on goods price and annuity amount...')
        # fit linear regression to infer goods price and annuity amount from credit using linear regression
        if fit_transform:
            amt_lr_rows = apps_clean['AMT_GOODS_PRICE'].notna()
            x = apps_clean.loc[amt_lr_rows]['AMT_CREDIT']
            y = apps_clean.loc[amt_lr_rows]['AMT_GOODS_PRICE']
            self._amt_gp_lr.fit(x.values.reshape(-1, 1), y)

        # use fitted linear regression to predict goods price
        amt_fill_rows = apps_clean['AMT_GOODS_PRICE'].isna()
        if sum(amt_fill_rows) > 0:
            x = apps_clean.loc[amt_fill_rows]['AMT_CREDIT']
            y = self._amt_gp_lr.predict(x.values.reshape(-1, 1))
            apps_clean.loc[amt_fill_rows, 'AMT_GOODS_PRICE'] = y

        # fit linear regression for annuity amount using credit amount and goods price]
        if fit_transform:
            amt_lr_rows = apps_clean['AMT_ANNUITY'].notna()
            x = apps_clean.loc[amt_lr_rows][['AMT_CREDIT', 'AMT_GOODS_PRICE']]
            y = apps_clean.loc[amt_lr_rows]['AMT_ANNUITY']
            self._amt_an_lr.fit(x.values, y)

        # use fitted linear regression to predict annuity amount
        amt_fill_rows = apps_clean['AMT_ANNUITY'].isna()
        if sum(amt_fill_rows) > 0:
            x = apps_clean.loc[amt_fill_rows][['AMT_CREDIT', 'AMT_GOODS_PRICE']]
            y = self._amt_an_lr.predict(x.values)
            apps_clean.loc[amt_fill_rows, 'AMT_ANNUITY'] = y

        # basic mean imputation of remaining na values
        if fit_transform:
            self._mean_imp_cols = apps_clean.columns[apps_clean.isna().sum(axis=0) > 0]
            self._mean_imp_means = apps_clean[self._mean_imp_cols].mean()

        apps_clean[self._mean_imp_cols] = apps_clean[self._mean_imp_cols].fillna(self._mean_imp_means)

        return apps_clean

    def pca_all_home_stats(self):
        apps_columns = self._applications.columns

        stat_suffixes = ['_AVG', '_MEDI', '_MODE']
        stat_cols = [col for col in apps_columns[apps_columns.str.contains('|'.join(stat_suffixes))]]

        stat_train = self._applications[stat_cols]
        stat_test = self._applications_test[stat_cols]
        stat_full = pd.concat([stat_train, stat_test])

        stat_full = self._cat_data_dummies(stat_full)
        stat_index = stat_full.index.values

        logging.debug('Performing soft impute on current home info...')
        self._curr_home_imputer.fit(stat_full.values)
        stat_full = self._curr_home_imputer.predict(stat_full.values)

        pca_components = 2
        pca_cols = ['CURR_HOME_' + str(pca_col) for pca_col in range(pca_components)]

        logging.debug('Running PCA on current home info...')
        self._st_pca = PCA(n_components=pca_components)
        self._st_pca.fit(stat_full)

        home_stats_pca = pd.DataFrame(self._st_pca.transform(stat_full),
                                      index=stat_index,
                                      columns=pca_cols)

        stat_pca_train = pd.DataFrame(home_stats_pca.loc[self.get_index().values],
                                      index=self.get_index().values,
                                      columns=pca_cols)
        stat_pca_test = pd.DataFrame(home_stats_pca.loc[self.get_test_index().values],
                                     index=self.get_test_index().values,
                                     columns=pca_cols)

        self._applications = self._applications.join(stat_pca_train)
        self._applications = self._applications.drop(stat_cols, axis=1)

        self._applications_test = self._applications_test.join(stat_pca_test)
        self._applications_test = self._applications_test.drop(stat_cols, axis=1)

    def read_bureau(self):
        # read in credit bureau data
        bureau = pd.read_csv('{}/bureau.csv'.format(self._data_dir))
        bureau_clean = bureau.copy()

        # convert categorical columns to Categorical dtype
        bureau_clean = self._cat_data_dummies(bureau_clean)

        # sum up the numerical data for each applicant for a simple summary
        bureau_summary = bureau_clean.groupby('SK_ID_CURR').sum()

        return bureau_summary

    def read_previous_application(self):
        previous_application = pd.read_csv('{}/previous_application.csv'.format(self._data_dir))
        previous_clean = previous_application.copy()

        # convert categorical columns to Categorical dtype
        previous_clean = self._cat_data_dummies(previous_clean)

        # create summary of the data and join them together
        prev_app_summary = previous_clean.groupby('SK_ID_CURR').sum()
        return prev_app_summary

    def read_credit_card_balance(self, sk_ids=None):
        # read cc balance csv and full list of id values
        logging.debug('Reading credit card balance file...')
        credit_card_balance = pd.read_csv('{}/credit_card_balance.csv'.format(self._data_dir))
        app_ix = self.get_index()

        # convert categorical columns to dummy values
        credit_card_balance = self._cat_data_dummies(credit_card_balance)

        # skim unused ids from input data
        if sk_ids is None:
            sk_ids = app_ix.values
        credit_card_balance = credit_card_balance[credit_card_balance['SK_ID_CURR'].isin(sk_ids)]

        # fill missing id values
        missing_ids = sk_ids[~np.isin(sk_ids, credit_card_balance['SK_ID_CURR'].unique())]
        missing_df = pd.DataFrame({'SK_ID_CURR': missing_ids})
        missing_df['MONTHS_BALANCE'] = -1

        # mix it all around
        logging.debug('Preparing credit card balance data...')
        cc_ts_summary = (credit_card_balance
                         .append(missing_df)
                         .drop(['SK_ID_PREV'], axis=1)
                         .fillna(0)
                         .groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).sum()
                         .unstack(level=0).reindex(np.arange(-self._cc_tmax, 0)).stack(dropna=False)
                         .swaplevel(0, 1).sort_index().unstack())

        logging.debug('Sparsifying...')
        cc_ts_sparse = csr_matrix(cc_ts_summary.fillna(0).values)

        logging.debug('Done')
        return cc_ts_sparse

    def cc_balance_summary(self):
        # read credit card balance csv
        cc_balance = pd.read_csv('data/credit_card_balance.csv')

        # convert categorical columns to dummy values
        cc_balance = self._cat_data_dummies(cc_balance)

        # group by id and aggregate statistics for each column
        cc_balance_sum = (cc_balance
                          .drop(['MONTHS_BALANCE', 'SK_ID_PREV'], axis=1)
                          .groupby('SK_ID_CURR')
                          .agg(['sum', 'min', 'max', 'mean']))
        cc_balance_sum.columns = ['_'.join(a) for a in itertools.product(*cc_balance_sum.columns.levels)]

        return cc_balance_sum

    def read_bureau_balance(self, sk_ids=None):
        # read bureau balance csv and full list of id values
        bureau_balance = pd.read_csv('{}/bureau_balance.csv'.format(self._data_dir))
        bureau = pd.read_csv('data/bureau.csv')
        id_xref = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
        app_ix = self.get_index()

        # merge bureau ids with application ids
        bureau_balance = bureau_balance.merge(id_xref).drop(['SK_ID_BUREAU'], axis=1)

        # convert categorical columns to dummy values
        bureau_balance = self._cat_data_dummies(bureau_balance)

        # skim unused ids from input data
        if sk_ids is None:
            sk_ids = app_ix.values
        bureau_balance = bureau_balance[bureau_balance['SK_ID_CURR'].isin(sk_ids)]

        # fill missing id values
        missing_ids = sk_ids[~np.isin(sk_ids, bureau_balance['SK_ID_CURR'].unique())]
        missing_df = pd.DataFrame({'SK_ID_CURR': missing_ids})
        missing_df['MONTHS_BALANCE'] = -1

        logging.debug('Preparing credit bureau balance data...')
        bureau_ts_summary = (bureau_balance
                             .append(missing_df)
                             .fillna(0)
                             .groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).sum()
                             .unstack(level=0).reindex(np.arange(-self._bureau_tmax, 0)).stack(dropna=False)
                             .swaplevel(0, 1).sort_index().unstack())

        logging.debug('Sparsifying...')
        bureau_sparse = csr_matrix(bureau_ts_summary.fillna(0).values)

        logging.debug('Done')
        return bureau_sparse

    def bureau_balance_summary(self):
        # read bureau balance csv and full list of id values
        bureau_balance = pd.read_csv('{}/bureau_balance.csv'.format(self._data_dir))
        # TODO: make id xref a class variable
        bureau = pd.read_csv('data/bureau.csv')
        id_xref = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
        app_ix = self.get_index()

        # merge bureau ids with application ids
        bureau_balance = bureau_balance.merge(id_xref).drop(['SK_ID_BUREAU'], axis=1)

        # convert categorical columns to dummy values
        bureau_balance = self._cat_data_dummies(bureau_balance)

        # group by id and sum for each column
        bureau_bal_sum = bureau_balance.drop('MONTHS_BALANCE', axis=1).groupby('SK_ID_CURR').agg(['sum'])
        bureau_bal_sum.columns = ['_'.join(col_name) for col_name in itertools.product(*bureau_bal_sum.columns.levels)]

        return bureau_bal_sum

    def read_pos_cash(self, sk_ids=None):
        # read pos cash csv and full list of id values
        pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
        bureau = pd.read_csv('data/bureau.csv')
        id_xref = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
        app_ix = self.get_index()

        pos_cash = pos_cash.merge(id_xref)
        pos_cash = self._cat_data_dummies(pos_cash).drop(['SK_ID_BUREAU', 'SK_ID_PREV'], axis=1)

        if sk_ids is None:
            sk_ids = app_ix.values
        pos_cash = pos_cash[pos_cash['SK_ID_CURR'].isin(sk_ids)]

        # fill missing id values
        missing_ids = sk_ids[~np.isin(sk_ids, pos_cash['SK_ID_CURR'].unique())]
        missing_df = pd.DataFrame({'SK_ID_CURR': missing_ids})
        missing_df['MONTHS_BALANCE'] = -1

        logging.debug('Preparing POS cash data...')
        pos_cash_summary = (pos_cash
                            .append(missing_df)
                            .fillna(0)
                            .groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).sum()
                            .unstack(level=0).reindex(np.arange(-self._pos_tmax, 0)).stack(dropna=False)
                            .swaplevel(0, 1).sort_index().unstack())

        logging.debug('Done')
        return pos_cash_summary.fillna(0).values

    def pos_cash_summary(self):
        # read pos cash csv and full list of id values
        pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
        bureau = pd.read_csv('data/bureau.csv')
        id_xref = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
        pos_cash = pos_cash.merge(id_xref).drop(['SK_ID_BUREAU', 'SK_ID_PREV'], axis=1)

        # merge bureau ids with application ids
        pos_cash = self._cat_data_dummies(pos_cash)

        agg_cols = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
        pos_cash_agg = pos_cash[[*agg_cols, 'SK_ID_CURR']].groupby('SK_ID_CURR').agg(['min', 'max', 'mean'])
        pos_cash_agg.columns = ['_'.join(a) for a in itertools.product(*pos_cash_agg.columns.levels)]

        pos_cash_sum = pos_cash.drop('MONTHS_BALANCE', axis=1).groupby('SK_ID_CURR').agg(['sum'])
        pos_cash_sum.columns = ['_'.join(a) for a in itertools.product(*pos_cash_sum.columns.levels)]

        pos_cash_summary = pos_cash_agg.join(pos_cash_sum)
        return pos_cash_summary
