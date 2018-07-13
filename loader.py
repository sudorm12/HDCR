import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def get_index(self):
        raise NotImplementedError

    def load_train_data(self, split_index=None, fit_transform=True):
        raise NotImplementedError

    def load_test_data(self):
        raise NotImplementedError

    def get_input_shape(self):
        raise NotImplementedError

    def load_train_val(self, train_index, val_index):
        train_data, train_target = self.load_train_data(split_index=train_index, fit_transform=True)
        val_data, val_target = self.load_train_data(split_index=val_index, fit_transform=False)
        return train_data, train_target, val_data, val_target

    def load_train_test(self):
        train_data, train_target = self.load_train_data(fit_transform=True)
        test_data, test_target = self.load_test_data()
        return train_data, train_target, test_data, test_target

    @staticmethod
    def _yn_cols_to_boolean(df, cols):
        yn_map = {'Y': 1,
                  'N': 0}
        return df[cols].replace(yn_map)

    @staticmethod
    def _cat_data(df):
        df_clean = df.copy()

        # detect columns with dtype 'object'
        cols = df.columns[df.dtypes == 'object']

        # fill na values with 'Unspecified'
        cat_na_count = df_clean[cols].isna().sum(axis=0)
        cat_na_map = {cat: 'Unspecified' for cat in cat_na_count[cat_na_count > 0].index}
        df_clean = df_clean.fillna(value=cat_na_map)

        # convert columns to categorical
        cat_labels = {}
        for cat_col in cols:
            cat_labels[cat_col] = df_clean[cat_col].unique()
            df_clean[cat_col] = pd.Categorical(df_clean[cat_col], categories=cat_labels[cat_col])

        return df_clean

    def _cat_data_dummies(self, df):
        df_clean = df.copy()

        # convert categorical columns to Categorical dtype
        df_clean = self._cat_data(df_clean)

        # convert Categorical columns to dummy columns
        df_clean = pd.get_dummies(df_clean)

        # return dataframe with dummy columns
        return df_clean
