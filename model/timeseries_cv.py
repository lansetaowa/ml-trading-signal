import numpy as np

np.random.seed(42)

def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 train_length=24*7*4,
                 test_length=24,
                 lookahead=1,
                 date_idx='datetime',
                 shuffle=False):
        self.lookahead = lookahead
        self.test_length = test_length
        self.train_length = train_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, x_matrix):

        unique_dates = x_matrix.index.get_level_values(self.date_idx).unique()
        n_dates = len(unique_dates)
        days = sorted(unique_dates, reverse=True)

        n_splits= (n_dates - self.train_length - self.lookahead) // self.test_length

        split_idx = []
        for i in range(n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = x_matrix.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))

            yield train_idx.to_numpy(), test_idx.to_numpy()

if __name__ == '__main__':
    import sqlite3
    import pandas as pd
    price_all = pd.read_sql_query(
        f"SELECT * FROM kline ORDER BY datetime",
        sqlite3.connect('../data/crypto_data.db'),
        parse_dates=['datetime']
    )

    price_all.set_index(['symbol', 'datetime'], inplace=True)

    print(price_all.index.get_level_values('datetime').max())

    train_period_length = 24 * 7 * 4
    test_period_length = 24
    lookahead = 1

    cv_test = MultipleTimeSeriesCV(
        train_length=train_period_length,
        test_length=test_period_length,
        lookahead=lookahead)

    i = 0
    for train_idx, test_idx in cv_test.split(x_matrix=price_all):

        train = price_all.iloc[train_idx]
        train_dates = train.index.get_level_values('datetime')

        test = price_all.iloc[test_idx]
        test_dates = test.index.get_level_values('datetime')

        print(train.shape[0],
              train_dates.min(), train_dates.max(),
              test.shape[0],
              test_dates.min(), test_dates.max())

        i += 1

        if i == 5:
            break