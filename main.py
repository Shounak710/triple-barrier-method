import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('TSM.csv')

def get_daily_vol(close_prices):
    '''Get baseline volatility based on past data'''

    daily_returns = (close_prices/close_prices.shift(1) - 1).dropna()
    squared_returns = daily_returns ** 2
    ewma_vol = squared_returns.ewm(span=100).mean()
    baseline_vol = np.sqrt(ewma_vol)

    return baseline_vol

def get_threshold_events(data, threshold):
    '''Generate events using a symmetric CUSUM filter.
       First we get the log returns for the close prices.
       Using these, we implement a CUSUM filter, such that new events are generated
       each time the cumulative returns cross a predefined threshold.
       Returns all the dates when the threshold was crossed'''

    t_events = []
    s_pos = 0
    s_neg = 0

    diff = np.log(data['Close']).diff().dropna()
    
    for i in diff.index[1:]:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg - diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    return pd.DatetimeIndex([data['Date'][x] for x in t_events])

def add_vertical_barriers(t_events, close, num_of_days=1):
    '''Add the third (vertical) barrier
       We take the events generated in get_threshold_events as the starting point for signal generation
       Number of days is the maximum number of days a trade can stay active
       This depends on the strategy and policies of the firm
       After the starting point the number of days is maximum days trade can stay active'''

    dates = pd.DatetimeIndex(data['Date'])
    t1 = dates.searchsorted(t_events + pd.Timedelta(days=num_of_days))
    t1 = t1[t1 < dates.shape[0]] # getting only times that fit within the date range
    t1 = pd.Series(dates[t1], index=t_events[:t1.shape[0]])
    return t1

ev = get_threshold_events(data, 0.02)
print(ev)
print(add_vertical_barriers(ev, data['Close']))