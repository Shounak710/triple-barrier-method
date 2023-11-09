import pandas as pd
import numpy as np

'''
Brief description of the methodology of Triple barrier labelling method:

1. Define a baseline price for a datapoint. Using the baseline volatility calculated from ewm method,
calculate the upper and lower barrier.
2. Vertical barrier will be calculated based on a predetermined number of days
3. Using the vertical barriers, we can break down the stock movement into discrete events
4. A primary model, like a Bollinger band model, will be used to generate buy/sell signals for the stock.
Our triple barrier labels help determine the accuracy of this signal within every event period.
5. If the Bollinger band model generates a buy signal for a time, but the lower barrier is hit first within the TBM,
this means the model's prediction would be counted as a false positive. If the primary model generates a sell signal,
but the upper barrier is hit within the TBM, this would be a false negative. We are concerned with maximizing our true
positives (recall), and minimizing the false positives (max precision). False negatives don't bother us that much here.
6. Using our primary model and the results from the classification done by TBM, we train a secondary model that will help
us to increase the precision and recall of our entire setup, and give better trading results.
'''

class TripleBarrier:
    '''Class for generating the three barrier levels'''

    def __init__(self, data, threshold):
        self.data = data
        self.threshold = threshold
    
    def get_horizontal_barriers(self, baseline_price):
        '''Define a baseline price (usually the closing price) for a datapoint.
        Get the barrier levels based on that.'''

        baseline_volatility = self.get_vol()
        upper_barrier = baseline_price * (1+baseline_volatility)
        lower_barrier = baseline_price * (1-baseline_volatility)

        return upper_barrier, lower_barrier
    
    def add_vertical_barriers(self, num_of_days=2):
        '''Add the third (vertical) barrier
        We take the events generated in get_threshold_events as the starting point for signal generation
        Number of days is the maximum number of days a trade can stay active
        This depends on the strategy and policies of the firm
        After the starting point the number of days is maximum days trade can stay active'''

        t_events = self.get_threshold_events()
        dates = pd.DatetimeIndex(data['Date'])
        t1 = dates.searchsorted(t_events + pd.Timedelta(days=num_of_days))
        t1 = t1[t1 < dates.shape[0]] # getting only times that fit within the date range
        t1 = pd.Series(dates[t1], index=t_events[:t1.shape[0]])

        return t1

    def get_vol(self, span=100, vol='ewm'):
        '''Get baseline volatility based on past data'''

        prices = self.data['Adj Close']
        daily_returns = (prices/prices.shift(1) - 1).dropna()
        squared_returns = daily_returns ** 2

        if vol == 'ewm':
            mean = squared_returns.ewm(span=span).mean()
        else:
            mean = squared_returns.mean()

        return np.sqrt(mean)
    
    def get_threshold_events(self):
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

            if s_neg < -self.threshold:
                s_neg = 0
                t_events.append(i)

            elif s_pos > self.threshold:
                s_pos = 0
                t_events.append(i)

        return pd.DatetimeIndex([data['Date'][x] for x in t_events])
    
class BollingerBandModel:
    # Class for calculating Bollinger band ranges

    def __init__(self, data, window, no_of_stdev=2):
        self.data = data
        self.window = window
        self.stdev_count = no_of_stdev # Number of standard deviations used for classification

    def bbands(self):
        mean, std = self.rolling_mean_and_std()
        
        upper_band = mean + (std*self.stdev_count)
        lower_band = mean - (std*self.stdev_count)

        return upper_band, lower_band

    def rolling_mean_and_std(self):
        prices = self.data['Adj Close']
        rolling_mean = prices.rolling(window=self.window).mean()
        rolling_std = prices.rolling(window=self.window).std()

        return rolling_mean, rolling_std
    
    # def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    #     # apply stop loss/profit taking, if it takes place before t1 (end of event)
    #     events_ = events.loc[molecule]
    #     out = events_[['t1']].copy(deep=True)
    #     if pt_sl[0] > 0:
    #         pt = pt_sl[0] * events_['trgt']
    #     else:
    #         pt = pd.Series(index=events.index)  # NaNs

    #     if pt_sl[1] > 0:
    #         sl = -pt_sl[1] * events_['trgt']
    #     else:
    #         sl = pd.Series(index=events.index)  # NaNs

    #     for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
    #         df0 = close[loc:t1]  # path prices
    #         df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
    #         out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
    #         out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking

    #     return out
    
    # def get_events(close, t_events, pt_sl, target, min_ret, num_threads, 
    #           vertical_barrier_times=False, side=None):
    #     # 1) Get target
    #     target = target.loc[target.index.intersection(t_events)]
    #     target = target[target > min_ret]  # min_ret

    #     # 2) Get vertical barrier (max holding period)
    #     if vertical_barrier_times is False:
    #         vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    #     # 3) Form events object, apply stop loss on vertical barrier
    #     if side is None:
    #         side_ = pd.Series(1., index=target.index)
    #         pt_sl_ = [pt_sl[0], pt_sl[0]]
    #     else:
    #         side_ = side.loc[target.index]
    #         pt_sl_ = pt_sl[:2]

    #     events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_},
    #                         axis=1)
    #     events = events.dropna(subset=['trgt'])

    #     # Apply Triple Barrier
    #     df0 = apply_pt_sl_on_t1(close, events, pt_sl_, )
    #     df0 = mpPandasObj(func=apply_pt_sl_on_t1,
    #                     pd_obj=('molecule', events.index),
    #                     num_threads=num_threads,
    #                     close=close,
    #                     events=events,
    #                     pt_sl=pt_sl_)

    #     events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan

    #     if side is None:
    #         events = events.drop('side', axis=1)

    #     return events
    
data = pd.read_csv('data/TSM.csv')
tbcl = TripleBarrier(data = data, threshold = 2)

bb = BollingerBandModel(data=data, window=50, no_of_stdev=2)
print(bb.bbands()[0])