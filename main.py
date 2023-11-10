import pandas as pd
import numpy as np
from datetime import timedelta

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
us to increase the precision and recall of our entire strategy, and give better trading results.
'''

class TripleBarrier:
    '''Class for generating the three barrier levels'''

    def __init__(self, data, threshold, num_of_days):
        self.data = data
        self.threshold = threshold
        self.num_of_days = num_of_days # Max number of days a trade can stay active
    
    def get_horizontal_barriers(self, baseline_price):
        '''Define a baseline price (usually the closing price) for a datapoint.
        Get the barrier levels based on that.'''

        baseline_volatility = self.get_vol()
        upper_barrier = baseline_price * (1+baseline_volatility)
        lower_barrier = baseline_price * (1-baseline_volatility)

        return upper_barrier, lower_barrier
    
    def get_vertical_barriers(self):
        '''Add the third (vertical) barrier
        We take the events generated in get_threshold_events as the starting point for signal generation
        Number of days is the maximum number of days a trade can stay active
        This depends on the strategy and policies of the firm
        After the starting point the number of days is maximum days trade can stay active'''

        t_events = self.get_threshold_events()
        dates = pd.DatetimeIndex(data['Date'])
        t1 = dates.searchsorted(t_events + pd.Timedelta(days=self.num_of_days))
        t1 = t1[t1 < dates.shape[0]] # getting only times that fit within the date range
        t1 = pd.Series(dates[t1], index=t_events[:t1.shape[0]])

        return t1
    
    def generate_labels(self):
        '''
        1. Separate data according to the vertical labels
        2. In each partition, get the data points that are above the upper barrier, or below the lower barrier
        3. Depending on whether the upper barrier was breached first or the lower barrier, label +1 or -1
        4. If no breaches are detected, label it 0
        '''

        labels = []

        vertical_barriers = self.get_vertical_barriers().dropna().sort_values(ascending=True)
        
        for i in range(0, len(vertical_barriers)):
            # Getting price data from threshold, to the barrier.

            prices = data[
                (pd.to_datetime(data['Date']) >= vertical_barriers.index[i]) & 
                (pd.to_datetime(data['Date']) <= vertical_barriers[i])
            ]

            upper_barrier, lower_barrier = self.get_horizontal_barriers(prices['Adj Close'].iloc[0])

            print("UB: ", upper_barrier, "LB: ", lower_barrier)
            print(prices)
            outlying_prices = prices[
                (prices['Adj Close'] < lower_barrier) | (prices['Adj Close'] > upper_barrier)
            ]

            if len(outlying_prices) == 0:
                labels.append(0)
            elif outlying_prices.iloc[0]['Adj Close'] > upper_barrier:
                labels.append(1)
            else:
                labels.append(-1)

        return labels

    def get_vol(self, span=100, vol='ewm'):
        '''Get baseline volatility based on past data'''

        prices = self.data['Adj Close']
        daily_returns = (prices/prices.shift(1) - 1).dropna()
        squared_returns = daily_returns ** 2

        if vol == 'ewm':
            mean = squared_returns.ewm(span=span).mean().mean()
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
    
data = pd.read_csv('data/TSM.csv')
tbcl = TripleBarrier(data = data, threshold = 1, num_of_days=100)
print(tbcl.generate_labels())

'''
References:
1. https://www.sefidian.com/2021/06/26/labeling-financial-data-for-machine-learning/
2. https://colab.research.google.com/drive/1FmnCJ1CI98khBu88kezLXKqvS7U8Nw_h?usp=sharing
3. ChatGPT
'''