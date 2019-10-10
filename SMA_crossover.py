"""
Author: TuanDo
Description: SMA cross over strategy.
Based on: Closed price, SMA 50, SMA 200
If Price > SMA 50 > SMA 200, we will buy those assets.
"""

# Import Libraries
import backtrader as bt
import numpy as np
import pandas as pd

from datetime import datetime
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from numpy.linalg import inv,pinv
from scipy.optimize import minimize

strategies = ['M_Max_sharpe', 'Risk_parity']
strategy = strategies[0]


# Strategy's class
class SMA_strat(bt.Strategy):
    params = (
        ('sma_50', 50),
        ('sma_200', 200),
        ('oneplot', True)
    )

    # Initialize parameters for strategy
    def __init__(self):
        self.inds = dict()
        self.day_counter = 0
        self.counter = 0
        self.counter_period = 200
        self.selected_assets =[]
        for i, d in enumerate(self.datas):
            self.inds[d] = dict()
            self.inds[d]['sma_50'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.p.sma_50
            )
            self.inds[d]['sma_200'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.p.sma_200
            )
            if i > 0:
                if self.p.oneplot==True:
                    d.plotinfo.plotmaster = self.datas[0]

    def next(self):
        # Pass counter_period days to compute return
        if self.counter < self.counter_period:
            self.counter += 1
        else:
            # Get data to dataframe in order to feed Pyopt
            appended_data =[]
            for i, d in enumerate(self.datas):
                dt, dn = self.datetime.date(), d._name
                get = lambda mydata: mydata.get(0, self.counter_period)
                time = [d.num2date(x) for x in get(d.datetime)]
                df = pd.DataFrame({dn:get(d.close)}, index = time)
                appended_data.append(df)
            df = pd.concat(appended_data, axis=1) # df is dataframe of n assets

            for i, d in enumerate(self.datas):
                dt, dn = self.datetime.date(), d._name
                if d.close[0] > self.inds[d]['sma_50'][0] and self.inds[d]['sma_50'][0] > self.inds[d]['sma_200']:
                    if dn in self.selected_assets:
                        pass
                    else:
                        self.selected_assets.append(dn)
                else:
                    if dn in self.selected_assets:
                        self.selected_assets.remove(dn)

            # Create dataframe of selected_assets portfolio
            portfolio_today = df[self.selected_assets]

            # Because there are some days having no assets in portfolio may cause error
            if strategy == 'Risk_parity':
                x_t = [0.25, 0.25, 0.25, 0.25]

                cons = ({'type': 'eq', 'fun': total_weight_constraint},
                        {'type': 'ineq', 'fun': long_only_constraint})

                res = minimize(risk_budget_objective, w0, args=[V, x_t], method='SLSQP', constraints=cons,
                               options={'disp': True})

                w_rb = np.asmatrix(res.x)
            elif strategy == 'M_Max_sharpe':
                try:
                    mu = mean_historical_return(portfolio_today)
                    S = CovarianceShrinkage(portfolio_today).ledoit_wolf()
                    ef = EfficientFrontier(mu, S)
                    weights = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()

                    # Rebalance monthly
                    if self.day_counter % 24 == 0:
                        for key, value in cleaned_weights.items():
                            self.order_target_percent(key, target=value)
                            print('on {} asset of portfolio is {} with value {}'.format(
                                self.datetime.date(), key, value
                            ))
                    self.day_counter +=1
                except:
                    pass

    def notify_trade(self, trade):
        dt = self.data.datetime.date()
        if trade.isclosed:
            print('{} {} Closed: PnL Gross {}, Net {}'.format(
                                                dt,
                                                trade.data._name,
                                                round(trade.pnl,2),
                                                round(trade.pnlcomm,2)))


# Feed data
class OandaCSVData(bt.feeds.GenericCSVData):
    params = (
        ('nullvalue', float('NaN')),
        ('dtformat', '%Y-%m-%dT%H:%M:%S.%fZ'),
        ('datetime', 6),
        ('time', -1),
        ('open', 5),
        ('high', 3),
        ('low', 4),
        ('close', 1),
        ('volume', 7),
        ('openinterest', -1),
    )

# Variable for our starting cash
startcash = 100000

# Create an instance of cerebro
cerebro = bt.Cerebro()

# Add strategy
cerebro.addstrategy(SMA_strat, oneplot=False)

#create our data list
datalist = [
    ('CAD_CHF-2005-2017-D1.csv', 'CADCHF'), #[0] = Data file, [1] = Data name
    ('EUR_USD-2005-2017-D1.csv', 'EURUSD'),
    ('GBP_AUD-2005-2017-D1.csv', 'GBPAUD'),
]

#Loop through the list adding to cerebro.
for i in range(len(datalist)):
    data = OandaCSVData(dataname=datalist[i][0])
    cerebro.adddata(data, name=datalist[i][1])

# Set our desired cash start
cerebro.broker.setcash(startcash)

# Run over everything
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
results = cerebro.run()

print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

#Get final portfolio Value
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash

#Print out the final result
print('Final Portfolio Value: ${}'.format(portvalue))
print('P/L: ${}'.format(pnl))

#Finally plot the end results
cerebro.plot(style='candlestick')

