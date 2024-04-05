import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.stattools import adfuller

# Oanda libraries
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades

# Configs
# INSTRUMENT_PAIR = ['USD_CAD', 'USD_JPY']
INSTRUMENT_PAIR = ['EUR_JPY', 'GBP_JPY']
ACCESS_TOKEN = '46bb4716cb283b1edcc7b9cb4acd8e1f-f4b2c2eb1d4d47f653b1682fbb419315'
ACCOUNT_ID = '101-003-26638180-007'
# Granularity of the data to use and train the model
GRANULARITY = 'M1'
# Amount to trade (currency determined by the account's currenacy)
AMOUNT = 1000
# Number of candles to fetch in a request
CANDLE_COUNT = 5000
# Get data 30 days ago for training the model
NUM_DAYS_DATA_FOR_MODELS = 30
# Set how often to train the model
MODEL_EXPIRE_SECONDS = 10

SHORT_SIGNAL = 0
SHORT_EXIT = 1
LONG_SIGNAL = 2
LONG_EXIT = 3

class PairsTrader:
  def __init__(self):
    self.client = oandapyV20.API(access_token=ACCESS_TOKEN)
    self.model = None
    self.inst1_trade_status = None
    self.inst2_trade_status = None
    self.inst1_last_close_log = None
    self.inst2_last_close_log = None
    # For calculating how much to long/short
    self.inst1_last_close = None
    self.inst2_last_close = None
    self.signal = None
    self.heartbeat = 1

  def get_candles(self, instrument, from_dt):
    params = {'count': CANDLE_COUNT, 'granularity': GRANULARITY, 'from': from_dt}
    r = instruments.InstrumentsCandles(instrument, params=params)
    candles = self.client.request(r)['candles']
    return self._to_candles_df(candles, from_dt is not None)

  def get_x_days_ago_dt(self):
    return (datetime.datetime.utcnow() - datetime.timedelta(NUM_DAYS_DATA_FOR_MODELS)).strftime('%Y-%m-%dT%H:%M:') + '000000000Z'

  def preprocess_data(self, inst1_candles, inst2_candles):
    datetime_set = set.intersection(set(inst1_candles['datetime']), set(inst2_candles['datetime']))
    inst1_candles = inst1_candles.loc[inst1_candles['datetime'].isin(datetime_set)]
    inst2_candles = inst2_candles.loc[inst2_candles['datetime'].isin(datetime_set)]
    inst1_log_price = np.log(inst1_candles['close'])
    inst2_log_price = np.log(inst2_candles['close'])
    inst1_log_price = inst1_log_price.reset_index()['close']
    inst2_log_price = inst2_log_price.reset_index()['close']
    self.inst1_last_close_log = list(inst1_log_price)[-1]
    self.inst2_last_close_log = list(inst2_log_price)[-1]
    self.inst1_last_close = list(inst1_candles['close'])[-1]
    self.inst2_last_close = list(inst2_candles['close'])[-1]
    return inst1_log_price, inst2_log_price

  def train(self, data1, data2):
    y = data2
    x = data1
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    alpha = results.params.values[0]
    beta = results.params.values[1]
    spread = y - (alpha + data1 * beta)
    return alpha, beta, np.mean(spread), np.std(spread), spread

  def dikey_fuller_test(self, spread):
    result = adfuller(spread, maxlag=1)
    print('dikey-fuller test (p-value):', result[1])
    return '1' if result[1] <= 0.05 else '0'

  def is_model_expired(self, last_dt):
    x_days_ago = datetime.datetime.strptime(self.get_x_days_ago_dt()[:16], '%Y-%m-%dT%H:%M')
    last_dt = datetime.datetime.strptime(last_dt[:16], '%Y-%m-%dT%H:%M')
    return (x_days_ago - last_dt).total_seconds() >= MODEL_EXPIRE_SECONDS

  def update_model(self):
    # Update every minute
    should_update = False
    if self.model is None:
      should_update = True
    else:
      last_dt, _, _, _, _, _ = self.model
      if self.is_model_expired(last_dt):
        should_update = True

    if self.heartbeat % 100 == 0:
      print(self.heartbeat)
    self.heartbeat = (self.heartbeat + 1) % 1000000

    if should_update:
      from_dt = self.get_x_days_ago_dt()
      # Get candles
      inst1_candles = self.get_candles(INSTRUMENT_PAIR[0], from_dt)
      inst2_candles = self.get_candles(INSTRUMENT_PAIR[1], from_dt)

      # Data preprocessing
      inst1_log_price, inst2_log_price = self.preprocess_data(inst1_candles, inst2_candles)

      # Train model on preprocessed data
      alpha, beta, mean, std, spread = self.train(inst1_log_price, inst2_log_price)

      # Apply Dikey-Fuller to test for cointegration
      can_trade = self.dikey_fuller_test(spread)

      # Update model
      self.model = (from_dt, alpha, beta, mean, std, can_trade)

  def update_trade_status(self):
    r = trades.OpenTrades(ACCOUNT_ID)
    open_trades = self.client.request(r)['trades']
    inst1_open_trade = [open_trade for open_trade in open_trades if open_trade['instrument'] == INSTRUMENT_PAIR[0]]
    inst2_open_trade = [open_trade for open_trade in open_trades if open_trade['instrument'] == INSTRUMENT_PAIR[1]]
    inst1_islong = int(inst1_open_trade[0]['initialUnits']) > 0 if inst1_open_trade else None
    inst2_islong = int(inst2_open_trade[0]['initialUnits']) > 0 if inst2_open_trade else None
    # Update if the trade is current long, short, or not in trade
    self.inst1_trade_status = inst1_islong
    self.inst2_trade_status = inst2_islong

  def check_signals(self):
    _, alpha, beta, mean, std, can_trade = self.model
    spread = self.inst2_last_close_log - (alpha + self.inst1_last_close_log * beta)
    z_score = (spread - mean) / std
    if self.heartbeat % 100 == 0:
      print('z_score', z_score)
      print('Cointegrate test:', can_trade)
    if z_score >= 1.2 and can_trade:
      self.signal = SHORT_SIGNAL
    elif z_score <= -1.2 and can_trade:
      self.signal = LONG_SIGNAL
    elif z_score < 0:
      self.signal = SHORT_EXIT
    elif z_score > 0:
      self.signal = LONG_EXIT

  def long_short_orders(self, is_long_signal):
    print('Long short order: ', is_long_signal)
    r = pricing.PricingInfo(ACCOUNT_ID, {'instruments': INSTRUMENT_PAIR[0], 'include_home_conversions': True})
    inst1_prices = self.client.request(r)['prices']
    r = pricing.PricingInfo(ACCOUNT_ID, {'instruments': INSTRUMENT_PAIR[1], 'include_home_conversions': True})
    inst2_prices = self.client.request(r)['prices']
    inst1_conversion = float(inst1_prices[0]['quoteHomeConversionFactors']['positiveUnits'])
    inst2_conversion = float(inst2_prices[0]['quoteHomeConversionFactors']['positiveUnits'])

    # Calculate how much to long/short
    inst1_units = int(AMOUNT / (self.inst1_last_close * inst1_conversion))
    inst2_units = int(AMOUNT / (self.inst2_last_close * inst2_conversion))
    if not is_long_signal:
      inst2_units = -inst2_units
    else:
      inst1_units = -inst1_units

    r = orders.OrderCreate(ACCOUNT_ID, {'order': {'type': 'MARKET', 'timeInForce': 'FOK', 'instrument': INSTRUMENT_PAIR[0], 'units': inst1_units}})
    self.client.request(r)
    r = orders.OrderCreate(ACCOUNT_ID, {'order': {'type': 'MARKET', 'timeInForce': 'FOK', 'instrument': INSTRUMENT_PAIR[1], 'units': inst2_units}})
    self.client.request(r)

  def trigger_orders(self):
    if self.inst1_trade_status is None and self.inst2_trade_status is None:
      if self.signal == SHORT_SIGNAL:
        # Short inst2, long inst1
        self.long_short_orders(False)
      elif self.signal == LONG_SIGNAL:
        # Long inst2, short inst1
        self.long_short_orders(True)
        pass
    elif self.inst1_trade_status is not None and self.inst2_trade_status is not None:
      if self.signal == SHORT_EXIT and self.inst2_trade_status == False and self.inst1_trade_status == True:
        # Exit positions
        print('Short exit')
        r = positions.PositionClose(ACCOUNT_ID, instrument=INSTRUMENT_PAIR[1], data={"shortUnits": "ALL"})
        self.client.request(r)
        r = positions.PositionClose(ACCOUNT_ID, instrument=INSTRUMENT_PAIR[0], data={"longUnits": "ALL"})
        self.client.request(r)
      elif self.signal == LONG_EXIT and self.inst2_trade_status == True and self.inst1_trade_status == False:
        # Exit positions
        print('Long exit')
        r = positions.PositionClose(ACCOUNT_ID, instrument=INSTRUMENT_PAIR[1], data={"longUnits": "ALL"})
        self.client.request(r)
        r = positions.PositionClose(ACCOUNT_ID, instrument=INSTRUMENT_PAIR[0], data={"shortUnits": "ALL"})
        self.client.request(r)
    else:
      print('Trades incorrect, please check!')

  def execute(self):
    while True:
      self.update_model()
      self.update_trade_status()
      self.check_signals()
      self.trigger_orders()

  # Converts candle result from Oanda API response to dataframe
  def _to_candles_df(self, candles, with_dt=False):
        data = {key: [float(candle['mid'][key[0]]) for candle in candles] for key in ['open', 'high', 'low', 'close']}
        if with_dt:
            data['datetime'] = [candle['time'] for candle in candles]
        return pd.DataFrame(data)

pairs_trader = PairsTrader()
pairs_trader.execute()
