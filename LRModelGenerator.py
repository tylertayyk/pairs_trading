import config
import os
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm

class LRModelGenerator:
    def __init__(self):
        self.client = oandapyV20.API(access_token=config.ACCESS_TOKEN)
        self.instrument_pairs_to_trade = set()
        self.in_trade = set()

    def is_model_expired(self, last_dt):
        x_days_ago = datetime.datetime.utcnow() - datetime.timedelta(config.NUM_DAYS_DATA_FOR_MODELS)
        last_dt = datetime.datetime.strptime(last_dt[:16], '%Y-%m-%dT%H:%M')
        return (x_days_ago - last_dt).total_seconds() >= config.MODEL_EXPIRE_SECONDS

    def read_models(self):
    	# Generate file if it doesn't exist (first run)
        if not os.path.isfile(config.LR_MODELS_FILENAME):
            with open(config.LR_MODELS_FILENAME, 'w') as f:
        	    # Creates empty file
                f.write('')
        self.models = {}
        with open(config.LR_MODELS_FILENAME) as f:
            for line in f:
                model = line.strip().split(',')
                if model and not self.is_model_expired(model[0]):
                    self.models[model[1]+','+model[2]] = model[0:6]

    def write_models_to_file(self, models):
        with open(config.LR_MODELS_FILENAME, 'w') as f:
            for pair in models:
                model = models[pair]
                p = pair.split(',')
                f.write(model[0] + ',' + p[0] + ',' + p[1] + ',' + model[1] + ',' + model[2] + ',' + model[3] + ',' + model[4] + '\n')

    def update_trades(self):
    	# Generate file if it doesn't exist (first run)
        if not os.path.isfile(config.IN_TRADE_FILENAME):
            with open(config.IN_TRADE_FILENAME, 'w') as f:
            	# Creates empty file
                f.write('')
        pair_set = set()
        with open(config.IN_TRADE_FILENAME) as f:
            for line in f:
                pair = line.strip().split(',')
                if len(pair) == 2:
                    self.instrument_pairs_to_trade.add(tuple(pair))
                    self.in_trade.add(tuple(pair))
                    pair_set.update(pair)
        with open(config.SELECTED_INSTRUMENTS_FILENAME) as f:
            for line in f:
                pair = line.strip().split(',')
                if len(pair) == 2 and pair[0] not in pair_set and pair[1] not in pair_set:
                    self.instrument_pairs_to_trade.add(tuple(pair))

    def to_candles_df(self, candles, with_dt=False):
        data = {key: [float(candle['mid'][key[0]]) for candle in candles] for key in ['open', 'high', 'low', 'close']}
        if with_dt:
            data['datetime'] = [candle['time'] for candle in candles]
        return pd.DataFrame(data)

    def get_candles(self, instrument, from_dt=None):
        params = {'count': config.CANDLE_COUNT, 'granularity': config.GRANULARITY, 'from': from_dt} if from_dt else {'count': config.CANDLE_COUNT, 'granularity': config.GRANULARITY}
        r = instruments.InstrumentsCandles(instrument, params=params)
        candles = self.client.request(r)['candles']
        return self.to_candles_df(candles, from_dt is not None)

    def get_x_days_ago_dt(self):
        return (datetime.datetime.utcnow() - datetime.timedelta(config.NUM_DAYS_DATA_FOR_MODELS)).strftime('%Y-%m-%dT%H:%M:') + '000000000Z'

    def update_model(self, pair):
        from_dt = self.get_x_days_ago_dt()
        pair1_candles = self.get_candles(pair[0], from_dt)
        pair2_candles = self.get_candles(pair[1], from_dt)
        datetime_set = set.intersection(set(pair1_candles['datetime']), set(pair2_candles['datetime']))
        pair1_candles = pair1_candles.loc[pair1_candles['datetime'].isin(datetime_set)]
        pair2_candles = pair2_candles.loc[pair2_candles['datetime'].isin(datetime_set)]
        pair1_log_price = np.log(pair1_candles['close'])
        pair2_log_price = np.log(pair2_candles['close'])

        pair1_log_price = pair1_log_price.reset_index()['close']
        pair2_log_price = pair2_log_price.reset_index()['close']

        # Run linear regression over two log price series
        x = list(pair1_log_price)
        x_const = sm.add_constant(x)
        y = list(pair2_log_price)
        linear_reg = sm.OLS(y, x_const)
        results = linear_reg.fit()

        alpha = results.params[0]
        beta = results.params[1]

        spreads = pair2_log_price - pair1_log_price*beta-alpha
        mean = spreads.mean()
        std = spreads.std()
        return from_dt, str(alpha), str(beta), str(mean), str(std)

    def save_pairs_to_file(self, pairs, filename):
        with open(filename, 'w') as f:
            for pair in pairs:
                f.write(','.join(pair) + '\n')

    def generate_models(self):
    	while True:
            self.update_trades()
            self.read_models()

            updated = False
            for pair in self.instrument_pairs_to_trade:
                pair_key = ','.join(pair[:2])
                if pair_key not in self.models:
                    updated = True
                    from_dt, alpha, beta, mean, std = self.update_model(pair)
                    self.models[pair_key] = [from_dt, alpha, beta, mean, std]
            if updated:
                print('Updated models')
                self.write_models_to_file(self.models)

if __name__ == "__main__":
    LRModelGenerator().generate_models()