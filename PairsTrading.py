import config
import datetime
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import pandas as pd

class PairsTrading:
    def __init__(self):
        self.client = oandapyV20.API(access_token=config.ACCESS_TOKEN)
        # Final instruments to build model for (based on instruments in trade and selected instruments)
        self.final_instruments = set()
        self.in_trade = set()
        # Models for each instrument
        self.models = {}
        # Index for printing on console so we know that it's still running
        self.index = 0

    # Update the instruments to generate model for
    def update_final_instruments(self):
        # Reset
        self.final_instruments = set()
        # Read in the instruments that are currently in trade
        # We have to train models for those instruments
        with open(config.IN_TRADE_FILENAME) as f:
            for line in f:
                pair = line.strip().split(',')
                if len(pair) == 2:
                    self.final_instruments.add(tuple(pair))
        # Consider selected instruments to trade, but remove if they are duplicates of instruments currently in trade
        with open(config.SELECTED_INSTRUMENTS_FILENAME) as f:
            for line in f:
                pair = line.strip().split(',')
                if len(pair) == 2 and all(p not in self.final_instruments for p in pair):
                    self.final_instruments.add(tuple(pair))

    # TODO: Put this in a helper function (share it with PairsTrading.py)
    def get_candles(self, instrument, from_dt=None):
        candles_params = {'count': config.CANDLE_COUNT, 'granularity': config.GRANULARITY}
        if from_dt:
            candles_params['from'] = from_dt
        r = instruments.InstrumentsCandles(instrument, params=candles_params)
        candles = client.request(r)['candles']
        return self._to_candles_df(candles, from_dt is not None)

    # Converts candle result from Oanda API response to dataframe
    def _to_candles_df(self, candles, with_dt=False):
        df = pd.DataFrame(candles)
        if with_dt:
            df['datetime'] = [c['time'] for c in candles]
        return df[['mid.o', 'mid.h', 'mid.l', 'mid.c', 'datetime']].rename(columns={'mid.o': 'open', 'mid.h': 'high', 'mid.l': 'low', 'mid.c': 'close'})

    # Read models from the LR_MODELS_FILENAME file
    def read_models(self):
        self.models = {}
        with open(config.LR_MODELS_FILENAME) as f:
            for line in f:
                model = line.strip().split(',')
                print(model)
                self.models[model[1] + ',' + model[2]] = list(map(float, model[3:]))

    # Save the parameters from linear regression to file
    def save_models_to_file(self, pairs):
        with open(config.LR_MODELS_FILENAME, 'w') as f:
            for pair in pairs:
                f.write(','.join(pair) + '\n')

    def print_status_to_console(self):
        self.index %= 10000
        if self.index % 10 == 0:
            print(self.index)
        if self.index % 50 == 0:
            print(self.final_instruments)
        self.index += 1

    def process_pair(self, pair):
        pair_key = ','.join(pair[:2])
        if pair_key not in self.models:
            print('Model for ' + pair[0] + '/' + pair[1] + ' does not exist.')
        else:
            alpha, beta, mean, std = self.models[pair_key]
            self.analyze_pair(pair, alpha, beta, mean, std)

    def analyze_pair(self, pair, alpha, beta, mean, std):
        r = trades.OpenTrades(config.ACCOUNT_ID)
        open_trades = self.client.request(r)['trades']
        pair1_open_trade = [open_trade for open_trade in open_trades if open_trade['instrument'] == pair[0]]
        pair2_open_trade = [open_trade for open_trade in open_trades if open_trade['instrument'] == pair[1]]
        pair1_islong, pair2_islong = self.get_trade_positions(pair1_open_trade, pair2_open_trade)
        self.print_trade_positions(pair1_islong, pair2_islong)

        pair1_conversion, pair1_bid, pair1_ask, pair2_conversion, pair2_bid, pair2_ask = self.get_prices(pair)
        spread1, spread2 = self.calculate_spreads(alpha, beta, pair1_bid, pair1_ask, pair2_bid, pair2_ask)
        buy_signal, sell_signal, exit_buy_signal, exit_sell_signal = self.check_signals(spread1, spread2, mean, std)

        self.trigger_orders(pair, pair1_islong, pair2_islong, pair1_ask, pair1_bid, pair2_ask, pair2_bid, pair1_conversion, pair2_conversion, buy_signal, sell_signal, exit_buy_signal, exit_sell_signal)

    def get_trade_positions(self, pair1_open_trade, pair2_open_trade):
        pair1_islong = int(pair1_open_trade[0]['initialUnits']) > 0 if pair1_open_trade else None
        pair2_islong = int(pair2_open_trade[0]['initialUnits']) > 0 if pair2_open_trade else None
        return pair1_islong, pair2_islong

    def print_trade_positions(self, pair1_islong, pair2_islong):
        if pair1_islong and pair2_islong:
            print('both Long')
        if not pair1_islong and not pair2_islong:
            print('both short')
        if pair1_islong is None and pair2_islong is not None:
            print('only pair1 is in position')
        if pair2_islong is None and pair1_islong is not None:
            print('only pair2 is in position')

    def get_prices(self, pair):
        r = pricing.PricingInfo(config.ACCOUNT_ID, {'instruments': pair[0], 'include_home_conversions': True})
        pair1_prices = self.client.request(r)['prices']
        r = pricing.PricingInfo(config.ACCOUNT_ID, {'instruments': pair[1], 'include_home_conversions': True})
        pair2_prices = self.client.request(r)['prices']
        pair1_conversion = float(pair1_prices[0]['quoteHomeConversionFactors']['positiveUnits'])
        pair2_conversion = float(pair2_prices[0]['quoteHomeConversionFactors']['positiveUnits'])
        pair1_bid = float(pair1_prices[0]['closeoutBid'])
        pair1_ask = float(pair1_prices[0]['closeoutAsk'])
        pair2_bid = float(pair2_prices[0]['closeoutBid'])
        pair2_ask = float(pair2_prices[0]['closeoutAsk'])
        return pair1_conversion, pair1_bid, pair1_ask, pair2_conversion, pair2_bid, pair2_ask

    def calculate_spreads(self, alpha, beta, pair1_bid, pair1_ask, pair2_bid, pair2_ask):
        spread1 = np.log(pair2_bid) - np.log(pair1_ask)*beta - alpha
        spread2 = np.log(pair2_ask) - np.log(pair1_bid)*beta - alpha
        return spread1, spread2

    def check_signals(self, spread1, spread2, mean, std):
        buy_signal = spread2 < mean - config.OPEN_POS_STD*std
        sell_signal = spread1 > mean + config.OPEN_POS_STD*std
        exit_buy_signal = spread1 > mean - config.CLOSE_POS_STD*std
        exit_sell_signal = spread2 < mean + config.CLOSE_POS_STD*std
        return buy_signal, sell_signal, exit_buy_signal, exit_sell_signal

    def trigger_orders(self, pair, pair1_islong, pair2_islong, pair1_ask, pair1_bid, pair2_ask, pair2_bid, pair1_conversion, pair2_conversion, buy_signal, sell_signal, exit_buy_signal, exit_sell_signal):
        if buy_signal and pair1_islong is None:
            self.open_long_short_orders(pair, pair1_ask, pair1_bid, pair2_ask, pair2_bid, pair1_conversion, pair2_conversion)
        elif sell_signal and pair1_islong is None:
            self.open_long_short_orders(pair, pair1_ask, pair1_bid, pair2_ask, pair2_bid, pair1_conversion, pair2_conversion, long_pair1=True)
        elif exit_buy_signal and not pair1_islong:
            self.close_positions(pair)
        elif exit_sell_signal and pair1_islong:
            self.close_positions(pair)

    def open_long_short_orders(self, pair, pair1_ask, pair1_bid, pair2_ask, pair2_bid, pair1_conversion, pair2_conversion, long_pair1=False):
        amount = config.AMOUNT
        units_pair1 = int(-amount / (pair1_ask * pair1_conversion)) if not long_pair1 else int(amount / (pair1_bid * pair1_conversion))
        units_pair2 = int(amount / (pair2_bid * pair2_conversion)) if not long_pair1 else int(-amount / (pair2_ask * pair2_conversion))

        r = orders.OrderCreate(config.ACCOUNT_ID, {'order': {'type': 'MARKET', 'timeInForce': 'FOK', 'instrument': pair[0], 'units': units_pair1}})
        self.client.request(r)
        r = orders.OrderCreate(config.ACCOUNT_ID, {'order': {'type': 'MARKET', 'timeInForce': 'FOK', 'instrument': pair[1], 'units': units_pair2}})
        self.client.request(r)
        self.in_trade.add(pair)
        self.save_pairs_to_file(self.in_trade, config.IN_TRADE_FILENAME)

    def close_positions(self, pair):
        r = positions.PositionClose(config.ACCOUNT_ID, instrument=pair[0], data={"shortUnits": "ALL"})
        self.client.request(r)
        r = positions.PositionClose(config.ACCOUNT_ID, instrument=pair[1], data={"longUnits": "ALL"})
        self.client.request(r)
        if pair in self.in_trade:
            self.in_trade.remove(pair)
        self.save_pairs_to_file(self.in_trade, config.IN_TRADE_FILENAME)

    def execute(self):
        while True:
            self.update_final_instruments()
            self.print_status_to_console()
            self.read_models()
            for pair in self.final_instruments:
                self.process_pair(pair)

if __name__ == "__main__":
    PairsTrading().execute()
