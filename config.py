# Acceess token from Oanda
ACCESS_TOKEN = 'your_access_token_here'
ACCOUNT_ID = 'your_account_id_here'
# Granularity of the data to use and train the model
GRANULARITY = 'M1'
# Amount to trade (currency determined by the account's currenacy)
AMOUNT = 1000
# Number of candles to fetch in a request
CANDLE_COUNT = 5000
# Open positions when both instrument's prices deviate some std away
OPEN_POS_STD = 1.96
# Exit positions when both instrument's prices revert back to some std
CLOSE_POS_STD = 1.5
# Number of days of data to be used for training the models
NUM_DAYS_DATA_FOR_MODELS = 30
# Set how often to train the model
MODEL_EXPIRE_SECONDS = 10 * 60

# Filenames
SELECTED_INSTRUMENTS_FILENAME = 'selected_instruments.txt'
LR_MODELS_FILENAME = 'lr_models.txt'
IN_TRADE_FILENAME = 'in_trade.txt'
