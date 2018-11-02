from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import requests

BASE_URL = 'https://api.iextrading.com/1.0'
ticker = 'AAPL'

resp = requests.get(
            f'{BASE_URL}/stock/{ticker}/chart/6m',
       )

print(resp.json())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

K.clear_session()

# model = Sequential()
