# %%
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
from tvDatafeed import TvDatafeed,Interval
import numpy as np

# %%
tv = TvDatafeed()
niftydf = tv.get_hist(symbol='NIFTY_50', exchange='NSE', interval=Interval.in_daily,n_bars=1000)

# %%
x = niftydf.index.values.astype('datetime64[s]')
x = [[int(i.item().year), int(i.item().day), int(i.item().month), int(i.item().hour), int(i.item().minute), int(i.item().second)] for i in x]
x = np.array(x)
y = niftydf.iloc[:,-2].values

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=120)

# %%
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

# %%
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=120)

# %%
model.fit(x_train, y_train)

# %%
from sklearn.metrics import r2_score
r2_score(y_test, model.predict(x_test))

# %%
y_pred = model.predict(x_test)

# %%
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %%
val = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
perct = np.apply_along_axis(lambda j: (j[0] - j[1]) * 100 / j[0], 1, val)

# %%
x_label = np.apply_along_axis(lambda j: str(j[0]) + '-' + str(j[1]) + '-' + str(j[2]) + ' ' + str(j[3]) + ':' + str(j[4]) + ':' + str(j[5]), 1, x_test)
x_label

# %%
from matplotlib import pyplot as plt
fig = plt.figure(figsize = (50, 25))
plt.bar(x_label,perct, color ='maroon',
        width = 0.4)
plt.xlabel("Stocks")
plt.ylabel("% Error")
plt.title("Kya pata")
plt.show()

# %%
# apple = tv.get_hist(symbol='AAPL', exchange='NASDAQ', interval=Interval.in_daily,n_bars=1000)

# %%
# x_2 = apple.index.values.astype('datetime64[s]')
# x_2 = [[int(i.item().year), int(i.item().day), int(i.item().month), int(i.item().hour), int(i.item().minute), int(i.item().second)] for i in x_2]
# x_2 = np.array(x_2)
# y_2 = apple.iloc[:,-2].values

# %%
# from sklearn.model_selection import train_test_split
# x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, test_size=0.2, random_state=120)

# %%
# len(y_train)

# %%
# scaler.fit(np.concatenate((y_train, y_train_2), axis=0).reshape(-1, 1))
# y_train_2 = scaler.fit_transform(y_train_2.reshape(-1, 1)).reshape(-1)
# y_test_2 = scaler.transform(y_test_2.reshape(-1, 1)).reshape(-1)

# %%
# len(np.concatenate((y_train, y_train_2), axis=0))

# %%
# model.fit(np.concatenate((x_train, x_train_2), axis=0), np.concatenate((y_train, y_train_2), axis=0))

# %%
# r2_score(y_test_2, model.predict(x_test_2))

# %%
# cartrade = tv.get_hist(symbol='CARTRADE', exchange='NSE', interval=Interval.in_daily,n_bars=1000)


# %%
# x_3 = apple.index.values.astype('datetime64[s]')
# x_3 = [[int(i.item().year), int(i.item().day), int(i.item().month), int(i.item().hour), int(i.item().minute), int(i.item().second)] for i in x_3]
# x_3 = np.array(x_3)
# y_3 = apple.iloc[:,-2].values
# from sklearn.model_selection import train_test_split
# x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_3, y_3, test_size=0.2, random_state=120)
# len(y_train)
# scaler.fit(np.concatenate((y_train, y_train_2, y_train_3), axis=0).reshape(-1, 1))
# y_train_3 = scaler.fit_transform(y_train_3.reshape(-1, 1)).reshape(-1)
# y_test_3 = scaler.transform(y_test_3.reshape(-1, 1)).reshape(-1)
# # len(np.concatenate((y_train, y_train_2), axis=0))

# %%
# # np.concatenate((y_train, y_train_2, y_train_3), axis=0)
# np.concatenate((x_train, x_train_2, x_train_3), axis=0)

# %%
# model.fit(np.concatenate((x_train, x_train_2, x_train_3), axis=0), np.concatenate((y_train, y_train_2, y_train_2), axis=0))
# r2_score(y_test_3, model.predict(x_test_3))

# %%
# cartrade = tv.get_hist(symbol='CARTRADE', exchange='NSE', interval=Interval.in_daily,n_bars=1000)



