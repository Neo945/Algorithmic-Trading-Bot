from matplotlib import pyplot as plt
import mplfinance as mpf
import pandas as pd
from tvDatafeed import TvDatafeed,Interval
import numpy as np
from PIL import Image
import base64
import io


def sma(stock, exchange, bar=200):
    tv = TvDatafeed()
    try:
        df = tv.get_hist(symbol=stock, exchange=exchange, interval=Interval.in_daily,n_bars=bar)
    except:
        return "Error", False
    df['SMA20']=df['close'].rolling(window=20).mean()
    df['SMA50']=df['close'].rolling(window=50).mean()
    buy=np.where((df['SMA20'].shift(-1)>df['SMA50'].shift(-1)),1,np.nan)*df['low']
    sell=np.where((df['SMA20'].shift(-1)<df['SMA50'].shift(-1)),1,np.nan)*df['low']
    apd = [mpf.make_addplot(buy, scatter=True, markersize=50, marker=r'$\Uparrow$', color='green'),mpf.make_addplot(sell, scatter=True, markersize=50, marker=r'$\Downarrow$', color='red'),mpf.make_addplot(df['SMA20']), mpf.make_addplot(df['SMA50'])]
    fig, _  = mpf.plot(df, type='candle', addplot=apd, returnfig=True, style="binance")
    fig.legend([None,None,'Buy','Sell','SMA20','SMA50'])
    img = fig2img(fig)
    data = io.BytesIO()
    img.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue()).decode('utf-8')
    return encoded_img_data, True


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def bb(stock, exchange, bar=200):
    tv=TvDatafeed()
    try:
        df = tv.get_hist(symbol=stock, exchange=exchange, interval=Interval.in_daily,n_bars=bar)
    except:
        return "Error", False
    df['sma_20days']=df['close'].rolling(window=20).mean()
    df['upper_bollinger']=df['sma_20days']+df['close'].rolling(window=20).std()
    df['lower_bollinger']=df['sma_20days']-df['close'].rolling(window=20).std()
    bollinger=df[['sma_20days','upper_bollinger','lower_bollinger']]
    buy=np.where(df['close'].shift(1)<df['lower_bollinger'].shift(1),1,np.nan)*df['close']
    sell=np.where(df['close'].shift(1)>df['upper_bollinger'].shift(1),1,np.nan)*df['close']
    apd=[mpf.make_addplot(bollinger),mpf.make_addplot(buy, scatter=True, markersize=50, marker=r'$\Uparrow$', color='green'),mpf.make_addplot(sell, scatter=True, markersize=50, marker=r'$\Downarrow$', color='red')]
    fig, _ = mpf.plot(df, type='candle', addplot=apd, returnfig=True, style="binance")
    fig.legend([None,None, "SMA20","UBB","LBB", "Buy", "Sell"])
    img = fig2img(fig)
    data = io.BytesIO()
    img.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue()).decode('utf-8')
    return encoded_img_data, True

def random_forest(stock, exchange, bar=1000):
    tv = TvDatafeed()
    try:
        df = tv.get_hist(symbol=stock, exchange=exchange, interval=Interval.in_daily,n_bars=bar)
    except:
        print("ERROR")
        return "Error", False
    x = df.index.values.astype('datetime64[s]')
    x = [[int(i.item().year), int(i.item().day), int(i.item().month), int(i.item().hour), int(i.item().minute), int(i.item().second)] for i in x]
    x = np.array(x)
    y = df.iloc[:,-2].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=120)
    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=120)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    val = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
    perct = np.apply_along_axis(lambda j: (j[0] - j[1]) * 100 / j[0], 1, val)
    x_label = np.apply_along_axis(lambda j: str(j[0]) + '-' + str(j[1]) + '-' + str(j[2]) + ' ' + str(j[3]) + ':' + str(j[4]) + ':' + str(j[5]), 1, x_test)
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize = (20, 10))
    plt.bar(x_label, perct, color ='maroon',
            width = 0.4)
    plt.xticks(rotation = 90, fontsize = 6)
    plt.xlabel("Stocks")
    plt.legend(['% Error'])
    plt.ylabel("% Error")
    plt.title("Error graph")
    fig = plt.gcf()
    img = fig2img(fig)
    data = io.BytesIO()
    img.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue()).decode('utf-8')
    return encoded_img_data, True