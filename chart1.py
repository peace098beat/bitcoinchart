# ! coding:utf-8
"""
chart1.py

UNIXTIMESTAMP : http://url-c.com/tc/

Created by 0160929 on 2017/02/09 15:26
"""
from datetime import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd









# ロードする取引の時間

now = datetime(2011, 10, 31, 00, 0, 0, 0)
# now = datetime(2015,2,9,15,0,0,0)
UNIXTIMESTAMP = int(time.mktime(now.timetuple()))
dtime = datetime.fromtimestamp(UNIXTIMESTAMP)
print(now, UNIXTIMESTAMP, dtime.strftime("%Y%m%d%H%M%S"))


def load_btcoin_numpy(reload=False, save=True):
    if reload:
        # CSV= "http://api.bitcoincharts.com/v1/trades.csv?symbol=coincheckJPY&start=" + str(UNIXTIMESTAMP)
        CSV = ".coincheckJPY2.csv"

        print("Server Requeset : " + CSV)

        df = pd.read_csv(CSV,
                         header=None,
                         parse_dates=True,
                         date_parser=lambda x: datetime.fromtimestamp(float(x)),
                         index_col='datetime',
                         names=['datetime', 'price', 'amount'])

        # 時間軸をunixtimeに YYYYMMDDHHMMSS
        # timecode = np.array([d.strftime("%Y%m%d%H%M%S") for d in df.index])
        timecode = df.index

        # 価格と取引量をndarrayに
        mat = df.as_matrix()
        price = mat[:, 0]
        amount = mat[:, 1]

        print("length : {}".format(price.size))
        print(timecode)

        # 保存
        if save:
            print("save .npy start")
            np.save("timecode", timecode)
            np.save("price", price)
            np.save("amount", amount)
            print("save .npy end")

    else:
        print("Load .npy start")
        # 呼び出し
        timecode = np.load("timecode.npy")
        price = np.load("price.npy")
        amount = np.load("amount.npy")

    print("Data Load End")
    return timecode, price, amount


def btc_fft(price, timestamp, nfft=512):
    N = price.size

    _total_frame = int(N / nfft)

    b = N - (_total_frame * nfft)

    total_frame = _total_frame + 1

    print("Total Sample : {}".format(price.size))
    print("Total Frame : {}, {}".format(total_frame, total_frame * nfft))

    n_zeropaddin = (nfft - b)

    N_New = N + n_zeropaddin

    print("zero padding  : {}".format(n_zeropaddin))

    assert N_New == (total_frame) * nfft

    _price = np.zeros(N_New)
    _timestamp = np.zeros(N_New)

    _price[:N] = price
    _timestamp[:N] = timestamp

    price = _price.copy()
    timestamp = _timestamp.copy()

    assert price.size == N_New, (price.size, N_New)
    assert timestamp.size == N_New

    X = np.zeros((total_frame, int(nfft / 2)))

    for i in range(total_frame):
        n = i * nfft

        frame = price[n:n + nfft]
        assert frame.size == nfft, (i, frame.size, nfft)

        x = np.fft.fft(frame, nfft)

        X[i, :] = abs(x[:int(nfft / 2)])
    return X


# =================================
#  sox な感じにスペクトログラム表示
# =================================
def imshow_sox(ax, spectrogram, rm_low=0.1):
    max_value = spectrogram.max()
    ### amp to dbFS
    db_spec = 20 * np.log10(spectrogram / float(max_value))
    ### カラーマップの上限と下限を計算
    vmin = -100
    vmax = 0

    print("db_spec ", db_spec.min(), db_spec.max())

    ax.imshow(db_spec.T, origin="lower", aspect="auto", cmap="hot", vmax=vmax, vmin=vmin)


from matplotlib.widgets import Slider, Button, RadioButtons


def main():
    timecode, price, amount = load_btcoin_numpy(False, False)

    print("Data Length : " + str(price.size))

    X = btc_fft(price, timecode, 1024*4)

    fig = plt.figure()
    # 2行1列の図を描く場所を確保
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)


    plt.subplots_adjust(left=0.25, bottom=0.25)

    ax1.plot(timecode, price)
    ax1.set_xticklabels(timecode, rotation=90, size="small")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # ax2.plot(20*np.log10(X[5,:]))
    X = X / X.max()
    imshow_sox(ax2, X)

    Pow = 20*np.log10(X)

    l, =ax3.plot(Pow[1,:])
    ax3.set_ylim([-120,0])

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03] )
    sfreq = Slider(axfreq, "Freq", 0, X.shape[0]-1, valinit=3)

    def update(val):
        l.set_ydata(Pow[int(val),:])
        fig.canvas.draw_idle()
    sfreq.on_changed(update)

    plt.savefig("chart1.png")
    plt.show()


if __name__ == '__main__':
    main()
