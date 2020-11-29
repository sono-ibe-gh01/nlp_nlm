from functions import *
from np import *


class RNN:
    def __init__(self, Wx, Wh, b):
        # ★理解しながらコードを記述しましょう．

    def forward(self, x, h_prev):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx    # ...: 要素や範囲の省略形
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

# Tステップ分の処理をまとめて行うレイヤ
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        # ★理解しながらコードを記述しましょう．

    def forward(self, xs):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)  # 勾配を合算
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class TimeEmbedding:
    def __init__(self, W):
        # ★理解しながらコードを記述しましょう．

    def forward(self, xs):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        # ★理解しながらコードを記述しましょう．

    def forward(self, x):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        # ★理解しながらコードを記述しましょう．

    def forward(self, xs, ts):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]   # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx


class Embedding:
    def __init__(self, W):
        # ★理解しながらコードを記述しましょう．

    def forward(self, idx):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None





