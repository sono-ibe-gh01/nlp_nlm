from np import *
from layers import *
from functions import *
from model_base import BaseModel


class RNN_LM(BaseModel):
    def __init__(self, vocab_size=1000, wordvec_size=100, hidden_size=100):
        # ★理解しながらコードを記述しましょう．

    def predict(self, xs):
        # ★理解しながらコードを記述しましょう．

    def forward(self, xs, ts):
        # ★理解しながらコードを記述しましょう．

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

