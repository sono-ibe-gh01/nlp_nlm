from np import *
from functions import softmax
from model_rnn_lm import RNN_LM
import dataset
from util import get_vocab_size


class Generator(RNN_LM):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        # ★理解しながらコードを記述しましょう．


# main（文章生成）

# ★理解しながらコードを記述しましょう．
