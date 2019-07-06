import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートする為のパス
from dataset.mnist import load_mnist

# 最初の呼び出しは数分かかる
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)