"""
まず、conda install keras
"""

# 脳死で覚えるkeras入門
# https://qiita.com/wataoka/items/5c6766d3e1c674d61425
"""
ポイント
①データを作成
②model = Sequential()  箱の作成
③model.add()         　層の追加
④model.compile()     　設定
⑤model.fit()  　　　　　訓練
⑥model.evaluate()　　 　評価
"""

from keras.datasets import mnist

# keras.datasetsっていうところから、mnistという何か（ファイル or クラス or 関数）を借りるよ～
# keras フォルダの中の datasets ＝ フォルダ or ファイル
# F4 で調べてみよう

# datasets はフォルダ、mnist はファイル


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist.py の load_data() 関数
# 変数命名の定番：、x = 入力データinput、y or t = 教師データ
# train とあるのは訓練データ、test とあるのはテストデータ。
# y_train, y_test には、x_train, x_test の画像に書かれている数字の正解が格納されている。
# インデックスが対応しているかどうか、x_test[0], y_test[0] で確認しよう


from keras.utils.np_utils import to_categorical
# 普通は、import 文は全部まとめてプログラムの頭に書く。今回は分かりやすさ優先。


# 画像を1次元化
# もともと (60000, 28, 28) だったのを、(60000, 784) に。画像データの行列を短冊状に切ってベクトル化。
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 画素を0~1の範囲に変換(正規化)
# DeepLearning は結構シビア
# たとえば、入力される値は標準正規分布（平均0、分散1）に近い分布でないと、訓練が安定しない。
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# もともと画像データは、0-255 整数intデータ。それを 0-1 に収めたいので、float に変換。
# x_train / 255.  でも同じ効果のはず


# 正解ラベルをone-hot-encoding
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)
# one-hot-vector とは、[0 0 0 1 0 0 0 0 0 0] みたいなやつ。
# y_test[0], y_test_onehot[0] を見よう

# 入力サイズが 784, 出力サイズが 10 で、出力層の値は 0-1 になるような neural network を作ればいい！
# 10 個の出力ノードは、それぞれ 0-9 の数字に対応。
# 出力 0-1 の値は、そのノードが正解である確率を表す。だから、出力層の 10 ノードの合計 ＝ 1 にすべき。・・・ softmax 関数


from keras.models import Sequential
from keras.layers import Dense

# neural network model を入れる箱のクラス。箱だけでは意味ないので、model.add で層layerを追加していく
model = Sequential()
# Dense ＝ 全結合層 ＝ よく見る普通の neural net layer。ほかに、畳み込み層（Conv2D）やLSTM層なども。
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))
# 64, 10 = 出力サイズ ＝ 出力ノードの数
# 活性化関数activation は、中間層なら relu でほぼ確定。出力層なら softmax か sigmoid って感じ、これは解く問題によって決まる。


# 各種設定
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 最適化手法optimizer は、とりあえず adam か rmsprop あたりでよいのでは。とりあえず生みたいなノリで決める。
# 損失関数loss は、問題によって決まる。
# 評価関数metrics は、accuracy にしておけばよい。


# 訓練の実行
model.fit(x_train, y_train_onehot,
          batch_size=100,
          epochs=12,
          verbose=1)
# 60000 枚を一気に訓練するのは不可能。ランダムにサンプリングして使う。訓練1ループあたりにいくつのデータを使うか： batch_size
# 訓練のループ回数：epoch
# verbose は 1 にしとけばよい


# 評価
score = model.evaluate(x_test, y_test_onehot)
print(score[0])
print(score[1])
