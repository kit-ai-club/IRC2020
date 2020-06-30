import os
import matplotlib.pyplot as plt
import h5py
from keras.models import Sequential, Model  # kerasというライブラリのmodelsパッケージにある、Sequential, Modelという何か（ファイル or 関数 or クラス）を使うよ～
import keras.layers as layers
import numpy as np
# pycharmでは、F4 を押す#ax = fig.add_subplot(122)
# #ax.imshow(x_train_gray[0], cmap='Greys')  # gray-scaleデータなので追加の引数
# #plt.show()  # 最後はpltに戻る
#
#
# """
# データの加工
# """
# # 全結合層を使うので、画像を1次元化。画像データの行列を短冊状に切ってベクトル化。
# #x_train_gray = x_train_gray.reshape(10099, 64 * 64)  # もともと (10099, 64, 64) だったのを、(10099, 64 * 64) に。
# #x_test_gray = x_test_gray.reshape(1000, 64 * 64)
#
# # 画素を0~1の範囲に変換(正規化)。入力される値は０付近でないと、訓練が安定しない。
# #x_train_gray = x_train_gray / 255
# #x_test_gray = x_test_gray / 255
#
# x_train_rgb = x_train_rgb / 255
# x_test_rgb = x_test_rgb / 255
#
#
# """
# keras Sequential での流れ
# ①model = Sequential()  箱の作成
# ②model.add()         　層の追加
# ③model.compile()     　設定
# ④model.fit()  　　　　　訓練
# ⑤model.evaluate()　　 　評価
# """
#
# # Sequentialは、モデルを入れる箱のクラス。
# model = Sequential()
# #model.add(layers.Dense(512, activation='relu', input_dim=64 * 64))  # Dense＝全結合層。activation＝活性化関数。input_dim＝入力次元＝入力ノードの数。512＝出力ノードの数。
# model.add(layers.Dense(512, activation='relu', input_dim=64 * 64 * 3))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(101, activation='softmax'))  # 最終層のactivationは、確率を出すためにsoftmax。101クラスの分類。
#
# # ちなみに、「次元」というのは、1層あたりのノードの数＝ベクトルの要素数や、画像データのピクセル数などを表す。
# # （「画像データ＝2次元」という表現とはまた違う「次元」の話）
#
#
# # 各種設定
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy
#
#
# # 訓練の実行
# epochs = 100
# batch_size = 512
# #history = model.fit(x=x_train_gray, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
# history = model.fit(x=x_train_rgb, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
# # historyに訓練の推移のデータが格納される
#
#
# # 評価
# #score = model.evaluate(x_test_gray, y_test)
# score = model.evaluate(x_test_rgb, y_test)
# print('test_loss:', score[0])
# print('test_acc:', score[1])
#
#
# # 訓練の推移
# fig = plt.figure(figsize=(10, 5))
#
# ax = fig.add_subplot(121)
# ax.plot(range(epochs), history.history['acc'], label='training')  # x軸、y軸、ラベル
# ax.plot(range(epochs), history.history['val_acc'], label='validation')
# ax.set_title('acc')
# ax.legend()  # 凡例を表示する
#
# ax = fig.add_subplot(122)
# ax.plot(range(epochs), history.history['loss'], label='training')
# ax.plot(range(epochs), history.history['val_loss'], label='validation')
# ax.set_title('loss')
# ax.legend()  # 凡例を表示する
#
# plt.show()
#
#
# """
# keras Functinal API での流れ
# ①Input()            入力の作成
# ②x = f(x)         　層の追加・出力の作成
# ③model = Model()    箱の作成
# ④model.compile()     設定
# ⑤model.fit()  　　　  訓練
# ⑥model.evaluate()　　 評価
# """
#
# #inputs = layers.Input(shape=(64*64,))
# inputs = layers.Input(shape=(64*64*3,))
# x = layers.Dense(512, activation='relu')(inputs)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(101, activation='softmax')(x)
#
# model = Model(inputs=inputs, outputs=x)
#
# # 以降は同じ
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy
#
# #history = model.fit(x=x_train_gray, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
# history = model.fit(x=x_train_rgb, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
#
# #score = model.evaluate(x_test_gray, y_test)
# score = model.evaluate(x_test_rgb, y_test)
# print('test_loss:', score[0])
# print('test_acc:', score[1])
#
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(121)
# ax.plot(range(epochs), history.history['acc'], label='training')
# ax.plot(range(epochs), history.history['val_acc'], label='validation')
# ax.set_title('acc')
# ax.legend()
# ax = fig.add_subplot(122)
# ax.plot(range(epochs), history.history['loss'], label='training')
# ax.plot(range(epochs), history.history['val_loss'], label='validation')
# ax.set_title('loss')
# ax.legend()
# plt.show()
#
#
# # いろんなハイパラを全てハードコーディングしていますが、
# # 通常はプログラムの最初にまとめたり、ハイパラだけをまとめたファイルから読み込んだりします。と、引用元・定義元に飛べるので使ってみよう


"""
データの取得
"""
# data-fileの場所（kaggleからDLする）
data_path = os.path.join("drive", "My Drive")
train_h5_path = os.path.join(data_path, "food_c101_n10099_r64x64x3.h5")  # train-data   pathの文字列を確認してみよう
test_h5_path = os.path.join(data_path, "food_test_c101_n1000_r64x64x3.h5")  # test-data
# windowsとlinuxで、スラッシュとバックスラッシュの違いがあることを気にしないために、 os.path.join を使う


# train-data
with h5py.File(train_h5_path, 'r') as file:  # train-dataを 'r' = read mode で読み込んで、変数fileに格納
    print(file.keys())  # .h5 形式では、関数keys()で中身が見れる
    x_train = file['images'].value  # imagesにアクセス。さらに、.value と書くとnumpy形式にしてくれる
    y_train = file['category'].value  # 教師データ（正解データ）。すでにone-hot vector になっている
    category_names = file['category_names'].value  # str型ではなく、bytes型で格納されている。見てみよう
    label_names = [x.decode() for x in file['category_names'].value]  # 全カテゴリ名を格納。bytes型は、decode()でstrに変換できる。
    print(label_names)
# with構文で開いたファイルは、構文が終われば自動的に閉じられる。

# x_train = (10099, 64, 64, 3), y_train = (10099, 101)
#  ↑ このようにshapeを調べてメモしておくと便利


# test-data
with h5py.File(test_h5_path, 'r') as file:
    print(file.keys())
    x_test = file['images'].value
    y_test = file['category'].value

x_train_rgb = np.reshape(x_train, (-1, 64 * 64 * 3))
x_test_rgb = np.reshape(x_test, (-1, 64 * 64 * 3))


"""
【matplotlib.pyplotの基本】
※pyplotはpltとしてimportしておく
①figure（ウィンドウ）を作成する
②figure上にaxes（グラフ領域）を１つ以上作成する
③axes上にグラフを作成する（axes.plot()なら折れ線グラフ、など）
④show
"""
# 画像チェック
#fig = plt.figure(figsize=(10, 5))  # figure-sizeはインチ単位
#ax = fig.add_subplot(121)  # Figure内にAxesを追加。121 =「1行2列のaxesを作って、その1番目(1列目)をreturnしろ」
#ax.imshow(x_train[0])  # 画像ならimshow()
#ax = fig.add_subplot(122)
#ax.imshow(x_train_gray[0], cmap='Greys')  # gray-scaleデータなので追加の引数
#plt.show()  # 最後はpltに戻る


"""
データの加工
"""
# 画素を0~1の範囲に変換(正規化)。入力される値は０付近でないと、訓練が安定しない。
x_train_rgb = x_train_rgb / 255
x_test_rgb = x_test_rgb / 255


"""
keras Sequential での流れ
①model = Sequential()  箱の作成
②model.add()         　層の追加
③model.compile()     　設定
④model.fit()  　　　　　訓練
⑤model.evaluate()　　 　評価
"""

# Sequentialは、モデルを入れる箱のクラス。
model = Sequential()
# Dense＝全結合層。activation＝活性化関数。input_dim＝入力次元＝入力ノードの数。512＝出力ノードの数。
model.add(layers.Dense(512, activation='relu', input_dim=64 * 64 * 3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(101, activation='softmax'))  # 最終層のactivationは、確率を出すためにsoftmax。101クラスの分類。

# ちなみに、「次元」というのは、1層あたりのノードの数＝ベクトルの要素数や、画像データのピクセル数などを表す。
# （「画像データ＝2次元」という表現とはまた違う「次元」の話）


# 各種設定
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy


# 訓練の実行
epochs = 100
batch_size = 512
history = model.fit(x=x_train_rgb, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
# historyに訓練の推移のデータが格納される


# 評価
score = model.evaluate(x_test_rgb, y_test)
print('test_loss:', score[0])
print('test_acc:', score[1])


# 訓練の推移
fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(121)
ax.plot(range(epochs), history.history['acc'], label='training')  # x軸、y軸、ラベル
ax.plot(range(epochs), history.history['val_acc'], label='validation')
ax.set_title('acc')
ax.legend()  # 凡例を表示する

ax = fig.add_subplot(122)
ax.plot(range(epochs), history.history['loss'], label='training')
ax.plot(range(epochs), history.history['val_loss'], label='validation')
ax.set_title('loss')
ax.legend()  # 凡例を表示する

plt.show()


"""
keras Functinal API での流れ
①Input()            入力の作成
②x = f(x)         　層の追加・出力の作成
③model = Model()    箱の作成
④model.compile()     設定
⑤model.fit()  　　　  訓練
⑥model.evaluate()　　 評価
"""

inputs = layers.Input(shape=(64*64*3,))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(101, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

# 以降は同じ
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy

history = model.fit(x=x_train_rgb, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score = model.evaluate(x_test_rgb, y_test)
print('test_loss:', score[0])
print('test_acc:', score[1])

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.plot(range(epochs), history.history['acc'], label='training')
ax.plot(range(epochs), history.history['val_acc'], label='validation')
ax.set_title('acc')
ax.legend()
ax = fig.add_subplot(122)
ax.plot(range(epochs), history.history['loss'], label='training')
ax.plot(range(epochs), history.history['val_loss'], label='validation')
ax.set_title('loss')
ax.legend()
plt.show()


# いろんなハイパラを全てハードコーディングしていますが、
# 通常はプログラムの最初にまとめたり、ハイパラだけをまとめたファイルから読み込んだりします。
