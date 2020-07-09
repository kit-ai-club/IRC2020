"""
【import構文とライブラリ】
ライブラリ：パッケージをまとめたものであって、Anaconda とかを使ってインストールできるようにしたもの。numpy、keras など。
パッケージ：.py（プログラムファイル）をまとめたディレクトリ（フォルダ）であって、 __init__.py  を放り込んだもの。
　パッケージの中にあるものは、importできる。
　自分のプロジェクト内のディレクトリにも、__init__.py というファイルがあれば、それをパッケージと認識させることができ、中のモジュール等をimportできる。
"""
import os
import matplotlib.pyplot as plt
import h5py
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Input, \
    Concatenate
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.losses import categorical_crossentropy
from keras.layers import GlobalAveragePooling2D
from keras import regularizers
import keras.layers as layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
import gc

# pycharmでは、F4 を押すと、引用元・定義元に飛べるので使ってみよう


"""
data path
"""
data_path = os.path.join("data")  # colabの場合、google-driveのルートが、"drive/My Drive" となる
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

# example
ex_path = os.path.join(train_path, "train_256_0.h5")
with h5py.File(ex_path, 'r') as file:  # train-dataを 'r' = read mode で読み込んで、変数fileに格納
    x_train = file['images'].value  # imagesにアクセス。さらに、.value と書くとnumpy形式にしてくれる
    y_train = file['category'].value  # 教師データ（正解データ）。すでにone-hot vector になっている

"""
【matplotlib.pyplotの基本】
※pyplotはpltとしてimportしておく
①figure（ウィンドウ）を作成する
②figure上にaxes（グラフ領域）を１つ以上作成する
③axes上にグラフを作成する（axes.plot()なら折れ線グラフ、など）
④show
"""
# 画像チェック
fig = plt.figure(figsize=(3, 3))  # figure-sizeはインチ単位
ax = fig.add_subplot(111)  # Figure内にAxesを追加。121 =「1行2列のaxesを作って、その1番目(1列目)をreturnしろ」
ax.imshow(x_train[0])  # 画像ならimshow()
plt.show()  # 最後はpltに戻る

"""
keras Functinal API での流れ
①Input()            入力の作成
②x = f(x)         　層の追加・出力の作成
③model = Model()    箱の作成
④model.compile()     設定
⑤model.fit()  　　　  訓練
⑥model.evaluate()　　 評価
"""

inputs = Input(shape=x_train.shape[1:])
f = 64
ki = 'he_normal'
kr = regularizers.l2(1e-11)
x = Conv2D(filters=f, kernel_size=7, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(inputs)
x = MaxPooling2D(pool_size=2)(x)
n = 5 #回数の設定（ここは実験で変更したい）
for i in range(n):
    shortcut = x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Conv2D(filters=f*(2**i), kernel_size=1, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f*(2**i), kernel_size=3, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f*(2**(i+2)), kernel_size=1, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = Concatenate()([x, shortcut])
    if i != (n - 1):
        x = MaxPooling2D(pool_size=2)(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(rate=0.4)(x)
x = Dense(units=101, kernel_initializer=ki, kernel_regularizer=kr)(x)
x = BatchNormalization()(x)
x = Activation('softmax')(x)
x = Dropout(rate=0.4)(x)

model = Model(inputs=inputs, outputs=x)

# 各種設定
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy

# メモリリーク対策のグローバル変数
x, y, datagen, generator = None, None, None, None  # いじらない


def flow_from_h5(directory, batch_size, data_aug=False):
    """
    directory内のh5ファイルを順に読み込んで出力するジェネレータ。
    ImageDataGeneratorの部分以外はいじらない。
    """

    files = glob.glob(os.path.join(directory, '*.h5'))
    while True:
        for file in files:
            global x, y, datagen, generator
            del x, y, datagen, generator
            gc.collect()

            with h5py.File(file, 'r') as f:
                x = f['images'].value
                y = f['category'].value

            if not data_aug:
                datagen = ImageDataGenerator(rescale=1 / 255.)  # いじらない。rescaleで画像を正規化している。
            else:
                datagen = ImageDataGenerator(rescale=1 / 255.)  # DataAugmentationするなら、引数（rescale以外）をいじる。

            generator = datagen.flow(
                x,
                y,
                batch_size=batch_size,
                shuffle=True, )
            epoch_per_file = x.shape[0] // batch_size + 1
            for e in range(epoch_per_file):
                yield next(generator)


"""
画像を256*256にしてから激重なので、pycharm上でデバッグしたいときは、
以下4つの値を全て小さめに設定すれば、動きが確認できる程度に軽くなるはず（colabで訓練するときは値を戻すことに注意）
"""
epochs = 100
batch_size = 100
steps_per_epoch = 75750 // batch_size + 1  # 元は 75750 // batch_size + 1
validation_steps = 25250 // batch_size + 1  # 元は  25250 // batch_size + 1

train_generator = flow_from_h5(train_path, batch_size, data_aug=True)
test_generator = flow_from_h5(test_path, batch_size, data_aug=False)

"""
以降は、原則いじらない
"""
# 訓練
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps,
    workers=0)

# 評価
score = model.evaluate_generator(
    test_generator,
    steps=validation_steps,
    workers=0)
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

# いろんなハイパラを全てハードコーディングしていますが、
# 通常はプログラムの最初にまとめたり、ハイパラだけをまとめたファイルから読み込んだりします。


# labels = []
# with open(os.path.join(data_path, "labels.txt")) as file:
#     line = file.readline()[:-1]
#     labels.append(line)


# train_path = os.path.join(data_path, "train")  # train-data
# test_path = os.path.join(data_path, "test")  # test-data
# meta_path = os.path.join(data_path, "meta")  # test-data

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
#
# meta = unpickle(meta_path)
# train = unpickle(train_path)
# test = unpickle(test_path)
