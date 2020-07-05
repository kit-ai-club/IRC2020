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
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Reshape, Activation, Concatenate, Lambda, Dropout, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, GlobalAveragePooling2D
from keras.models import Model
import keras.layers as layers

# pycharmでは、F4 を押すと、引用元・定義元に飛べるので使ってみよう

"""
ここの数値を変更しよう！
*カーネルサイズはフィルタサイズと同じ
"""

# エポック数(とりあえず50スタート)
nb_epoch = 200
# バッチサイズ(とりあえず100スタート)
nb_batch = 100
# 入力サイズ(とりあえず64スタート)！！いじらない！！
size = 64
# 一枚目のフィルタサイズ(とりあえず7スタート)
f_size = 14
# for文の1つ目のフィルタサイズ(とりあえず1スタート)
f_size1 = 1
# for文の2つ目のフィルタサイズ(とりあえず3スタート)
f_size2 = 3
# for文の3つ目のフィルタサイズ(とりあえず1スタート)
f_size3 = 1
# 初期フィルタ枚数(とりあえず64スタート)
nb_filter = 64
# L2正則化の値(とりあえず1e-4スタート)
nb_l2 = 1e-4
# Dropoutの1つ目（for文の中にある）(とりあえず0.3スタート)
nb_dropout1 = 0.3
# Dropoutの2つ目(とりあえず0.4スタート)
nb_dropout2 = 0.4
# Dropoutの3つ目(とりあえず0.4スタート)
nb_dropout3 = 0.4

"""
データの取得
"""
# data-fileの場所（google-colabの場合、google-driveのルートが、"drive/My Drive" となる）
data_path = os.path.join("drive", "My Drive", "Colab Notebooks")
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

"""
【matplotlib.pyplotの基本】
※pyplotはpltとしてimportしておく
①figure（ウィンドウ）を作成する
②figure上にaxes（グラフ領域）を１つ以上作成する
③axes上にグラフを作成する（axes.plot()なら折れ線グラフ、など）
④show
"""
# 画像チェック
fig = plt.figure(figsize=(10, 5))  # figure-sizeはインチ単位
ax = fig.add_subplot(121)  # Figure内にAxesを追加。121 =「1行2列のaxesを作って、その1番目(1列目)をreturnしろ」
ax.imshow(x_train[0])  # 画像ならimshow()
plt.show()  # 最後はpltに戻る

"""
データの加工
"""
# 画素を0~1の範囲に変換(正規化)。入力される値は０付近でないと、訓練が安定しない。
x_train = x_train / 255
x_test = x_test / 255

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
ki = 'he_normal'
kr = regularizers.l2(nb_l2)
x = Conv2D(filters=nb_filter, kernel_size=f_size, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(inputs)
x = MaxPooling2D(pool_size=2)(x)
n = 5
for i in range(n):
    shortcut = x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=nb_dropout1)(x)
    x = Conv2D(filters=nb_filter * (2 ** i), kernel_size=f_size1, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=nb_filter * (2 ** i), kernel_size=f_size2, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=nb_filter * (2 ** (i + 2)), kernel_size=f_size3, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = Concatenate()([x, shortcut])
    if i != (n - 1):
        x = MaxPooling2D(pool_size=2)(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(rate=nb_dropout2)(x)
x = Dense(units=101, kernel_initializer=ki, kernel_regularizer=kr)(x)
x = BatchNormalization()(x)
x = Activation('softmax')(x)
x = Dropout(rate=nb_dropout3)(x)
model = Model(inputs=inputs, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# try
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.1
)

train_generator = datagen.flow(
    x_train,
    y_train,
    batch_size=nb_batch,
    shuffle=True,
    subset="training")
val_generator = datagen.flow(
    x_train,
    y_train,
    batch_size=nb_batch,
    shuffle=True,
    subset="validation")

# 訓練
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(x_train) // nb_batch,
    epochs=nb_epoch,
    validation_data=val_generator,
    validation_steps=len(x_train) * 0.1 // nb_batch,
)

score = model.evaluate(x_test, y_test)
print('test_loss:', score[0])
print('test_acc:', score[1])

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.plot(range(nb_epoch), history.history['acc'], label='training')
ax.plot(range(nb_epoch), history.history['val_acc'], label='validation')
ax.set_title('acc')
ax.legend()
ax = fig.add_subplot(122)
ax.plot(range(nb_epoch), history.history['loss'], label='training')
ax.plot(range(nb_epoch), history.history['val_loss'], label='validation')
ax.set_title('loss')
ax.legend()
plt.show()
