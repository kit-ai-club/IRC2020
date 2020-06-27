import os
import matplotlib.pyplot as plt
import h5py
# kerasというライブラリのmodelsパッケージにある、Modelを使う
from keras.models import Model
from keras import backend as K
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
# L2正則化のパッケージ
from keras.regularizers import l2
# pycharmでは、F4 を押すと、引用元・定義元に飛べる

# ハイパーパラメータの設定
epochs = 10
batch_size = 100
optimizer = "rmsprop"
loss = "categorical_crossentropy"

# Conv2Dに関するパラメータ
kernel_init = "he_normal"
kernel_regularizer = l2(1.0e-4)
default_strides = (1, 1)

def shortcut(x, residual):
    """shortcut connection を作成する関数
        x: CNNを通す前の結果
        residual: CNNを通した後の結果,残余
        return:
    """
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)

    if x_shape == residual_shape:
        # x と residual の形状が同じ場合
        shortcut = x
    else:
        # x と residualの形状が異なる場合
        # x と residualの形状を一致させる
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_w, stride_h),
                          kernel_initializer=kernel_init,
                          kernel_regularizer=kernel_regularizer,
                          )(x)
    return Add()([shortcut, residual])

def ResNetConv2D(filters,kernel_size, strides, x):
    """
        ResNetConv2D : Conv2Dを計算するヘルパー関数
    """
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding="same",
                  kernel_initializer=kernel_init,
                  kernel_regularizer=kernel_regularizer)(x)
    return conv

def bn_relu_conv(filters, kernel_size, strides, x):
    """
        BatchNormalization -> Activation -> Conv の順に入れ込みを行う
    """
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    conv = ResNetConv2D(filters, kernel_size, strides, x)
    return conv

# Architecture type
def plain_block(filters, first_strides, is_first_block_of_first_layer):
    """
        plain_block: plain_blockを生成する
        filters: フィルター数
        first_strides: 初めに指定する畳み込みのストライド
        is_first_block_of_first_layer: 初めに入力する residualかどうか
        return: plain_blockを実現する関数
    """

    def f(x):

        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters, (3, 3), default_strides, x)
        else:
            conv1 = bn_relu_conv(filters, (3, 3), first_strides, x)
        conv2 = bn_relu_conv(filters, (3, 3), default_strides, conv1)

        return shortcut(x, conv2)

    return f

def bottleneck_block(filters, first_strides, is_first_block_of_first_layer):
    """
        bottleneck_block: bottleneck_blockを生成する
        filters: フィルター数
        first_strides: 初めに指定する畳み込みのストライド
        is_first_block_of_first_layer: 初めに入力するresidualかどうか
        return: bottleneck_blockを実現する関数
    """

    def f(x):

        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters, (3, 3), default_strides, x)
        else:
            conv1 = ResNetConv2D(filters, (1, 1), first_strides, x)
        conv2 = bn_relu_conv(filters, (3, 3), default_strides, conv1)
        conv3 = bn_relu_conv(filters * 4, (1, 1), default_strides, conv2)
        return shortcut(x, conv3)

    return f

def residual_blocks(block_function, filters, repetitions, is_first_layer):
    """
        residual block を反復する構造を作成する。
        block_function: residual block を作成する関数
        filters: フィルター数
        repetitions: residual block を何個繰り返すか。
        is_first_layer: max pooling 直後かどうか
    """

    def f(x):
        for i in range(repetitions):
            # conv3_x, conv4_x, conv5_x の最初の畳み込みは、
            # プーリング目的の畳み込みなので、strides を (2, 2) にする。
            # ただし、conv2_x の最初の畳み込みは直前の max pooling 層でプーリングしているので
            # strides を (1, 1) にする。
            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)

            x = block_function(
                filters=filters,
                first_strides=first_strides,
                is_first_block_of_first_layer=(i == 0 and is_first_layer),
            )(x)
        return x

    return f

"""
データの取得
"""
# data-fileの場所（google-colabの場合、google-driveのルートが、"drive/My Drive" となる）
# ローカル環境でのディレクトリ
# ../data -> 'IRC2020'フォルダに戻り，'data'フォルダを参照する，
# data_path = "../data"
# google-colab 上のディレクトリ
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

# test-data
with h5py.File(test_h5_path, 'r') as file:
    print(file.keys())
    x_test = file['images'].value
    y_test = file['category'].value

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

# Resnet層の作成
# ResNetに関するパラメータ
inputs = Input(shape=(64, 64, 3))
filter_size = 64
cycle = 5
block_fn = bottleneck_block
# plain_block -> Plain アーキテクチャ
# bottleneck_block -> Bottleneck アーキテクチャ

# 第1層
x = Conv2D(filters=filter_size,
           kernel_size=(7, 7),
           strides=(1, 1),
           kernel_initializer=kernel_init,
           padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
block = x
# 第2層　~ (最終層 - 1)層
for i in range(cycle):
    block = residual_blocks(block_fn,
                            filters=filter_size,
                            repetitions=1,
                            is_first_layer=(i == 0))(block)
    # filter_size *= 2

x = BatchNormalization()(block)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
outputs = Dense(units=101, kernel_initializer=kernel_init, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])  # metrics=評価関数、acc=accuracy

history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score = model.evaluate(x_test, y_test)
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
