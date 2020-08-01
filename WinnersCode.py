"""
Dropout0.35 ResNet18 batch_size:1024 optimizer:Adam(0.001) 30~
"""
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
import gc

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model  # kerasというライブラリのmodelsパッケージにある、Sequential, Modelという何か（ファイル or 関数 or クラス）を使うよ～
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2


# pycharmでは、F4 を押すと、引用元・定義元に飛べるので使ってみよう

"""
画像を256*256にしてから激重なので、pycharm上でデバッグしたいときは、
以下4つの値を全て小さめに設定すれば、動きが確認できる程度に軽くなるはず（colabで訓練するときは値を戻すことに注意）

tpuでは、batch_sizeは、「メモリが耐える範囲でなるべく大きな、8の倍数」にした方がいい。
ただし、今回は1つのh5ファイルのサイズが5050なので、それよりは小さくしてください。
"""
epochs = 50
batch_size = 1024
wide_k = 1
steps_per_epoch = 5050 // batch_size * 15  # 元は  5050 // batch_size * 15 （15=trainのh5ファイル数）
validation_steps = 5050 // batch_size * 5  # 元は  5050 // batch_size * 5  （5=testのh5ファイル数）
# 0~30   lr = 1e-3
# 30~52  lr = 1e-4
opt_param = tf.train.AdamOptimizer(learning_rate=1e-3)  # optimizerはここで変更する

root_path = os.path.join("drive", "My Drive")
model_path = os.path.join(root_path, 'ResNet18_model.h5')    #model保存用
weights_path = os.path.join(root_path, 'ResNet18_normal.hdf5')  #weight保存用

"""
early stopping 関連の変数・パラメタ
"""
loss_min = float('inf')  # ロスの最小値を格納する。最初は無限大で定義しておく。
patience = 0  # loss_minが更新されないまま、何エポック経過したかを格納。0で定義しておく。
patience_max = 5  # patienceがpatience_maxに達するまでloss_minが更新されなかったら打ち切る。必要ならいじる。

# Conv2Dに関するパラメータ
kernel_init = "he_normal"
kernel_regularizer = l2(1e-4)
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
        shortcut = layers.Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_w, stride_h),
                          kernel_initializer=kernel_init,
                          kernel_regularizer=kernel_regularizer,
                          )(x)
    return layers.Add()([shortcut, residual])

def ResNetConv2D(filters,kernel_size, strides, x):
    """
        ResNetConv2D : Conv2Dを計算するヘルパー関数
    """
    conv = layers.Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer=kernel_init,
                  kernel_regularizer=kernel_regularizer)(x)
    return conv

def bn_relu_conv(filters, kernel_size, strides, x, Drop_chk):
    """
        BatchNormalization -> Activation -> Conv の順に入れ込みを行う
    """
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if Drop_chk == True:
      x = layers.Dropout(0.4)(x)
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
            conv1 = bn_relu_conv(filters, (3, 3), first_strides, x, False)

        #conv1 = layers.Dropout(0.4)(conv1)
        conv2 = bn_relu_conv(filters, (3, 3), default_strides, conv1, True)

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
        conv2 = bn_relu_conv(filters, (3, 3), default_strides, conv1, False)
        conv3 = bn_relu_conv(filters * 4, (1, 1), default_strides, conv2, False)

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
                filters=filters*wide_k,
                first_strides=first_strides,
                is_first_block_of_first_layer=(i == 0 and is_first_layer),
            )(x)
        return x

    return f

"""
tpuでの注意点
・kerasではなくtensorflow.kerasをimportしていることを確認。
・tensorflow1.13向けコードに変えた。colabで新しいコードブロックに以下を書いて実行してから、プログラムを動かすべし。
    !pip3 uninstall tensorflow
    !pip3 install tensorflow==1.13.2
"""
tpu = True  # colabでtpu使うときはTrueにする

if tpu:
    from tensorflow.contrib.tpu.python.tpu import keras_support

    # tpu使用時のバグ対策で、tensorflow.train パッケージの中にあるoptimizerを使う必要がある。
    optim = opt_param
else:
    optim = 'adam'  # optimizerはここで変更する

"""
data path
"""
data_path = os.path.join("drive", "My Drive", "data")  # colabの場合、google-driveのルートが、"drive/My Drive" となる
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")


""" 
  cycle: residual_blocksをblockタイプに応じて何回反復させるか
  block_fn: blockタイプ
  cycle:[2, 2, 2, 2] block_fn: plain_block   ResNet18
  cycle:[3, 4, 6, 3] block_fn: plain_block   ResNet34
  cycle:[3, 4, 6, 3] block_fn: bottleneck_block ResNet50
  cycle:[3, 4, 23, 3] block_fn: bottleneck_block ResNet101
  cycle:[3, 8, 36, 3] block_fn: bottleneck_block ResNet152
"""
filter_size = 64
cycle = [2, 2, 2, 2]
block_fn = plain_block
# plain_block -> Plain アーキテクチャ
# bottleneck_block -> Bottleneck アーキテクチャ

"""
keras Functinal API での流れ
①Input()            入力の作成
②x = f(x)         　層の追加・出力の作成
③model = Model()    箱の作成
④model.compile()     設定
⑤model.fit()  　　　  訓練
⑥model.evaluate()　　 評価
"""
inputs = layers.Input(shape=(256, 256, 3))
# 第1層
x = layers.Conv2D(filters=filter_size,
           kernel_size=(7, 7),
           strides=(1, 1),
           kernel_initializer=kernel_init,
           padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
block = x
# ResNet18の実装
for i, r in enumerate(cycle):
    block = residual_blocks(block_fn,
                            filters=filter_size,
                            repetitions=r,
                            is_first_layer=(i == 0))(block)
    filter_size *= 2

x = layers.GlobalAveragePooling2D()(block)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
outputs = layers.Dense(units=101, kernel_initializer=kernel_init, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# 各種設定
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy
model.load_weights(weights_path)
model.summary()     # modelの概略図

# メモリリーク対策のグローバル変数
x, y, datagen, generator = None, None, None, None  # いじらない


def flow_from_h5(directory, batch_size, data_aug=False):
    """
    directory内のh5ファイルを順に読み込んで出力するジェネレータ。
    ImageDataGeneratorの部分以外はいじらない。
    """
    files = glob.glob(os.path.join(directory, '*.h5'))
    assert len(files) != 0
    while True:
        for file in files:
            global x, y, datagen, generator
            del x, y, datagen, generator
            gc.collect()

            with h5py.File(file, 'r') as f:  # 'r' = read mode で読み込んで、変数fに格納
                x = f['images'].value  # train dataであるimagesにアクセス。さらに、.value と書くとnumpy形式にしてくれる
                y = f['category'].value  # 教師データ（正解データ）。すでにone-hot vector になっている
                assert batch_size <= x.shape[0], 'データ読み込み失敗。データどこにあるん？。data path 設定ミスってない？'

            if not data_aug:
                datagen = ImageDataGenerator(rescale=1 / 255.)  # いじらない。rescaleで画像を正規化している。
            else:
                datagen = ImageDataGenerator(rescale=1 / 255.,
                                             rotation_range=20,
                                             vertical_flip=True,
                                             horizontal_flip=True,
                                             height_shift_range=0.3,
                                             width_shift_range=0.3,
                                             shear_range=5,
                                             zoom_range=0.3,
                                             channel_shift_range=5.0,
                                             brightness_range=[0.3, 1.0]
                                             )  # DataAugmentationするなら、引数（rescale以外）をいじる。
            generator = datagen.flow(
                x,
                y,
                batch_size=batch_size,
                shuffle=True, )
            for _ in range(x.shape[0] // batch_size):  # 1ファイルあたり、5050 // batch_size 回学習
                yield next(generator)

"""
以降は、原則いじらない.
ただし、early stopping, 学習率減衰などを使う場合は、訓練ループ内をいじる。
"""
if tpu:
    tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
train_generator = flow_from_h5(train_path, batch_size, data_aug=True)
validation_generator = flow_from_h5(test_path, batch_size, data_aug=False)

# 訓練ループ fitの代わり
for e in range(epochs):
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    print("=" * 30)
    print(f"epoch: {e + 1}/{epochs}")

    # train
    for step in range(steps_per_epoch):
        x_batch, y_batch = next(train_generator)
        loss, acc = model.train_on_batch(x_batch, y_batch)
        train_loss += loss
        train_acc += acc
        print(f"\rtrain step: {step + 1}/{steps_per_epoch}", end="")
    print()

    # validation
    for step in range(validation_steps):
        x_batch, y_batch = next(validation_generator)
        loss, acc = model.test_on_batch(x_batch, y_batch)
        val_loss += loss
        val_acc += acc
        print(f"\rval step: {step + 1}/{validation_steps}", end="")
    print()

    print(f"train_loss: {train_loss / steps_per_epoch}, train_acc: {train_acc / steps_per_epoch}, val_loss: {val_loss / validation_steps}, val_acc: {val_acc / validation_steps}")
    history["train_loss"].append(train_loss / steps_per_epoch)
    history["train_acc"].append(train_acc / steps_per_epoch)
    history["val_loss"].append(val_loss / validation_steps)
    history["val_acc"].append(val_acc / validation_steps)

    """
    early stopping, 学習率減衰は、以下（ループ内）にコードを追加するといけるはず
    """
    # early stopping
    if val_loss > loss_min:
        patience += 1
        if patience >= patience_max:
            print('Early Stopping...')
            break
    else:
        patience = 0
        loss_min = val_loss
    """
    学習率変更
    """
    #if e >= 30: optim.lr = 1e-6
    #elif e >= 20: optim.lr = 1e-6
    """
    モデル保存
    """
    if e == 10: model.save_weights(weights_path)
    if e == 20: model.save_weights(weights_path)
    if e == 30: model.save_weights(weights_path)
    if e == 40: model.save_weights(weights_path)

# 評価ループ evaluateの代わり
test_loss, test_acc = 0, 0
for step in range(validation_steps):
    x_batch, y_batch = next(validation_generator)
    loss, acc = model.test_on_batch(x_batch, y_batch)
    test_loss += loss
    test_acc += acc
    print(f"\rtest step: {step + 1}/{validation_steps}", end="")
print()
print(f'test_loss: {test_loss / validation_steps}')
print(f'test_acc: {test_acc / validation_steps}')

"""
【matplotlib.pyplotの基本】
※pyplotはpltとしてimportしておく
①figure（ウィンドウ）を作成する
②figure上にaxes（グラフ領域）を１つ以上作成する
③axes上にグラフを作成する（axes.plot()なら折れ線グラフ、など）
④show
"""
# 訓練の推移
fig = plt.figure(figsize=(10, 5))  # figure-sizeはインチ単位

ax = fig.add_subplot(121)  # Figure内にAxesを追加。121 ->「1行2列のaxesを作って、その1番目(1列目)のaxesをreturnしろ」
ax.plot(range(len(history['train_acc'])), history['train_acc'], label='training')  # x軸、y軸、ラベル
ax.plot(range(len(history['val_acc'])), history['val_acc'], label='validation')
ax.set_title('acc')
ax.legend()  # 凡例を表示する

ax = fig.add_subplot(122)
ax.plot(range(len(history['train_loss'])), history['train_loss'], label='training')
ax.plot(range(len(history['val_loss'])), history['val_loss'], label='validation')
ax.set_title('loss')
ax.legend()  # 凡例を表示する

plt.show()  # 表示

model.save_weights(weights_path)
# いろんなハイパラを全てハードコーディングしていますが、
# 通常はプログラムの最初にまとめたり、ハイパラだけをまとめたファイルから読み込んだりします。