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
import numpy as np
import glob
import gc

# pycharmでは、F4 を押すと、引用元・定義元に飛べるので使ってみよう

"""
tpuでの注意点
・kerasではなくtensorflow.kerasをimportしていることを確認。
・tensorflow1.13向けコードに変えた。colabで新しいコードブロックに以下を書いて実行してから、プログラムを動かすべし。
    !pip3 uninstall tensorflow
    !pip3 install tensorflow==1.13.2
"""
tpu = True  # colabでtpu使うときはTrueにする

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model  # kerasというライブラリのmodelsパッケージにある、Sequential, Modelという何か（ファイル or 関数 or クラス）を使うよ～
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if tpu:
    from tensorflow.contrib.tpu.python.tpu import keras_support
    # tpu使用時のバグ対策で、tensorflow.train パッケージの中にあるoptimizerを使う必要がある。
    optim = tf.train.AdamOptimizer(learning_rate=1e-3)  # optimizerはここで変更する
else:
    optim = 'adam'  # optimizerはここで変更する

"""
data path
"""
data_path = os.path.join("data")  # colabの場合、google-driveのルートが、"drive/My Drive" となる
if tpu:
    data_path = os.path.join("drive", "My Drive", "data")  # colabの場合、google-driveのルートが、"drive/My Drive" となる
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

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
x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
x = layers.MaxPool2D(2)(x)
x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D(2)(x)
x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D(2)(x)
x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D(2)(x)
x = layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(101, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

# 各種設定
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy

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
                assert batch_size <= x.shape[0]

            if not data_aug:
                datagen = ImageDataGenerator(rescale=1 / 255.)  # いじらない。rescaleで画像を正規化している。
            else:
                datagen = ImageDataGenerator(rescale=1 / 255.)  # DataAugmentationするなら、引数（rescale以外）をいじる。

            generator = datagen.flow(
                x,
                y,
                batch_size=batch_size,
                shuffle=True, )
            for _ in range(x.shape[0] // batch_size):
                yield next(generator)

"""
画像を256*256にしてから激重なので、pycharm上でデバッグしたいときは、
以下4つの値を全て小さめに設定すれば、動きが確認できる程度に軽くなるはず（colabで訓練するときは値を戻すことに注意）

tpuでは、batch_sizeは、「メモリが耐える範囲でなるべく大きな、8の倍数」にした方がいい。
ただし、今回は1つのh5ファイルのサイズが5050なので、それよりは小さくしてください。
"""
epochs = 10
batch_size = 1024
steps_per_epoch = 15   # 元は 15（h5ファイル数に合わせて）
validation_steps = 5   # 元は  5（h5ファイル数に合わせて）

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

# 訓練ループ fitの代わり
for e in range(epochs):
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    print("=" * 30)
    print(f"epoch: {e + 1}/{epochs}")

    # train
    for step in range(steps_per_epoch):
        x_batch, y_batch = next(flow_from_h5(train_path, batch_size, data_aug=True))
        loss, acc = model.train_on_batch(x_batch, y_batch)
        train_loss += loss
        train_acc += acc
        print(f"\rtrain step: {step + 1}/{steps_per_epoch}", end="")
    print()

    # validation
    for step in range(validation_steps):
        x_batch, y_batch = next(flow_from_h5(test_path, batch_size, data_aug=False))
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
    early stopping, 学習率減衰は、ここ（ループ内）にコードを追加するといけるはず
    """

# 評価ループ evaluateの代わり
test_loss, test_acc = 0, 0
for step in range(validation_steps):
    x_batch, y_batch = next(flow_from_h5(test_path, batch_size, data_aug=False))
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
ax.plot(range(epochs), history['train_acc'], label='training')  # x軸、y軸、ラベル
ax.plot(range(epochs), history['val_acc'], label='validation')
ax.set_title('acc')
ax.legend()  # 凡例を表示する

ax = fig.add_subplot(122)
ax.plot(range(epochs), history['train_loss'], label='training')
ax.plot(range(epochs), history['val_loss'], label='validation')
ax.set_title('loss')
ax.legend()  # 凡例を表示する

plt.show()  # 表示

# いろんなハイパラを全てハードコーディングしていますが、
# 通常はプログラムの最初にまとめたり、ハイパラだけをまとめたファイルから読み込んだりします。
