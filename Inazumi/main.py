import os
import matplotlib.pyplot as plt
import h5py
# kerasというライブラリのmodelsパッケージにある、Modelを使う
from keras.models import Model
import keras.layers as layers
# pycharmでは、F4 を押すと、引用元・定義元に飛べるので使ってみよう

# パラメータの設定
epochs = 10
batch_size = 100

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
ax = fig.add_subplot(122)
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
# Resnet 楽に最適化するため，設計部分は関数化したほうが良い
inputs = layers.Input(shape=(64, 64, 3))
filter_size = 64
ki = "he_normal"
cycle = 5
x = layers.Conv2D(filters=filter_size, kernel_size=(7, 7), strides=(1, 1), kernel_initializer=ki, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
# forループ内はCNN層
for i in range(cycle):
    shortcut = x

    x = layers.Conv2D(filters=filter_size, kernel_size=(1, 1), kernel_initializer=ki, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.3)(x)

    x = layers.Conv2D(filters=filter_size*2, kernel_size=(3, 3), padding= "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=filter_size*4, kernel_size=(3, 3), padding= "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Concatenate([a1,a2,...], axis=0,1...) -> 2個以上の配列を軸指定して結合
    # axis = 0 -> 1次元配列　axis = 1 -> 2次元配列　...　
    x = layers.Concatenate()([x, shortcut])
    if i != (cycle - 1):
        x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.3)(x)
x = layers.Dense(101)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('softmax')(x)

model = Model(inputs=inputs, outputs=x)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy

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

# いろんなハイパラを全てハードコーディングしていますが、
# 通常はプログラムの最初にまとめたり、ハイパラだけをまとめたファイルから読み込んだりします。
