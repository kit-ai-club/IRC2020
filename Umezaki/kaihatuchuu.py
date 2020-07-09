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
import glob
import gc

"""
ハイパラ調整
"""
epochs = 100
batch_size = 100
steps_per_epoch = 75750 // batch_size + 1  # 元は 75750 // batch_size + 1
validation_steps = 25250 // batch_size + 1  # 元は  25250 // batch_size + 1
"""
データの取得
"""
# data-fileの場所（kaggleからDLする）
data_path = os.path.join("drive", "My Drive")
train_h5_path = os.path.join(data_path, "food_c101_n10099_r64x64x3 (1).h5")  # train-data   pathの文字列を確認してみよう
test_h5_path = os.path.join(data_path, "food_test_c101_n1000_r64x64x3 (1).h5")  # test-data
checkpoint_filepath = os.path.join(data_path, "modelimage.h5")
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
fig = plt.figure(figsize=(3, 3))  # figure-sizeはインチ単位
ax = fig.add_subplot(111)  # Figure内にAxesを追加。121 =「1行2列のaxesを作って、その1番目(1列目)をreturnしろ」
ax.imshow(x_train[0])  # 画像ならimshow()
# 最後はpltに戻る
x_train = x_train / 255.0
x_test = x_test / 255.0

"""
resnetの部分
"""
inputs = Input(shape=x_train.shape[1:])
f = 64 #ここも変えるべきだろうか？
ki = 'he_normal'
kr = regularizers.l2(1e-11)
x = Conv2D(filters=f, kernel_size=3, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(inputs)
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

# 以降は同じ
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])  # metrics=評価関数、acc=accuracy
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

"""
Data Augmentation
"""
from keras.preprocessing.image import ImageDataGenerator

# メモリリーク対策のグローバル変数
x, y, datagen, generator = None, None, None, None  # いじらない
def flow_from_h5(directory, batch_size, data_aug=False):
    """
    directory内のh5ファイルを順に読み込んで出力するジェネレータ。
    ImageDataGeneratorの部分以外はいじらない。
    """

    files = glob.glob(os.path.join(directory, '*.h5'))#[*]はワイルドカード文字っていって長さ0文字以上の任意の文字列にマッチする。今回なら.h5が末尾につくファイルを全部取得する。
    while True:#これなに？ずっと回すことにならん？whileって条件が1なら回し続けるんやろ？
        for file in files:#filesの中にあるfileの数だけ回す
            global x, y, datagen, generator #関数内だと変数を変更しても、その値を返さない限り変更の内容は保存されないが、global宣言を行うと返り値の設定無しで値を変えれる
            del x, y, datagen, generator
            gc.collect()#不要になったメモリ領域を自動的に解放する機能です.とのことですが、どゆことですかい？？まぁメモリの使用量減らして他に割り当てるってことかな？

            with h5py.File(file, 'r') as f:
                x = f['images'].value
                y = f['category'].value

            if not data_aug:#いつものifの逆で、条件を満たさないとき実行
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














"""
datagen = ImageDataGenerator(
    featurewise_center=False,#データセット全体で、入力の平均を０にする。これいんのかな
    featurewise_std_normalization=False,#入力をデータセットの標準偏差で正規化する。さすがはNormalization
    rotation_range=90,#画像をランダムに回転する回転範囲
    width_shift_range=0.2,#ランダムに水平シフト
    height_shift_range=0.2,#ランダムに垂直シフト
    horizontal_flip=True,#ランダムに水平方向反転
    vertical_flip=True,#ランダムに垂直方向反転
    #zoom_range=10,
    validation_split=0.1)#ランダムにズームする範囲

from keras import callbacks#下記のModelCheckpointはEpoch終了後の各数値（acc,loss,val_acc,val_loss)を監視して条件が揃った場合モデルを保存する

g = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True, subset='training')
v = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True, subset='validation')

modelcheckpoint = callbacks.ModelCheckpoint(filepath = checkpoint_filepath,#重みのファイル名そのもの
                                  monitor='loss',#監視する値
                                  verbose=1,#1なら結果表示
                                  save_best_only=True,#判定結果から保存を決定
                                  save_weights_only=False,#True=モデルの重みが保存False＝モデル全体を保存
                                  mode='min',#小さい時保存
                                  period=1)#何epoch数ごとに


er_stop = callbacks.EarlyStopping(monitor='loss', min_delta=0.07, patience=3, verbose=1, mode='min')

for e in range(epochs):#epoch数分だけ回す。今回は10
    #print('Epoch', e)
    batches = 0
    for x_batch, y_batch in g:
        history = model.fit_generator(g,#history追加してみた
                            steps_per_epoch=len(x_train) / batch_size,
                            epochs=epochs,
                            callbacks=[modelcheckpoint,er_stop])  # ここの訓練にcallbacksを追
        batches += 1
        if batches >= len(x_train) / 64:
            # we need to break the loop by hand because the generator loops indefinitely
            # これはなんでなん？決まった数まわしてるんじゃないんか
            #あー普通にあれか、dataが無限に作られるからか
            break
#model.load_weights(checkpoint_filepath)

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
"""