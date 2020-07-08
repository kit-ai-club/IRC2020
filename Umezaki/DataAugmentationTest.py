import PIL
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_FILE = "C:/Users/davinci/PycharmProjects/DavinciProject/Practice/chap1/cat.jpg"

# 画像をロード（PIL形式画像）
img = load_img(IMAGE_FILE)

print(img.size)

img = img.resize((img.width // 2, img.height // 2))

# numpyの配列に変換
x = img_to_array(img)

# 4次元配列に変換、以下同じ意味
# x = np.expand_dims(x, axis=0)
x = x.reshape((1,) + x.shape)

# （1,縦サイズ, 横サイズ, チャンネル数)
print(x.shape)

datagen = ImageDataGenerator(
    featurewise_center=False,  # データセット全体で、入力の平均を０にする。これいんのかな
    featurewise_std_normalization=False,  # 入力をデータセットの標準偏差で正規化する。さすがはNormalization
    rotation_range=90,  # 画像をランダムに回転する回転範囲
    width_shift_range=0.2,  # ランダムに水平シフト
    height_shift_range=0.2,  # ランダムに垂直シフト
    horizontal_flip=True,  # ランダムに水平方向反転
    vertical_flip=True,  # ランダムに垂直方向反転
    zoom_range=10
)  # ランダムにズームする範囲

n = 9
a = 1
b = 1
c = a * b

while c < n:
    if c < n:
        a += 1
        c = a * b
    if c < n:
        b += 1
        c = a * b
    print(c)

g = datagen.flow(x, batch_size=1)
for i in range(n):
    batches = g.next()

    # 画像として表示するため、４次元から3次元データにし、配列から画像にする。
    gen_img = array_to_img(batches[0])

    plt.subplot(a, b, i + 1)
    plt.imshow(gen_img)
    plt.axis('off')

plt.show()
