from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# 載入CIFAR-10資料集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 顯示訓練集的形狀
print(x_train.shape)  # (50000, 32, 32, 3)

# 資料正規化：將數據標準化為均值為0，標準差為1
def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

# 正規化訓練集和測試集
x_train, x_test = normalize(x_train, x_test)

# 將標籤進行One-Hot編碼
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train).toarray()
y_test = one_hot.transform(y_test).toarray()

# 建立卷積神經網絡模型
classifier = Sequential()

# 第一個卷積層 + 批標準化層 + 最大池化層
classifier.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 第二個卷積層 + 批標準化層 + 最大池化層
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 展平層：將多維的輸出展平成一維
classifier.add(Flatten())

# 全連接層
classifier.add(Dense(output_dim=100, activation='relu'))
classifier.add(Dropout(p=0.3))  # Dropout層，隨機丟棄一些神經元來避免過擬合
classifier.add(Dense(output_dim=10, activation='softmax'))  # 輸出層，10個類別

# 編譯模型：使用交叉熵損失函數
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
classifier.fit(x_train, y_train, batch_size=100, epochs=100)

# 評估測試集上的表現
score = classifier.evaluate(x_test, y_test)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
