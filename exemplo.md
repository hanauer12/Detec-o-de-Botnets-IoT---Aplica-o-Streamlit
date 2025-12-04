Botnet Detection with an Autoencoder
20 May 2021 This notebook was created for a course at Istanbul Technical University.
* We implement (a simplified version of) the autoencoder-based anomaly detection described in the N-BaIoT paper [1].
[1] Meidan, Yair, et al. "N-BaIoT—Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders." IEEE Pervasive Computing 17.3 (2018): 12-22. https://arxiv.org/pdf/1805.03409


import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


print(tf.config.list_physical_devices("GPU"))

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]



1. N-BaIoT Dataset
2. Autoencoder Architecture
3. Python Reimlementation
4. Conclusion

1. N-BaIoT Dataset
https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
* Normal traffic was captured for 9 IoT devices connected to the network.
* Then, they were infected with Mirai and BASHLITE (aka gafgyt) malware.
* Traffic was captured for each device for different phases of the malware execution.
* From the network traffic, 115 features were extracted as described in [1].
For now, we start with data from a smart doorbell: normal execution and the different phases of Mirai.


def load_nbaiot(filename):
    return np.loadtxt(
        os.path.join("/kaggle/input/nbaiot-dataset", filename),
        delimiter=",",
        skiprows=1
    )

benign = load_nbaiot("1.benign.csv")
X_train = benign[:40000]
X_test0 = benign[40000:]
X_test1 = load_nbaiot("1.mirai.scan.csv")
X_test2 = load_nbaiot("1.mirai.ack.csv")
X_test3 = load_nbaiot("1.mirai.syn.csv")
X_test4 = load_nbaiot("1.mirai.udp.csv")
X_test5 = load_nbaiot("1.mirai.udpplain.csv")
X_test6 = load_nbaiot("1.gafgyt.udp.csv")
X_test7 = load_nbaiot("1.gafgyt.junk.csv")
X_test8 = load_nbaiot("1.gafgyt.scan.csv")
X_test9 = load_nbaiot("1.gafgyt.tcp.csv")
X_test10 = load_nbaiot("1.gafgyt.combo.csv")


print(X_train.shape, X_test0.shape, X_test1.shape, X_test2.shape,
      X_test3.shape, X_test4.shape, X_test5.shape,X_test6.shape,X_test7.shape,X_test8.shape,X_test9.shape,X_test10.shape)

(40000, 115) (9548, 115) (107685, 115) (102195, 115) (122573, 115) (237665, 115) (81982, 115) (105874, 115) (29068, 115) (29849, 115) (92141, 115) (59718, 115)



2. Autoencoder Architecture
Relevant parts of [1] describing the autoencoder architecture:
* The general idea is autoencoder-based anomaly detection, p. 4: [W]e use deep autoencoders and maintain a model for each IoT device separately. An autoencoder is a neural network which is trained to reconstruct its inputs after some compression. The compression ensures that the network learns the meaningful concepts and the relation among its input features. If an autoencoder is trained on benign instances only, then it will succeed at reconstructing normal observations, but fail at reconstructing abnormal observations (unknown concepts). When a significant re- construction error is detected, then we classify the given observations as being an anomaly. 
* Details, p. 5: Each autoencoder had an input layer whose dimension is equal to the number of features in the dataset (i.e., 115). As noted by [16] and [15], autoencoders effectively perform dimen- sionality reduction internally, such that the code layer be- tween the encoder(s) and decoder(s) efficiently compresses the input layer and reflects its essential characteristics. In our experiments, four hidden layers of encoders were set at decreasing sizes of 75%, 50%, 33%, and 25% of the input layer’s dimension. The next layers were decoders, with the same sizes as the encoders, however with an increasing order (starting from 33%). 
* Anomaly Detection threshold, p.4: This anomaly threshold, above which an instance is considered anomalous, is calculated as the sum of the sample mean and standard deviation of [the mean squared error over the validation set]. 
* Sequences of packets, p.4: Preliminary experiments revealed that deciding whether a device’s packet stream is anomalous or not based on a single instance enables very accurate detection of IoT-based botnet attacks (high TPR). However, benign instances were too often (in approximately 5-7% of cases) falsely marked as anomalous. Thus we base the abnormality decision on a sequence of instances by implementing a majority vote on a moving window. We determine the minimal window size ws∗ as the shortest sequence of instances, a majority vote which produces 0% FPR on [the validation set]. 
* Final hyperparameters for the Danmini smart doorbell, p. 5:
    * Learning rate: 0.012
    * Number of epochs: 800
    * Anomaly Threshold: 0.042
    * Window Size: 82

3. Python Reimplementation
Adapting this Keras tutorial: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/autoencoder.ipynb


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(115, activation="relu"),
            layers.Dense(86, activation="relu"),
            layers.Dense(57, activation="relu"),
            layers.Dense(37, activation="relu"),
            layers.Dense(28, activation="relu")
        ])
        self.decoder = Sequential([
            layers.Dense(37, activation="relu"),
            layers.Dense(57, activation="relu"),
            layers.Dense(86, activation="relu"),
            layers.Dense(115, activation="sigmoid")
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


How can we determine the hyperparameters?
* In Keras, the fault learning rate for Adam optimizer is 0.001. With that, training is relatively slow, so we quickly tried 0.01.
* We use Early Stopping to find the number of epochs.
* The anomaly threshold is calculated as one standard deviation above the mean of training data losses.


with tf.device("/GPU:0"):
    scaler = MinMaxScaler()
    x = scaler.fit_transform(X_train)
    
    
    ae = Autoencoder()
    ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-9,
        patience=20,
        verbose=1,
        mode='auto'
    )
    ae.fit(
        x=x,
        y=x,
        epochs=400,
        validation_split=0.3,
        shuffle=True,
        callbacks=[monitor]
    )
    
    training_loss = losses.mse(x, ae(x))
    threshold = np.mean(training_loss)+np.std(training_loss)

Epoch 1/400
875/875 [==============================] - 4s 2ms/step - loss: 0.0248 - val_loss: 0.0013
Epoch 2/400
875/875 [==============================] - 2s 2ms/step - loss: 0.0010 - val_loss: 0.0012
Epoch 3/400
875/875 [==============================] - 2s 2ms/step - loss: 7.7638e-04 - val_loss: 7.0625e-04
Epoch 4/400
875/875 [==============================] - 2s 2ms/step - loss: 5.2404e-04 - val_loss: 5.9256e-04
Epoch 5/400
875/875 [==============================] - 2s 2ms/step - loss: 4.3835e-04 - val_loss: 4.9042e-04
Epoch 6/400
875/875 [==============================] - 2s 2ms/step - loss: 4.0549e-04 - val_loss: 4.9152e-04
Epoch 7/400
875/875 [==============================] - 2s 2ms/step - loss: 3.6432e-04 - val_loss: 4.3290e-04
Epoch 8/400
875/875 [==============================] - 2s 2ms/step - loss: 2.8778e-04 - val_loss: 3.4163e-04
Epoch 9/400
875/875 [==============================] - 2s 2ms/step - loss: 2.9155e-04 - val_loss: 3.6342e-04
Epoch 10/400
875/875 [==============================] - 2s 2ms/step - loss: 2.6845e-04 - val_loss: 3.4596e-04
Epoch 11/400
875/875 [==============================] - 2s 2ms/step - loss: 2.4727e-04 - val_loss: 2.9032e-04
Epoch 12/400
875/875 [==============================] - 2s 2ms/step - loss: 2.3318e-04 - val_loss: 2.6449e-04
Epoch 13/400
875/875 [==============================] - 2s 2ms/step - loss: 1.9888e-04 - val_loss: 2.4464e-04
Epoch 14/400
875/875 [==============================] - 2s 2ms/step - loss: 1.8231e-04 - val_loss: 1.9158e-04
Epoch 15/400
875/875 [==============================] - 2s 2ms/step - loss: 1.3866e-04 - val_loss: 1.7957e-04
Epoch 16/400
875/875 [==============================] - 2s 2ms/step - loss: 1.4498e-04 - val_loss: 2.0010e-04
Epoch 17/400
875/875 [==============================] - 2s 2ms/step - loss: 1.3817e-04 - val_loss: 3.0856e-04
Epoch 18/400
875/875 [==============================] - 2s 2ms/step - loss: 1.9530e-04 - val_loss: 1.6575e-04
Epoch 19/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0686e-04 - val_loss: 1.5951e-04
Epoch 20/400
875/875 [==============================] - 2s 2ms/step - loss: 1.2851e-04 - val_loss: 1.3886e-04
Epoch 21/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0853e-04 - val_loss: 1.1709e-04
Epoch 22/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0909e-04 - val_loss: 1.2095e-04
Epoch 23/400
875/875 [==============================] - 2s 2ms/step - loss: 1.1635e-04 - val_loss: 1.4220e-04
Epoch 24/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0179e-04 - val_loss: 1.3130e-04
Epoch 25/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0383e-04 - val_loss: 1.1418e-04
Epoch 26/400
875/875 [==============================] - 2s 2ms/step - loss: 9.2353e-05 - val_loss: 1.1790e-04
Epoch 27/400
875/875 [==============================] - 2s 2ms/step - loss: 8.5592e-05 - val_loss: 1.1838e-04
Epoch 28/400
875/875 [==============================] - 2s 2ms/step - loss: 8.4535e-05 - val_loss: 1.0916e-04
Epoch 29/400
875/875 [==============================] - 2s 2ms/step - loss: 9.0985e-05 - val_loss: 1.1379e-04
Epoch 30/400
875/875 [==============================] - 2s 2ms/step - loss: 1.1095e-04 - val_loss: 9.8714e-05
Epoch 31/400
875/875 [==============================] - 2s 2ms/step - loss: 7.6561e-05 - val_loss: 1.0742e-04
Epoch 32/400
875/875 [==============================] - 2s 2ms/step - loss: 7.4185e-05 - val_loss: 1.1901e-04
Epoch 33/400
875/875 [==============================] - 2s 2ms/step - loss: 8.2174e-05 - val_loss: 9.6080e-05
Epoch 34/400
875/875 [==============================] - 2s 2ms/step - loss: 6.5561e-05 - val_loss: 8.0792e-05
Epoch 35/400
875/875 [==============================] - 2s 2ms/step - loss: 6.8539e-05 - val_loss: 1.9393e-04
Epoch 36/400
875/875 [==============================] - 2s 2ms/step - loss: 1.2617e-04 - val_loss: 1.0634e-04
Epoch 37/400
875/875 [==============================] - 2s 2ms/step - loss: 7.4865e-05 - val_loss: 9.4312e-05
Epoch 38/400
875/875 [==============================] - 2s 2ms/step - loss: 6.9327e-05 - val_loss: 8.3487e-05
Epoch 39/400
875/875 [==============================] - 2s 2ms/step - loss: 5.6754e-05 - val_loss: 9.1593e-05
Epoch 40/400
875/875 [==============================] - 2s 2ms/step - loss: 8.5944e-05 - val_loss: 7.6053e-05
Epoch 41/400
875/875 [==============================] - 2s 2ms/step - loss: 6.4408e-05 - val_loss: 9.4178e-05
Epoch 42/400
875/875 [==============================] - 2s 2ms/step - loss: 7.2368e-05 - val_loss: 7.9057e-05
Epoch 43/400
875/875 [==============================] - 2s 2ms/step - loss: 5.2784e-05 - val_loss: 9.9309e-05
Epoch 44/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0089e-04 - val_loss: 8.1291e-05
Epoch 45/400
875/875 [==============================] - 2s 2ms/step - loss: 6.3900e-05 - val_loss: 6.8306e-05
Epoch 46/400
875/875 [==============================] - 2s 2ms/step - loss: 5.8598e-05 - val_loss: 1.4456e-04
Epoch 47/400
875/875 [==============================] - 2s 2ms/step - loss: 9.3989e-05 - val_loss: 9.7734e-05
Epoch 48/400
875/875 [==============================] - 2s 2ms/step - loss: 8.0861e-05 - val_loss: 7.5106e-05
Epoch 49/400
875/875 [==============================] - 2s 2ms/step - loss: 5.5878e-05 - val_loss: 1.6624e-04
Epoch 50/400
875/875 [==============================] - 2s 2ms/step - loss: 9.7612e-05 - val_loss: 7.1872e-05
Epoch 51/400
875/875 [==============================] - 2s 2ms/step - loss: 1.0237e-04 - val_loss: 8.4353e-05
Epoch 52/400
875/875 [==============================] - 2s 2ms/step - loss: 5.9492e-05 - val_loss: 6.8816e-05
Epoch 53/400
875/875 [==============================] - 2s 2ms/step - loss: 5.0367e-05 - val_loss: 6.2788e-05
Epoch 54/400
875/875 [==============================] - 2s 2ms/step - loss: 5.3812e-05 - val_loss: 7.2108e-05
Epoch 55/400
875/875 [==============================] - 2s 2ms/step - loss: 4.3036e-05 - val_loss: 6.1349e-05
Epoch 56/400
875/875 [==============================] - 2s 2ms/step - loss: 5.6276e-05 - val_loss: 1.1392e-04
Epoch 57/400
875/875 [==============================] - 2s 2ms/step - loss: 7.1741e-05 - val_loss: 8.0639e-05
Epoch 58/400
875/875 [==============================] - 2s 2ms/step - loss: 5.2456e-05 - val_loss: 7.8671e-05
Epoch 59/400
875/875 [==============================] - 2s 2ms/step - loss: 5.5463e-05 - val_loss: 7.0592e-05
Epoch 60/400
875/875 [==============================] - 2s 2ms/step - loss: 5.4312e-05 - val_loss: 1.1724e-04
Epoch 61/400
875/875 [==============================] - 2s 2ms/step - loss: 6.8223e-05 - val_loss: 7.1872e-05
Epoch 62/400
875/875 [==============================] - 2s 2ms/step - loss: 4.2737e-05 - val_loss: 6.4872e-05
Epoch 63/400
875/875 [==============================] - 2s 2ms/step - loss: 3.6714e-05 - val_loss: 6.3267e-05
Epoch 64/400
875/875 [==============================] - 2s 2ms/step - loss: 4.4113e-05 - val_loss: 7.7730e-05
Epoch 65/400
875/875 [==============================] - 2s 2ms/step - loss: 1.4106e-04 - val_loss: 6.7689e-05
Epoch 66/400
875/875 [==============================] - 2s 2ms/step - loss: 7.7755e-05 - val_loss: 8.7786e-05
Epoch 67/400
875/875 [==============================] - 2s 2ms/step - loss: 5.0633e-05 - val_loss: 6.2870e-05
Epoch 68/400
875/875 [==============================] - 2s 2ms/step - loss: 4.6459e-05 - val_loss: 7.7425e-05
Epoch 69/400
875/875 [==============================] - 2s 2ms/step - loss: 5.1855e-05 - val_loss: 6.3803e-05
Epoch 70/400
875/875 [==============================] - 2s 2ms/step - loss: 4.5598e-05 - val_loss: 6.8490e-05
Epoch 71/400
875/875 [==============================] - 2s 2ms/step - loss: 7.4960e-05 - val_loss: 5.7073e-05
Epoch 72/400
875/875 [==============================] - 2s 2ms/step - loss: 7.1631e-05 - val_loss: 9.8261e-05
Epoch 73/400
875/875 [==============================] - 2s 2ms/step - loss: 5.1618e-05 - val_loss: 5.2637e-05
Epoch 74/400
875/875 [==============================] - 2s 2ms/step - loss: 3.8224e-05 - val_loss: 6.7200e-05
Epoch 75/400
875/875 [==============================] - 2s 2ms/step - loss: 6.3636e-05 - val_loss: 5.8906e-05
Epoch 76/400
875/875 [==============================] - 2s 2ms/step - loss: 4.2322e-05 - val_loss: 5.7383e-05
Epoch 77/400
875/875 [==============================] - 2s 2ms/step - loss: 3.8640e-05 - val_loss: 5.7319e-05
Epoch 78/400
875/875 [==============================] - 2s 2ms/step - loss: 3.7272e-05 - val_loss: 5.2643e-05
Epoch 79/400
875/875 [==============================] - 2s 2ms/step - loss: 3.8308e-05 - val_loss: 6.8151e-05
Epoch 80/400
875/875 [==============================] - 2s 2ms/step - loss: 5.0072e-05 - val_loss: 9.1856e-05
Epoch 81/400
875/875 [==============================] - 2s 2ms/step - loss: 4.3908e-05 - val_loss: 5.6555e-05
Epoch 82/400
875/875 [==============================] - 2s 2ms/step - loss: 3.9465e-05 - val_loss: 6.0333e-05
Epoch 83/400
875/875 [==============================] - 2s 2ms/step - loss: 3.4622e-05 - val_loss: 7.3263e-05
Epoch 84/400
875/875 [==============================] - 2s 2ms/step - loss: 4.5281e-05 - val_loss: 7.6630e-05
Epoch 85/400
875/875 [==============================] - 2s 2ms/step - loss: 4.2419e-05 - val_loss: 6.2425e-05
Epoch 86/400
875/875 [==============================] - 2s 2ms/step - loss: 5.3014e-05 - val_loss: 2.0518e-04
Epoch 87/400
875/875 [==============================] - 2s 2ms/step - loss: 6.9070e-05 - val_loss: 5.4853e-05
Epoch 88/400
875/875 [==============================] - 2s 2ms/step - loss: 2.9369e-05 - val_loss: 4.9997e-05
Epoch 89/400
875/875 [==============================] - 2s 2ms/step - loss: 3.1955e-05 - val_loss: 4.9763e-05
Epoch 90/400
875/875 [==============================] - 2s 2ms/step - loss: 5.4065e-05 - val_loss: 4.7544e-05
Epoch 91/400
875/875 [==============================] - 2s 2ms/step - loss: 3.0135e-05 - val_loss: 5.1166e-05
Epoch 92/400
875/875 [==============================] - 2s 2ms/step - loss: 3.7005e-05 - val_loss: 5.7777e-05
Epoch 93/400
875/875 [==============================] - 2s 2ms/step - loss: 3.5228e-05 - val_loss: 6.8032e-05
Epoch 94/400
875/875 [==============================] - 2s 2ms/step - loss: 5.3141e-05 - val_loss: 4.9142e-05
Epoch 95/400
875/875 [==============================] - 2s 2ms/step - loss: 3.1666e-05 - val_loss: 7.2177e-05
Epoch 96/400
875/875 [==============================] - 2s 2ms/step - loss: 3.7399e-05 - val_loss: 4.9227e-05
Epoch 97/400
875/875 [==============================] - 2s 2ms/step - loss: 3.1606e-05 - val_loss: 6.7916e-05
Epoch 98/400
875/875 [==============================] - 2s 2ms/step - loss: 3.8759e-05 - val_loss: 6.7392e-05
Epoch 99/400
875/875 [==============================] - 2s 2ms/step - loss: 6.3679e-05 - val_loss: 8.6895e-05
Epoch 100/400
875/875 [==============================] - 2s 2ms/step - loss: 4.9777e-05 - val_loss: 5.4194e-05
Epoch 101/400
875/875 [==============================] - 2s 2ms/step - loss: 2.9269e-05 - val_loss: 4.7584e-05
Epoch 102/400
875/875 [==============================] - 2s 2ms/step - loss: 2.9911e-05 - val_loss: 4.7381e-05
Epoch 103/400
875/875 [==============================] - 2s 2ms/step - loss: 3.1002e-05 - val_loss: 4.9793e-05
Epoch 104/400
875/875 [==============================] - 2s 2ms/step - loss: 3.2124e-05 - val_loss: 1.2106e-04
Epoch 105/400
875/875 [==============================] - 2s 2ms/step - loss: 7.1796e-05 - val_loss: 6.7910e-05
Epoch 106/400
875/875 [==============================] - 2s 2ms/step - loss: 4.2192e-05 - val_loss: 4.8231e-05
Epoch 107/400
875/875 [==============================] - 2s 2ms/step - loss: 2.6134e-05 - val_loss: 5.2173e-05
Epoch 108/400
875/875 [==============================] - 2s 2ms/step - loss: 4.5109e-05 - val_loss: 5.1905e-05
Epoch 109/400
875/875 [==============================] - 2s 2ms/step - loss: 2.4304e-05 - val_loss: 6.4637e-05
Epoch 110/400
875/875 [==============================] - 2s 2ms/step - loss: 3.1980e-05 - val_loss: 5.9206e-05
Epoch 111/400
875/875 [==============================] - 2s 2ms/step - loss: 2.9479e-05 - val_loss: 9.4571e-05
Epoch 112/400
875/875 [==============================] - 2s 2ms/step - loss: 3.1134e-05 - val_loss: 4.5216e-05
Epoch 113/400
875/875 [==============================] - 2s 2ms/step - loss: 5.9911e-05 - val_loss: 6.5266e-05
Epoch 114/400
875/875 [==============================] - 2s 2ms/step - loss: 4.0264e-05 - val_loss: 7.8948e-05
Epoch 115/400
875/875 [==============================] - 2s 2ms/step - loss: 5.4728e-05 - val_loss: 4.4814e-05
Epoch 116/400
875/875 [==============================] - 2s 2ms/step - loss: 2.7601e-05 - val_loss: 4.5153e-05
Epoch 117/400
875/875 [==============================] - 2s 2ms/step - loss: 2.7517e-05 - val_loss: 5.2942e-05
Epoch 118/400
875/875 [==============================] - 2s 2ms/step - loss: 2.5576e-05 - val_loss: 5.3340e-05
Epoch 119/400
875/875 [==============================] - 2s 2ms/step - loss: 2.7373e-05 - val_loss: 6.9770e-05
Epoch 120/400
875/875 [==============================] - 2s 2ms/step - loss: 3.0311e-05 - val_loss: 6.0957e-05
Epoch 121/400
875/875 [==============================] - 2s 2ms/step - loss: 7.1142e-05 - val_loss: 1.0097e-04
Epoch 122/400
875/875 [==============================] - 2s 2ms/step - loss: 6.7519e-05 - val_loss: 7.0413e-05
Epoch 123/400
875/875 [==============================] - 2s 2ms/step - loss: 4.9190e-05 - val_loss: 6.6984e-05
Epoch 124/400
875/875 [==============================] - 2s 2ms/step - loss: 3.8252e-05 - val_loss: 6.5936e-05
Epoch 125/400
875/875 [==============================] - 2s 2ms/step - loss: 3.8948e-05 - val_loss: 5.1071e-05
Epoch 126/400
875/875 [==============================] - 2s 2ms/step - loss: 2.5191e-05 - val_loss: 5.4360e-05
Epoch 127/400
875/875 [==============================] - 2s 2ms/step - loss: 3.3535e-05 - val_loss: 4.7477e-05
Epoch 128/400
875/875 [==============================] - 2s 2ms/step - loss: 3.0313e-05 - val_loss: 6.0746e-05
Epoch 129/400
875/875 [==============================] - 2s 2ms/step - loss: 4.3949e-05 - val_loss: 8.8363e-05
Epoch 130/400
875/875 [==============================] - 2s 2ms/step - loss: 3.4421e-05 - val_loss: 5.2748e-05
Epoch 131/400
875/875 [==============================] - 2s 2ms/step - loss: 3.2358e-05 - val_loss: 5.5725e-05
Epoch 132/400
875/875 [==============================] - 2s 2ms/step - loss: 2.6811e-05 - val_loss: 5.1572e-05
Epoch 133/400
875/875 [==============================] - 2s 2ms/step - loss: 2.6857e-05 - val_loss: 4.6538e-05
Epoch 134/400
875/875 [==============================] - 2s 2ms/step - loss: 2.6779e-05 - val_loss: 5.8036e-05
Epoch 135/400
875/875 [==============================] - 2s 2ms/step - loss: 2.5820e-05 - val_loss: 5.0039e-05
Epoch 00135: early stopping


def predict(x, threshold=threshold, window_size=82):
    x = scaler.transform(x)
    predictions = losses.mse(x, ae(x)) > threshold
    # Majority voting over `window_size` predictions
    return np.array([np.mean(predictions[i-window_size:i]) > 0.5
                     for i in range(window_size, len(predictions)+1)])

def print_stats(data, outcome):
    print(f"Shape of data: {data.shape}")
    print(f"Detected anomalies: {np.mean(outcome)*100}%")
    print()


test_data = [X_test0, X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, X_test9, X_test10]

with tf.device("/GPU:0"):
    for i, x in enumerate(test_data):
        print(i)
        outcome = predict(x)
        print_stats(x, outcome)



0
Shape of data: (9548, 115)
Detected anomalies: 0.0%

1
Shape of data: (107685, 115)
Detected anomalies: 100.0%

2
Shape of data: (102195, 115)
Detected anomalies: 100.0%

3
Shape of data: (122573, 115)
Detected anomalies: 100.0%

4
Shape of data: (237665, 115)
Detected anomalies: 100.0%

5
Shape of data: (81982, 115)
Detected anomalies: 100.0%

6
Shape of data: (105874, 115)
Detected anomalies: 0.0%

7
Shape of data: (29068, 115)
Detected anomalies: 100.0%

8
Shape of data: (29849, 115)
Detected anomalies: 100.0%

9
Shape of data: (92141, 115)
Detected anomalies: 0.0%

10
Shape of data: (59718, 115)
Detected anomalies: 100.0%




4. Conclusion
According to the above output, it seems to work well.
The following are some possibilities where to go from here:
* Run on the full N-BaIoT dataset: all 9 devices, all attacks.
* Implement (some of) the improvements from [2].
* In [3], it is suggested to use a subset of 23 features instead of all 115. However, different algorithms were used.
    * Question: Are these 23 features enough also for an autoencoder system like in [1] or [2]?
* Run on the MedBIoT dataset [4].
* Run on the IoT-23 dataset [5] after performing feature extraction.
References
[1] Meidan, Yair, et al. "N-BaIoT—Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders." IEEE Pervasive Computing 17.3 (2018): 12-22. https://arxiv.org/pdf/1805.03409 [2] Mirsky, Yisroel et al. "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS (2018). https://arxiv.org/abs/1802.09089v2 [3] Alhowaide, Alaa, et al. "Towards the design of real-time autonomous IoT NIDS." Cluster Computing (2021): 1-14. https://doi.org/10.1007/s10586-021-03231-5 [4] Guerra-Manzanares, Alejandro, et al. "MedBIoT: Generation of an IoT Botnet Dataset in a Medium-sized IoT Network." ICISSP 1 (2020): 207-218. https://doi.org/10.5220/0009187802070218 [5] Garcia, Sebastian et al. "IoT-23: A labeled dataset with malicious and benign IoT network traffic" (2020). (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4743746


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(test_data, true_labels, threshold, window_size=82):
    predictions = []
    adjusted_labels = []
    
    for i, x in enumerate(test_data):
        pred = predict(x, threshold, window_size)
        predictions.append(pred)
        
        # Adjust true_labels to match prediction length
        adjusted_labels.append(true_labels[i][window_size-1:])
    
    predictions = np.concatenate(predictions)
    adjusted_labels = np.concatenate(adjusted_labels)
    
    acc = accuracy_score(adjusted_labels, predictions)
    f1 = f1_score(adjusted_labels, predictions)
    precision = precision_score(adjusted_labels, predictions)
    recall = recall_score(adjusted_labels, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return acc, f1, precision, recall


# Create ground truth labels (0 for benign, 1 for attack)
with tf.device("/GPU:0"):
    true_labels = [
        np.zeros(len(X_test0)),  # Benign -> 0
        np.ones(len(X_test1)),   # Attack -> 1
        np.ones(len(X_test2)),
        np.ones(len(X_test3)),
        np.ones(len(X_test4)),
        np.ones(len(X_test5)),
        np.ones(len(X_test6)),
        np.ones(len(X_test7)),
        np.ones(len(X_test8)),
        np.ones(len(X_test9)),
        np.ones(len(X_test10))
    ]
    
    # Evaluate model
    evaluate_model(test_data, true_labels, threshold)
