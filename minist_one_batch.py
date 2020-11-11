import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import pickle

with open('./mnist_data/data.pkl', 'rb') as f:
    dataset = pickle.load(f)

class OneLayerNet:
    def __init__(self, t_data, t_label, t_rate=0.01, hnn=0, onn=0, act="relu", act_l="softmax"):
        self.w1 = np.random.randn(t_data.shape[1], hnn) * 0.01
        self.w2 = np.random.randn(hnn, onn) * 0.01
        self.b1 = np.random.randn(1, hnn)
        self.b2 = np.random.randn(1, onn)
        self.t_rate = t_rate
        self.t_data = t_data
        self.t_label = t_label
        self.err = []

        if act == "relu":
            self.act = self.__relu
            self.act_b = self.__relu_b
        elif act == "sigmoid":
            self.act = self.__sigmoid
            self.act_b = self.__sigmoid_b

        if act_l == "relu":
            self.act_l = self.__relu
        elif act_l == "sigmoid":
            self.act_l = self.__sigmoid
        else:
            self.act_l = self.__softmax

    def update_data(self,t_data,t_label):
        self.t_data = t_data
        self.t_label = t_label

    ### activate function ###
    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -34.538776394910684,34.538776394910684)))

    def __sigmoid_b(self, delta, dummy, x_forward):
        return x_forward * (1.0 - x_forward) * delta

    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_b(self, delta, x_foward, dummy):
        mask = (x_foward <= 0)
        delta[mask] = 0
        delta_x = delta
        return delta_x

    def __softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

    def __cross_b(self, x_foward, t):
        return (x_foward - t) / t.shape[0]

    ### Error function ###
    def __cross(self, x, t):
        if x.ndim == 1:
            t = t.reshape(1,t.size)
            x = x.reshape(1,x.size)
        batch_size = x.shape[0]
        return -np.sum(t * np.log(x)) / batch_size
    
    def __dotplus(self, x, w, b):
        return np.dot(x, w) + b

    def __dotplus_b(self, delta, x_foward, w_foward, b_foward):
        delta_x = np.dot(delta, w_foward.T)
        delta_w = np.dot(x_foward.T, delta)
        delta_b = np.sum(delta, axis=0)
        return delta_x, delta_w, delta_b

    def predict(self, x):
        # input -> hidden
        self.t1 = self.__dotplus(x, self.w1, self.b1)
        self.l1 = self.act(self.t1)

        # hidden -> output
        out = self.__dotplus(self.l1, self.w2, self.b2)
        out = self.act_l(out)
        return out

    def __backward(self, x, t):
        # output -> hidden
        d_cross = self.__cross_b(x, t)
        d_dotplus_ho, delta_w, delta_b = self.__dotplus_b(
            d_cross, self.l1, self.w2, self.b2)
        self.w2 -= self.t_rate * delta_w
        self.b2 -= self.t_rate * delta_b

        # hidden -> input
        d_act = self.act_b(d_dotplus_ho, self.t1, self.l1)
        forward = self.t_data
        dummy, delta_w, delta_b = self.__dotplus_b(
            d_act, forward, self.w1, self.b1)
        self.w1 -= self.t_rate * delta_w
        self.b1 -= self.t_rate * delta_b

    def train(self):
        # Forward
        last = self.predict(self.t_data)

        # Get error
        self.err.append(self.__cross(last, self.t_label))

        # Backward
        self.__backward(last, self.t_label)

print("Training")
batch_size = 100
itr = 10000

t_index = np.random.choice(60000, batch_size)
train_data = dataset["train_data"][t_index]
train_label = dataset["train_label_one_hot"][t_index]
train_data = train_data / 255

net = OneLayerNet(train_data, train_label, 0.1, 50, 10, "relu", "softmax")

for j in range(itr):

    net.train()

    t_index = np.random.choice(60000, batch_size)
    train_data = dataset["train_data"][t_index]
    train_label = dataset["train_label_one_hot"][t_index]
    train_data = train_data / 255
    net.update_data(train_data, train_label)
    if j % 100 == 0:
        print("{0}%done".format(j / (itr / 100)))

### Test ###
test_size = 10000

print("Testing")
test_data = dataset["test_data"]
test_label = dataset["test_label"]
test_data = test_data / 255
correct = 0
for i in range(0, test_size):
    if np.argmax(net.predict(test_data[i])) == test_label[i]:
        correct += 1
print("Accuracy: {0}%".format((correct / test_size) * 100))

plt.plot([i for i in range(itr)],net.err,"black")
plt.show()