import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
a = 1000
t_list = []
for i in range(int(a / 5)):
    t_list.append([random.randint(1, 28), random.randint(65, 90)])  # 0
for i in range(int(a / 5)):
    t_list.append([random.randint(10, 20), random.randint(3, 30)])  # 1
for i in range(int(a / 5)):
    t_list.append([random.randint(25, 55), random.randint(32, 59)])  # 2
for i in range(int(a / 5)):
    t_list.append([random.randint(70, 90), random.randint(85, 100)])  # 3
for i in range(int(a / 5)):
    t_list.append([random.randint(57, 99), random.randint(23, 49)])  # 4

train_data = np.asarray(t_list)
train_data = train_data / 100

t_label = np.asarray([[1, 0, 0, 0, 0]] * int(a / 5) + 
                     [[0, 1, 0, 0, 0]] * int(a / 5) +
                     [[0, 0, 1, 0, 0]] * int(a / 5) +
                     [[0, 0, 0, 1, 0]] * int(a / 5) + 
                     [[0, 0, 0, 0, 1]] * int(a / 5))


class OneLayerNet:
    def __init__(self, t_data, t_label, t_rate=0.01, hnn=0, onn=0):
        self.w1 = np.random.rand(t_data.shape[1], hnn)
        self.w2 = np.random.rand(hnn, onn)
        self.b1 = np.random.rand(1, hnn)
        self.b2 = np.random.rand(1, onn)
        self.t_rate = t_rate
        self.t_data = t_data
        self.t_label = t_label

    # activate function
    # def __sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def __sigmoid_b(self,h):
    #     pass

    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_b(self, delta, x_foward):
        mask = (x_foward <= 0)
        delta[mask] = 0
        delta_x = delta
        return delta_x

    def __softmax(self, x):
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

    def __softmax_b(self, x_foward, i):
        return x_foward - self.t_label[i]

    # err function
    def __mse(self, x, t):
        return 0.5 * np.sum((x - t) ** 2)

    def __cross(self, x, t):
        d = 1e-7
        return -np.sum(t * np.log(x + d))

    def __dotplus(self, x, w, b):
        return np.dot(x, w) + b

    def __dotplus_b(self, delta, x_foward, w_foward, b_foward):
        delta_x = np.dot(delta, w_foward.T)
        delta_w = np.dot(x_foward.T, delta)
        delta_b = delta
        return delta_x, delta_w, delta_b

    def predict(self, x):
        # input -> hidden
        self.t1 = self.__dotplus(x, self.w1, self.b1)
        self.l1 = self.__relu(self.t1)
        # hidden -> output
        out = self.__dotplus(self.l1, self.w2, self.b2)
        out = self.__softmax(out)
        return out

    def train(self):
        for i in range(0, int(a)):
            # forward
            # out = self.predict(self.t_data[i])
            # last = self.__softmax(out)

            last = self.predict(self.t_data[i])
            # label = np.expand_dims(self.t_label[i], axis=0)
            # err = self.__cross(last, label)

            # backward
            # output to hidden
            d_softmax = self.__softmax_b(last, i)
            d_dotplus_ho, delta_w, delta_b = self.__dotplus_b(
                d_softmax, self.l1, self.w2, self.b2)
            self.w2 -= self.t_rate * delta_w
            self.b2 -= self.t_rate * delta_b

            # hidden to input
            d_relu = self.__relu_b(d_dotplus_ho, self.t1)
            forward = self.t_data[i]
            forward = np.expand_dims(forward, axis=0)
            dummy, delta_w, delta_b = self.__dotplus_b(
                d_relu, forward, self.w1, self.b1)
            self.w1 -= self.t_rate * delta_w
            self.b1 -= self.t_rate * delta_b


net = OneLayerNet(train_data, t_label, 0.001, 7, 5)
print("trainig")
for j in range(200):
    net.train()

print("painting")
for i in range(0, 100, 2):
    for j in range(0, 100, 2):
        array = np.array([i, j]) / 100
        if np.argmax(net.predict(array)) == 0:
            style = "g."
        elif np.argmax(net.predict(array)) == 1:
            style = "b."
        elif np.argmax(net.predict(array)) == 2:
            style = "r."
        elif np.argmax(net.predict(array)) == 3:
            style = "c."
        elif np.argmax(net.predict(array)) == 4:
            style = "m."
        plt.plot(i, j, style)

plt.xlim([-10, 110])
plt.ylim([-10, 110])
plt.show()
