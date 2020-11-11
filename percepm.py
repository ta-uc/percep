import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
a = 30
x = []
for i in range(int(a/2)):
    x.append([random.randint(40,100),random.randint(1,80)])
for i in range(int(a/2)):
    x.append([random.randint(1,35),random.randint(25,100)])
train_data = np.asarray(x)
t_label = [0] * int(a / 2) + [1] * int(a / 2)

class Percep:

    def __init__(self,t_data,t_label,t_rate,inn):
        self.w = np.random.randn(inn)
        self.b = np.random.randn(1)
        self.t_rate = t_rate
        self.t_data = t_data
        self.t_label = t_label
    
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def __relu(self,x):
        return x if x > 0 else 0

    def __step(self,x):
        return 1 if x > 0 else 0

    def predict(self,x):
        t = np.dot(x,self.w.T) + self.b
        return self.__step(t)

    def train(self):
        for i in range(0,train_data.shape[0]):
            t = np.dot(train_data[i],self.w.T) + self.b
            out = self.__sigmoid(t)
            err = t_label[i] - out
            self.w = self.w + (self.t_rate * err * train_data[i])
            self.b = self.b + err
        return self.w,self.b
        
errs =[]
frames = []
fig = plt.figure()
net = Percep(train_data,t_label,0.01,2)
x = np.array(range(0,101))
green = train_data[0:14]
blue = train_data[15:]

for i in range(300):
    w,b = net.train()
    y = -(w[0] / w[1]) * x - b / w[1]
    g1 = plt.plot([green[i][0] for i in range(0,14)],[green[i][1] for i in range(0,14)],"g^")
    g2 = plt.plot([blue[i][0] for i in range(0,14)],[blue[i][1] for i in range(0,14)],"bs")
    l1 = plt.plot(x,y,color="black")
    frames.append(g1+g2+l1)

ani = animation.ArtistAnimation(fig, frames, interval=2)
plt.xlim([-10,110])
plt.ylim([-10,110])
plt.show()