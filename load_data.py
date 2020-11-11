import gzip
import numpy as np
import pickle


with gzip.open('./mnist_data/train-images-idx3-ubyte.gz', 'rb') as f1:
    train_data = np.frombuffer(f1.read(), np.uint8, offset=16)
    train_data = train_data.reshape(-1, 784)


with gzip.open('./mnist_data/train-labels-idx1-ubyte.gz', 'rb') as f2:
    train_label = np.frombuffer(f2.read(), np.uint8, offset=8)


with gzip.open('./mnist_data/t10k-images-idx3-ubyte.gz', 'rb') as f3:
    test_data = np.frombuffer(f3.read(), np.uint8, offset=16)
    test_data = test_data.reshape(-1, 784)


with gzip.open('./mnist_data/t10k-labels-idx1-ubyte.gz', 'rb') as f4:
    test_label = np.frombuffer(f4.read(), np.uint8, offset=8)


data = {}
data["train_data"] = train_data
data["train_label"] = train_label
data["test_data"] = test_data
data["test_label"] = test_label

def to_one_hot(label):
    T = np.zeros((label.size, 10))
    for i in range(label.size):
        T[i][label[i]] = 1
    return T

data['train_label_one_hot'] = to_one_hot(data['train_label'])
data['test_label_one_hot'] = to_one_hot(data['test_label'])

with open('./mnist_data/data.pkl', 'wb') as f5:
    pickle.dump(data, f5, -1)
