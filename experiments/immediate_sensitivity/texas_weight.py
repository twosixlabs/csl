import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn
from sklearn.model_selection import train_test_split

print("test")

features = pickle.load(open("../../inputs/texas_100_features.p", 'rb')).astype(np.float32)
labels = pickle.load(open("../../inputs/texas_100_labels.p", 'rb'))

ds = list(zip(features, labels))

_, ds = train_test_split(ds, shuffle=True)

texas_train, texas_test = train_test_split(ds, test_size=.3, shuffle=True)
print(len(texas_train), len(texas_test))

print("test 2")

class Texas_Classifier(nn.Module):
    def __init__(self, w):
        super(Texas_Classifier, self).__init__()
        self.fc1 = nn.Linear(6169, w)
        self.fc2 = nn.Linear(w, w)
        self.fc3 = nn.Linear(w, 100)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return torch.log_softmax(x,dim=1)

    print("test 3")
    
    
epsilons = [1, 10, 100, 0] #, 10000, 50000, 100000]
throw_outs = [0]
widths = [128]

for w in widths:
    for e in epsilons:
        infos = []
        for t in range(1):
            print(f"model: {w}, {e}, {0}, {64} begin")
            model = Texas_Classifier(w)
            info, _ = er.weight_experiment(model,
                                            texas_train,
                                            texas_test,
                                            epsilon=e,
                                            alpha=2,
                                            epochs=20,
                                            add_noise=True,
                                            batch_size=64,
                                            lf=torch.nn.NLLLoss,
                                            print_rate=1)
            infos.append(info)
        pickle.dump(infos, open(f"../../data/texas/texas_we_{w}_{e}_{0}_{64}.b", 'wb'))
