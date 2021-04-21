import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn
import autograd_hacks


from sklearn.model_selection import train_test_split
print("test")

features = pickle.load(open("../../inputs/texas_100_features.p", 'rb')).astype(np.float32)
labels = pickle.load(open("../../inputs/texas_100_labels.p", 'rb'))

ds = list(zip(features, labels))

_, ds = train_test_split(ds, shuffle=True)

texas_train, texas_test = train_test_split(ds, test_size=.3, shuffle=True)
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
    
    
batch_sizes = [64]
epsilons = [100000, 1000000]
clips  = [.1, .2, .4, .8, 1, 5]
widths = [128]#, 256, 512, 1024]


for w in widths:
    for e in epsilons:
        infos = []
        for t in range(5):
            print(f"model: {e}, {.2}, {w} begin")
            model = Texas_Classifier(w)
            info, _ = er.baseline_experiment(model,
                                               texas_train,
                                               texas_test,
                                               epsilon=e,
                                               alpha=2,
                                               epochs=20,
                                               add_noise=True,
					       C=.2,
                                               batch_size=64,
                                               lf=torch.nn.NLLLoss,
                                               print_rate=1)
        pickle.dump(infos, open(f"../../data/texas/texas_mb_{w}_{e}_{.2}_{64}.b", 'wb'))
