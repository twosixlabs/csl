import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn
print("test")

features = pickle.load(open("../../data/texas_100_features.p", 'rb')).astype(np.float32)
labels = pickle.load(open("../../data/texas_100_labels.p", 'rb'))

ds = list(zip(features, labels))

texas_train = ds[:60000]
texas_test = ds[60000:]
print("test 2")

class Texas_Classifier(nn.Module):
    def __init__(self):
        super(Texas_Classifier, self).__init__()
        self.fc1 = nn.Linear(6169, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return torch.log_softmax(x,dim=1)

print("test 3")
    
    
#batch_sizes = [16, 32, 64]
batch_sizes = [64]
epsilons = [10000, 50000, 100000]
#epsilons = [0, 1, 100, 1000, 10000, 50000, 100000]
throw_outs = [False, 2, 5, 10]

for b in batch_sizes:
    for e in epsilons:
        for t in throw_outs:
            print(f"model: {e}, {t}, {b} begin")
            model = Texas_Classifier()
            info, mode = er.run_experiment(model,
                                               texas_train,
                                               texas_test,
                                               epsilon=e,
                                               alpha=2,
                                               epochs=20,
                                               add_noise=True,
                                               throw_out_threshold=t,
                                               batch_size=b,
                                               lf=torch.nn.NLLLoss,
                                               print_rate=1)
            pickle.dump(info, open(f"../../data/texas_{e}_{t}_{b}.b", 'wb'))
