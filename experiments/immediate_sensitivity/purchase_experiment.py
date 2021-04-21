import numpy as np
import torch
import pickle
import experiment_runner as er
from sklearn.model_selection import train_test_split
from torch import nn
print("test")

X= pickle.load(open("../../inputs/purchase_100_features.p", 'rb')).astype(np.float32)
y = pickle.load(open("../../inputs/purchase_100_labels.p", 'rb'))

# I use one of these for the sake of sampling because lazy
(X_trash, X_real, y_trash, y_real) = train_test_split(X, y, test_size=0.05)
(X_train, X_test, y_train, y_test) = train_test_split(X_real, y_real, test_size=0.2)
print(len(X_train), len(y_test))
purchase_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), 
                                 torch.from_numpy(y_train).long())

purchase_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), 
                                torch.from_numpy(y_test).long())

print("test 2")

class Purchase_Classifier(nn.Module):
    def __init__(self, w):
        super(Purchase_Classifier, self).__init__()
        self.fc1 = nn.Linear(600, w)
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
    
    
epsilons = [1000, 10000, 50000, 100000]
throw_outs = [1.25, 2.5, 3]
batches = [64]
widths = [256]

for w in widths:
    for e in epsilons:
        for t in throw_outs:
            for b in batches:
                print(f"model: {w}, {e}, {t}, {b} begin")
                model = Purchase_Classifier(w)
                info, mode = er.run_experiment(model,
                                                purchase_train,
                                                purchase_test,
                                                epsilon=e,
                                                alpha=2,
                                                epochs=20,
                                                add_noise=True,
                                                throw_out_threshold=False,
                                                throw_out_std=t,
                                                batch_size=b,
                                                lf=torch.nn.NLLLoss,
                                                print_rate=1)
                pickle.dump(info, open(f"../../data/purchase/purchase_std_{w}_{e}_{t}_{b}.b", 'wb'))