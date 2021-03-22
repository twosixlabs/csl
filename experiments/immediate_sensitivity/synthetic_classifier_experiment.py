import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn
print("test")

n_classes = 10

X, y = sklearn.datasets.make_classification(n_samples=1000,
                                            n_features=10,
                                            n_informative=5,
                                            n_redundant=2,
                                            n_repeated=0,
                                            class_sep=1.0,
                                            n_classes=n_classes)

n_features = X.shape[1]

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=7)


BATCH_SIZE = 64

training_dataset = TensorDataset(torch.from_numpy(X_train).float(), 
                                 torch.from_numpy(y_train).long())

testing_dataset = TensorDataset(torch.from_numpy(X_test).float(), 
                                torch.from_numpy(y_test).long())


class Classifier(nn.Module):
    def __init__(self, n_features, n_hidden=256):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.network(x)

print("test 3")
    
    
epsilons = [0 ,1 ,5, 10, 100, 1000, 10000, 50000, 100000]
#epsilons = [0, 1, 100, 1000, 10000, 50000, 100000]
throw_outs = [False, .03, .1, .2, .5, 1, 3]

for e in epsilons:
    for t in throw_outs:
        print(f"model: {e}, {t}, {b} begin")
        model = Texas_Classifier()
        info, mode = er.run_experiment(model,
                                               texas_train,
                                               texas_test,
                                               epsilon=e,
                                               alpha=2,
                                               epochs=200,
                                               add_noise=True,
                                               throw_out_threshold=t,
                                               batch_size=BATCH_SIZE,
                                               lf=torch.nn.NLLLoss,
                                               print_rate=10)
            pickle.dump(info, open(f"../../data/synth_{e}_{t}_{b}.p", 'wb'))