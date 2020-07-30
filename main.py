#

import uproot
from sklearn.model_selection import train_test_split
from awkwardNN.awkwardNN import awkwardNN
from awkwardNN.preprocessRoot import get_events
from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def say_something(self):
        print("something")

class Top(Base):
    def say_something(self, x):
        print("{}".format(x))
        super().say_something()


if __name__ == "__main__":
    # tree1 = uproot.open("./data/test_qcd_1000.root")["Delphes"]
    # tree2 = uproot.open("./data/test_ttbar_1000.root")["Delphes"]
    # #fields = ["Jet*"]
    # fields = ['Particle.E', 'Particle.P[xyz]']
    # X1 = get_events(tree1, fields)
    # X2 = get_events(tree2, fields)
    # y1 = [1] * len(X1)
    # y2 = [0] * len(X2)
    # X = X1 + X2
    # y = y1 + y2
    # #X = torch.tensor(X, dtype=torch.float32)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # #model = awkwardNN(mode='rnn', max_iter=2, verbose=True, feature_size_fixed=True)
    # #model = awkwardNN(mode='rnn', max_iter=2, verbose=True, feature_size_fixed=False)
    # model = awkwardNN(mode='deepset', max_iter=2, verbose=True, feature_size_fixed=True)
    # #model = awkwardNN(mode='deepset', max_iter=2, verbose=True, feature_size_fixed=False)
    #
    # model.train(X_train, y_train)
    # model.test(X_test, y_test)

    x = Top()
    x.say_something("here you go")


