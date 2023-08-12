
import pandas as pd


def train(model, mode, polo = None):

    if mode == "multi":

        X_train = pd.read_pickle(f"../data/model_input/X_train_{polo}.pkl").drop(['Polo'], axis=1)
        y_train = pd.read_pickle(f"../data/model_input/y_train_{polo}.pkl")

        model.fit(X_train, y_train)



    elif mode == "single":

        X_train = pd.read_pickle("data/model_input/X_train.pkl")
        y_train = pd.read_pickle("data/model_input/y_train.pkl")

        model.fit(X_train, y_train)
    
    else:
        pass

    return model




