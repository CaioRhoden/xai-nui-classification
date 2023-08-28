
import pandas as pd
import pickle



def train_model(model, X_train, y_train, evaluation):

    model.fit(X_train, y_train, eval_set=evaluation, early_stopping_rounds=10, verbose=False)
    return model

def save_model(model, name):

    pickle.dump(model, open(f"data/models/{name}.sav", 'wb'))


