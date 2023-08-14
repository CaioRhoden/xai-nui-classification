
import pickle

def load_model(path):

    model = pickle.load(open(path, "rb"))
    return model

