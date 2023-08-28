
import pickle
from itertools import combinations, permutations
import pandas as pd

def load_model(path):

    model = pickle.load(open(path, "rb"))
    return model

def generate_combinations(polos, len_1, len_2):
    valid_combinations = []

    for group1 in combinations(polos, len_1):
        remaining_elements = [e for e in polos if e not in group1]
        for group2 in combinations(remaining_elements, len_2):
            valid_combinations.append([list(group1), list(group2)])

    return valid_combinations


def generate_dataset_split(polos):
    print(polos)
    
    X = pd.read_pickle(f"../../data/model_input/X_{polos[0]}.pkl").drop(['Polo'], axis=1)
    y = pd.read_pickle(f"../../data/model_input/y_{polos[0]}.pkl")

    
    if len(polos) > 1:
        for i in range(1,len(polos)):
            X = pd.concat([X,  pd.read_pickle(f"../../data/model_input/X_{polos[i]}.pkl").drop(['Polo'], axis=1)], ignore_index=True)
            y = pd.concat([y, pd.read_pickle(f"../../data/model_input/y_{polos[i]}.pkl")], ignore_index=True)
    
    return X, y

def save_combination(polo, list):
    with open(f'../../data/models/{polo}/combination.pkl', 'wb') as file:
        pickle.dump(list, file)
