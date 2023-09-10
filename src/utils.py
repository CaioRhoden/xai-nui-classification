
import pickle
from itertools import combinations
import pandas as pd
import os

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
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../data/model_input"))
    file_x = os.path.join(data_dir, f"X_{polos[0]}.pkl")
    file_y = os.path.join(data_dir, f"y_{polos[0]}.pkl")
    
    X = pd.read_pickle(file_x).drop(['Polo'], axis=1)
    y = pd.read_pickle(file_y)

    
    if len(polos) > 1:
        for i in range(1,len(polos)):
            file_x = os.path.join(data_dir, f"X_{polos[i]}.pkl")
            file_y = os.path.join(data_dir, f"y_{polos[i]}.pkl")
            X = pd.concat([X,  pd.read_pickle(file_x).drop(['Polo'], axis=1)], ignore_index=True)
            y = pd.concat([y, pd.read_pickle(file_y)], ignore_index=True)
    
    return X, y

def save_combination(polo, list):
    with open(f'../../data/models/{polo}/combination.pkl', 'wb') as file:
        pickle.dump(list, file)
