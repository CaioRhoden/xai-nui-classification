import sys
import os
sys.path.append('../../../src')


import numpy as np
from utils import  generate_combinations, generate_dataset_split, save_combination
from itertools import combinations
import random


def random_data_generator(classification_type: str, n_samples: int):
    splits_collection = []
    for i in range(5):
        cities = ["Porto Alegre", "Marabá", "Brasília", "Belo Horizonte", "Juazeiro do Norte", "Recife"]

        splits = []
        test_cities = random.sample(range(0, 5), 2)
        selected_split = random.randint(0,6)
        splits.append(generate_combinations([cities[j] for j in range(len(cities)) if j not in test_cities], 2, 2))



        combination = splits[0][selected_split]

        X_train, y_train = generate_dataset_split(combination[0], classification_type)
        X_val, y_val = generate_dataset_split(combination[1], classification_type)
        X_test, y_test = generate_dataset_split([cities[j] for j in range(len(cities)) if j in test_cities], classification_type)

        splits_collection.append([X_train, y_train, X_val, y_val, X_test, y_test])
    
    return splits_collection