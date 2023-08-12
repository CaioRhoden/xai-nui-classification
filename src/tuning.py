from sklearn.model_selection import RandomizedSearchCV

def random_search_tuning(model, parameters, random_state=42):

    new_model = RandomizedSearchCV(model, parameters, random_state=random_state)

    return new_model