import sys
sys.path.append('../../src')
import pickle
import optuna
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from models.base_model import BaseModel

class LightGBModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.threshold = kwargs.get('threshold', 0.5)
        self.config = kwargs.get("parameters", None)
        if self.config is not None:
            self.model = LGBMClassifier(**self.config)
        else:
            self.model = LGBMClassifier()

    def load_model(self, path: str) -> None:
        # Open the model file in read binary mode
        with open(path, "rb") as file:
            # Load the model from the file
            self.model = pickle.load(file)
            self.config = self.model.best_params_
        
        return self.model
        
    
    def fit(self, X_train, y_train, evaluation):
        return self.fit(X_train, y_train, eval_set=evaluation, verbose=False)
            
    


    def _objective(self, trial, y_pred_proba, y_true):

        threshold = trial.suggest_float("threshold", 0.0, 1.0)

        # Apply the threshold to your model's predictions
        y_pred_thresholded = (y_pred_proba >= threshold).astype(int)

        # Calculate F1 Score
        score = f1_score(y_true, y_pred_thresholded, average='weighted')

        return score
    
    def find_best_threshold(self, X, y):
        study = optuna.create_study(direction="maximize")
        y_pred_proba = self.model.predict_proba(X)[::,1]
        study.optimize(lambda trial: self._objective(trial, y_pred_proba, y), n_trials=50)
        best_threshold = study.best_trial.params['threshold']
        self.threshold = best_threshold
        return best_threshold
        
    def predict(self, input_data):

        if self.threshold == 0.5:
            return super().predict(input_data)

        else:
            y_pred_proba = self.model.predict_proba(input_data)[::,1]
            y_pred_thresholded = (y_pred_proba >= self.threshold).astype(int)
            return y_pred_thresholded

        




    
    