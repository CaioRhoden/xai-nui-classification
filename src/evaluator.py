import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Evaluator:
    
    def __init__(self, model) -> None:
        self._model = model
        self.acc = None
        self.f1_score = None
        self.roc_auc_score = None
        self.recall = None
        self.precision = None
        self._y_pred = None
        self._y_pred_proba = None
    
    def print_metrics(self):
        """
        Print the metrics of the model.
        This function prints the accuracy, F1 score, AUC score, recall, and precision of the model.
        Parameters:
            self: The object instance.
        Returns:
            None
        """
        
        print(f"Accuracy: { self.acc}")
        print(f"F1 Score: {self.f1_score}")
        print(f"ROC AUC Score: {self.roc_auc_score}")
        print(f"Recall: {self.recall}")
        print(f"Precision: {self.precision}")
    

    def evaluate(self, X_test, y_test):
        pass

    def plot_confusion_matrix(self, y_test):
        pass


    def plot_roc_curve(self, y_test,plot_name):

        fpr, tpr, thresholds = metrics.roc_curve(y_test, self._y_pred_proba)
        metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=self.roc_auc_score, estimator_name=plot_name).plot()
        plt.show()
    
    def generate_csv_line(self, type, train, val, test):
        pass
    
    


class XGBoostEvaluator(Evaluator):

    def __init__(self, model) -> None:
        super().__init__(model)

    def evaluate(self, X_test, y_test):
        """
    	Evaluates the performance of the model on the given test data.

    	Parameters:
    	- X_test: The input features for the test data. Shape: (n_samples, n_features)
    	- y_test: The true labels for the test data. Shape: (n_samples,)

    	Returns:
    	None
    	"""

        self._y_pred = self._model.predict(X_test)
        self._y_pred_proba = self._model.predict_proba(X_test)[:, 1]
        
        #Acc
        self.acc = self._model.model.score(X_test, y_test)

        #F1 Score
        self.f1_score = metrics.f1_score(y_test, self._y_pred)

        #Auc Score
        self.roc_auc_score = metrics.roc_auc_score(y_test, self._y_pred_proba)

        #Recall
        self.recall = metrics.recall_score(y_test, self._y_pred)

        #Precision
        self.precision = metrics.precision_score(y_test, self._y_pred)

        self.print_metrics()
    
    def plot_confusion_matrix(self, y_test):
        """
        Plot a confusion matrix
        Parameters:
            y_test (array-like): The true labels of the test data.
        Returns:
            None
        """

        cm = metrics.confusion_matrix(y_test, self._y_pred, labels=self._model.model.classes_)
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._model.model.classes_).plot()
        plt.show()

    def generate_csv_line(self, type, train, val, test):
        
        cities = ['Porto Alegre', 'Marabá', 'Brasília', 'Juazeiro do Norte', 'Recife', 'Belo Horizonte']

        values_cities = {}
        for c in cities:
            if c in train:
                values_cities[c] = 'Treino'
            elif c in val:
                values_cities[c] = 'Validação'
            else:
                values_cities[c] = 'Teste'
        
        
        basis = {
            'type': [type],
            'porto_alegre': [values_cities['Porto Alegre']],
            'brasilia': [values_cities['Brasília']],
            'maraba': [values_cities['Marabá']],
            'juazeiro_do_norte': [values_cities['Juazeiro do Norte']],
            'recife': [values_cities['Recife']],
            'belo_horizonte': [values_cities['Belo Horizonte']],
            'auc_score': [self.roc_auc_score],
            'f1_score': [self.f1_score],
            'acc': [self.acc],
            'recall': [self.recall],
            'precision': [self.precision]
        }

        df = pd.DataFrame(basis)
        return df



class EnsambleEvaluator(Evaluator):

    def __init__(self, model) -> None:
        super().__init__(model)

    def evaluate(self, X_test, y_test):
        """
    	Evaluates the performance of the model on the given test data.

    	Parameters:
    	- X_test: The input features for the test data. Shape: (n_samples, n_features)
    	- y_test: The true labels for the test data. Shape: (n_samples,)

    	Returns:
    	None
    	"""

        self._y_pred = self._model.predict(X_test)
        self._y_pred_proba = self._model.predict_proba(X_test)[:, 1]
        
        #Acc
        self.acc = self._model.score(X_test, y_test)

        #F1 Score
        self.f1_score = metrics.f1_score(y_test, self._y_pred)

        #Auc Score
        self.roc_auc_score = metrics.roc_auc_score(y_test, self._y_pred_proba)

        #Recall
        self.recall = metrics.recall_score(y_test, self._y_pred)

        #Precision
        self.precision = metrics.precision_score(y_test, self._y_pred)

        self.print_metrics()

    def plot_confusion_matrix(self, y_test):
        """
        Plot a confusion matrix
        Parameters:
            y_test (array-like): The true labels of the test data.
        Returns:
            None
        """

        cm = metrics.confusion_matrix(y_test, self._y_pred, labels=self._model.classes_)
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._model.classes_).plot()
        plt.show()
    
    def generate_csv_line(self, type, test):
        
        cities = ['Porto Alegre', 'Marabá', 'Brasília', 'Juazeiro do Norte', 'Recife', 'Belo Horizonte']

        values_cities = {}
        for c in cities:
            if c in test:
                values_cities[c] = 'Teste'
            else:
                values_cities[c] = 'Treino'
        
        
        basis = {
            'type': [type],
            'porto_alegre': [values_cities['Porto Alegre']],
            'brasilia': [values_cities['Brasília']],
            'maraba': [values_cities['Marabá']],
            'juazeiro_do_norte': [values_cities['Juazeiro do Norte']],
            'recife': [values_cities['Recife']],
            'belo_horizonte': [values_cities['Belo Horizonte']],
            'auc_score': [self.roc_auc_score],
            'f1_score': [self.f1_score],
            'acc': [self.acc],
            'recall': [self.recall],
            'precision': [self.precision]
        }

        df = pd.DataFrame(basis)
        return df

class MultiClassEvaluator(Evaluator):

    def __init__(self, model) -> None:
        super().__init__(model)
        self.roc_auc_class_1 = None
        self.roc_auc_class_2 = None
        self.roc_auc_score = None
        self.f1_score = None

        self.y_pred = None



    def _change_labels(self, df, target_class) -> np.ndarray:
        new_df = np.zeros_like(df)
        new_df[df == target_class] = 1
        return new_df
        
    def evaluate(self, X_test, y_test):


        self._y_pred = self._model.predict(X_test)
        self._y_pred_proba = self._model.predict_proba(X_test)

        #F1 Score
        self.f1_score = metrics.f1_score(y_test, self._y_pred, average='macro')

        #Auc Score
        self.roc_auc_score = metrics.roc_auc_score(y_test, self._y_pred_proba, multi_class="ovr")

        #Auc Score 1
        self.roc_auc_class_1 = metrics.roc_auc_score(self._change_labels(y_test, 1), self._change_labels(self._y_pred, 1))


        #Auc Score 2
        self.roc_auc_class_2 = metrics.roc_auc_score(self._change_labels(y_test, 2), self._change_labels(self._y_pred, 2))


        print(f"F1 Score: {self.f1_score}")
        print(f"ROC AUC Score: {self.roc_auc_score}")
        print(f"ROC AUC Score Class 1: {self.roc_auc_class_1}")
        print(f"ROC AUC Score Class 2: {self.roc_auc_class_2}")
        
    def generate_csv_line(self, type, test):
        
        cities = ['Porto Alegre', 'Marabá', 'Brasília', 'Juazeiro do Norte', 'Recife', 'Belo Horizonte']

        values_cities = {}
        for c in cities:
            if c in test:
                values_cities[c] = 'Teste'
            else:
                values_cities[c] = 'Treino'
        
        
        basis = {
            'type': [type],
            'porto_alegre': [values_cities['Porto Alegre']],
            'brasilia': [values_cities['Brasília']],
            'maraba': [values_cities['Marabá']],
            'juazeiro_do_norte': [values_cities['Juazeiro do Norte']],
            'recife': [values_cities['Recife']],
            'belo_horizonte': [values_cities['Belo Horizonte']],
            'auc_score': [self.roc_auc_score],
            'f1_score': [self.f1_score],
            'roc_auc_class_1': [self.roc_auc_class_1],
            'roc_auc_class_2': [self.roc_auc_class_2]
        }

        df = pd.DataFrame(basis)
        return df

    


