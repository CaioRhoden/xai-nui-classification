import sklearn.metrics as metrics
import matplotlib.pyplot as plt

class XGBoostEvaluator:

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


    def plot_roc_curve(self, y_test,plot_name):

        fpr, tpr, thresholds = metrics.roc_curve(y_test, self._y_pred_proba)
        metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=self.roc_auc_score, estimator_name=plot_name).plot()
        plt.show()

    
    
