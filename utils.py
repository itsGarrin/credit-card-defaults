import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(
    data: pd.DataFrame, threshold: float = 10, removed_variables: list = None
) -> pd.DataFrame:
    """
    Calculate and remove variables with high Variance Inflation Factor (VIF) from a DataFrame.

    This function recursively calculates the VIF for each variable in the provided DataFrame.
    Variables with a VIF greater than the specified threshold are removed to reduce multicollinearity.
    The function prints the names of the removed variables and their VIF values once the recursion is complete.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the variables to be evaluated.
    threshold : float, optional
        The VIF threshold above which variables will be removed. Default is 10.
    removed_variables : list, optional
        A list to collect the names and VIF values of removed variables during recursion. Default is None.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with variables having VIF greater than the threshold removed.
    """
    if removed_variables is None:
        removed_variables = []

    vif = pd.DataFrame()
    vif["variables"] = data.columns
    vif["VIF"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    vif = vif.sort_values(by="VIF", ascending=False)

    if vif["VIF"].max() > threshold:
        removed_variable = vif.iloc[0]
        removed_variables.append(
            (removed_variable["variables"], removed_variable["VIF"])
        )
        data = data.drop(removed_variable["variables"], axis=1)
        return calculate_vif(data, threshold, removed_variables)

    if removed_variables:
        print("Removed variables with high VIF:")
        for var, vif_value in removed_variables:
            print(f"{var}: {vif_value:.2f}")

    return data


def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot the ROC curve for a given set of true labels and predicted probabilities.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    model_name : str, optional
        The name of the model (used for the plot title). Default is "Model".

    Returns:
    -------
    None
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--"
    )  # Diagonal line for random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {model_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def evaluate_models(
    models: list, predictions_base: list, predictions_hyper: list, y_test: list
) -> pd.DataFrame:
    """
    Compare the performance of base models and models with hyperparameter tuning.

    Parameters:
    ----------
    models : list
        List of model names.
    predictions_base : list
        List of predicted values from the base models.
    predictions_hyper : list
        List of predicted values from hyperparameter-tuned models.
    y_test : array-like
        Ground truth (correct) labels for the test set.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with accuracy, F1 score, precision, and recall for both base and hyperparameter-tuned models.
    """
    def compute_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
        }

    metrics_base = [compute_metrics(y_test, y_pred) for y_pred in predictions_base]
    metrics_hyper = [
        compute_metrics(y_test, y_pred_hyper) for y_pred_hyper in predictions_hyper
    ]

    results = pd.DataFrame(
        {
            "Model": models,
            "Accuracy": [metrics["Accuracy"] for metrics in metrics_base],
            "Accuracy (Hyperparameter Tuning)": [
                metrics["Accuracy"] for metrics in metrics_hyper
            ],
            "Precision": [metrics["Precision"] for metrics in metrics_base],
            "Precision (Hyperparameter Tuning)": [
                metrics["Precision"] for metrics in metrics_hyper
            ],
            "Recall": [metrics["Recall"] for metrics in metrics_base],
            "Recall (Hyperparameter Tuning)": [
                metrics["Recall"] for metrics in metrics_hyper
            ],
            "F1 Score": [metrics["F1 Score"] for metrics in metrics_base],
            "F1 Score (Hyperparameter Tuning)": [
                metrics["F1 Score"] for metrics in metrics_hyper
            ],
        }
    )

    return results
