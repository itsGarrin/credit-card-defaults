import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(
    data: pd.DataFrame, threshold=10, removed_variables=None
) -> pd.DataFrame:
    """
    Calculate and remove variables with high Variance Inflation Factor (VIF) from a DataFrame.

    This function recursively calculates the VIF for each variable in the provided DataFrame.
    Variables with a VIF greater than the specified threshold are removed to reduce multicollinearity.
    The function prints the names of the removed variables and their VIF values once the recursion is complete.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the variables to be evaluated.
    threshold (float): The VIF threshold above which variables will be removed. Default is 10.
    removed_variables (list, optional): A list to collect the names and VIF values of removed variables during recursion. Default is None.

    Returns:
    pd.DataFrame: The DataFrame with variables having VIF greater than the threshold removed.
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
            print(f"{var}: {vif_value}")

    return data


def plot_roc_curve(y_true, y_pred):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a binary classification model.

    This function uses the roc_curve function from scikit-learn to calculate the True Positive Rate (TPR)
    and False Positive Rate (FPR) at different thresholds. It then plots the ROC curve and annotates the
    area under the curve (AUC) in the plot.

    Parameters:
    y_true (array-like): The true binary labels for the data.
    y_pred (array-like): The predicted probabilities for the positive class.

    Returns:
    None
    """

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
