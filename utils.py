import pandas as pd
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
