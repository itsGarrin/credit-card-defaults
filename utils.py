import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(data: pd.DataFrame, threshold=5):
    """
    Calculate Variance Inflation Factor (VIF) for each variable in the data.
    :param data:  Pandas DataFrame
    :param threshold:  float
    :return:  Pandas DataFrame
    """
    vif = pd.DataFrame()
    vif["variables"] = data.columns
    vif["VIF"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    vif = vif.sort_values(by="VIF", ascending=False)
    if vif["VIF"].max() > threshold:
        data = data.drop(vif["variables"].values[0], axis=1)
        return calculate_vif(data)
    return vif
