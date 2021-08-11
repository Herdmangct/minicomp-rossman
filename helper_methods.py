import numpy as np

def print_percentage_of_original_data(current_data):
    original_data_rows = 618473
    percentage_of_original_data = round((current_data.shape[0] / original_data_rows) * 100, 2)
    print(f"Percentage of original data is {percentage_of_original_data}%")

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def pretty_metric(predictions, actuals, model):
    prediction = metric(predictions, actuals)
    print(f"The prediction for {model} is: {round(prediction, 2)}%")
