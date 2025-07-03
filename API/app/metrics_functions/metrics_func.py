from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error,mean_squared_error

def calculate_metrics(real_data,predicted_data):
    length_real = len(real_data)
    length_predict = len(predicted_data)
    if length_predict > length_real:
        predicted_data = predicted_data[:length_real]
    elif length_real > length_predict:
        real_data = real_data[:length_predict]

    return {"R^2":r2_score(real_data,predicted_data),
            "MAE":mean_absolute_error(real_data,predicted_data),
            "MAPE":mean_absolute_percentage_error(real_data,predicted_data),
            "MSE":mean_squared_error(real_data,predicted_data)
            }
