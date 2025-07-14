from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error,mean_squared_error


def calculate_metrics(real_data,predicted_data):
    length_real = len(real_data)
    length_predict = len(predicted_data)
    if length_predict > length_real:
        predicted_data_sh = predicted_data.iloc[:length_real]
        real_data_sh = real_data
    elif length_real > length_predict:
        real_data_sh = real_data.iloc[:length_predict]
        predicted_data_sh = predicted_data
    else:
        real_data_sh = real_data
        predicted_data_sh = predicted_data

    #
    return {
            "R^2": r2_score(real_data_sh, predicted_data_sh),
            "MAE":mean_absolute_error(real_data_sh, predicted_data_sh),
            "MAPE":mean_absolute_percentage_error(real_data_sh, predicted_data_sh),
            "MSE":mean_squared_error(real_data_sh, predicted_data_sh)
            }
