from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # no shuffle for time series
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "model": model,
        "mse": mse,
        "y_test": y_test,              
        "predictions": predictions,
        "last_predicted": predictions[-1],
        "last_actual": y_test.iloc[-1]
    }
