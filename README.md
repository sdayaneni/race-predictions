# Race Predictions

This is my solution to a take home exercises given to applicants for a Dime Line Trading Intern role. The objective for the task was to create a model that minimizes Mean Squared Error (MSE). The full prompt for the exercise can be found in `problem.pdf`.

## Imports:
- `pip install tensorflow`
- `pip install xgboost`
- `pip install scikit-learn`
- `pip install numpy`
- `pip install pandas`

## Models
I've developed two different solutions for different approaches to the problem, which can be found in the `models` folder.

1. The first approach (`./models/submission`) uses a Gradient Boosting model, created with **XGBoost**. This approach resulted in an MSE of **0.1479**.
2. The second approach (`./models.submission2`) uses a neural network created with **TensorFlow** and **Keras** This approach resulted in an MSE of **0.1533**.

## Outputs
Predictions for each of the respective models can be found in `./predictions/mypred.csv` and `./predictions/mypred2.csv` respectively.
