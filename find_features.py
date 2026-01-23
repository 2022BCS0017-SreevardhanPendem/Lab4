# since the best performance model is random forest , in the model i took 6 top features as input ,but in api we will give all the features as input . hence this file finds the features used in model and takes only those from the api input.

import joblib

artifact = joblib.load("model/model.joblib")

selected_features = artifact["selected_features"]
print(selected_features)