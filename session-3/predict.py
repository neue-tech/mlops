import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("models:/oncology_breast_classifier/Production")
sample = pd.DataFrame([{
    "mean_radius": 14.0, "mean_texture": 20.0, "mean_perimeter": 90.0, "mean_area": 600.0
}])
print("Prediction:", model.predict(sample))
