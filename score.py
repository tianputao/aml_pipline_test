import json
import numpy
from azureml.core.model import Model
import joblib


#  global model
#     # AZUREML_MODEL_DIR is an environment variable created during deployment.
#     # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
#     # Please provide your model's folder name if there is one
#     model_path = os.path.join(
#         os.getenv("AZUREML_MODEL_DIR"), "sklearn_regression_model.pkl"



model_path = Model.get_model_path(model_name="sklearn_regression_model.pkl")
model = joblib.load(model_path)

raw_data = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'

data = json.loads(raw_data)["data"]
data = numpy.array(data)



request_headers = {}

result = model.predict(data)
print("Test result: ", {"result": result.tolist()})
