from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace().project("rc-cars-xlkrl")
model = project.version('1').model

path = "./rc_car_dataset/images/bootcamp_0076.jpg"
# infer on a local image
# print(model.predict(path, confidence=40, overlap=30).json())

# visualize your prediction
model.predict(path, confidence=20, overlap=30).save("prediction_0076.jpg")

