from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn


MODEL = tf.keras.models.load_model("../models/my_model.keras")
class_names = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever',
               'Poodle', 'Rottweiler', 'Yorkshire_Terrier']


app = FastAPI()


@app.get("/ping")
def ping():
    return {"message": "Hello World, Rushi Here"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        prediction = MODEL.predict(img_batch)
        class_name = class_names[np.argmax(prediction)]
        confidence = 100 * np.max(prediction[0])
        return {"class_name": class_name, "confidence": float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
