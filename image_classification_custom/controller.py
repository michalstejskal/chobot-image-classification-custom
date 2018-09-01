import io
import logging
from flask import Flask, jsonify, request
from PIL import Image
from config.config import logs_path

from network import load_model, predict, train_model

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_image():
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            data = predict(image)
            data["success"] = True
    return jsonify(data)


@app.route("/train", methods=["GET"])
def train():
    data = {"success": False}
    if request.method == "GET":
        train_model()
        data["success"] = True
    return jsonify(data)


def setup_logging():
    logging.basicConfig(filename=logs_path + '/image_classificator_core_custom.log', level=logging.INFO)


if __name__ == "__main__":
    setup_logging()
    load_model()
    app.run()

# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
