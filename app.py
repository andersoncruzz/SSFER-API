from ssfer import SSFER
import numpy as np
from flask import Flask, jsonify, request, abort
import cv2
import base64

app = Flask(__name__)
ssfer = SSFER()

@app.errorhandler(500)
def resource_not_found(e):
    return jsonify(error=str(e)), 500

@app.route('/status')
def get_status():
    return jsonify(status="SSFER is ok and running")

@app.route('/teste')
def get_test():
    img = cv2.imread("input/teste.png")
    probs = ssfer.predict(img)
    return jsonify(probs)

@app.route('/ssfer', methods=['POST'])
def get_predict():
    try:
        r = request

        if r.content_type in ["image/jpeg", "image/png"]:
            print("PNG or JPEG")
            # convert string of image data to uint8
            nparr = np.frombuffer(r.data, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        elif r.content_type == "application/octet-stream":
            print("base64")
            img = base64.b64decode(r.data)
            nparr = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        probs = ssfer.predict(img)
        return jsonify(probs)
    except:
        abort(500, description="Your requesting is not ok")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
