from cnn import VGG19
from faceDetector import FaceDetector
import cv2
import numpy as np
from tensorflow.keras import backend as K
import pickle
import base64

class SSFER:
    def __init__(self):
        self.SIZE = 185
        self.EMOTIONS = ['angry', 'disgust', 'fear', \
            'happiness', 'sadness', 'surprise', 'neutral']
        self.PATH_NET_INPUT = "models/vgg.hdf5"
        self.PATH_MODEL_INPUT = "models/rf.model"
        self.net = VGG19()
        self.model = None
        self.faceDetector = FaceDetector(mtcnn=True)

        self.loading_net()
        self.loading_classifier()

    def loading_net(self):
        self.net = self.net.build_network((int(self.SIZE), int(self.SIZE), 3), len(self.EMOTIONS))
        self.net.load_weights(self.PATH_NET_INPUT)

    def loading_classifier(self):
        with open(self.PATH_MODEL_INPUT, 'rb') as model_bn:
            self.model, emotions_ls = pickle.load(model_bn)

    def detect_faces(self, img):
        return self.faceDetector.detectMTCNN(img)

    def crop_face(self, img, coord):
        return img[coord[0]:coord[2], coord[3]:coord[1]]

    def resize_image(self, img):
        return cv2.resize(img, (self.SIZE, self.SIZE))

    def prepare_to_net(self, img):
        img = np.array(img)
        return img.reshape(1, self.SIZE, self.SIZE, 3)

    def get_embbeding(self, img):
        last_output = K.function([self.net.layers[0].input], [self.net.layers[-2].output])
        return last_output([img])[0][0]

    def classify(self, embbeding):
        return np.around(self.model.predict_proba([embbeding]), decimals=2)[0]

    def predict(self, img):
        faces = []
        coordinates = self.detect_faces(img)
        img_rectangle = img
        for coord in coordinates:
            img_cropped = self.crop_face(img, coord)
            img_resize = self.resize_image(img_cropped)
            img_resize = self.prepare_to_net(img_resize)
            embedding = self.get_embbeding(img_resize)
            probabilities = self.classify(embedding)

            face = {}
            face["faceRectangle"] = {"X": coord[3], "width": coord[1] - coord[3],
                                  "Y": coord[0], "height": coord[2] - coord[0]}
            face["emotionsProbabilities"] = {
                                        self.EMOTIONS[0]: probabilities[0],
                                        self.EMOTIONS[1]: probabilities[1],
                                        self.EMOTIONS[2]: probabilities[2],
                                        self.EMOTIONS[3]: probabilities[3],
                                        self.EMOTIONS[4]: probabilities[4],
                                        self.EMOTIONS[5]: probabilities[5],
                                        self.EMOTIONS[6]: probabilities[6],
                                     }
            face["emotionMajority"] = self.EMOTIONS[np.argmax(probabilities)]
            faces.append(face)

            cv2.rectangle(img_rectangle,
                          (coord[3], coord[0]),
                          (coord[1], coord[2]),
                          (0,155,255),
                          2)

        ret = {}
        ret["total_faces"] = len(faces)
        ret["faces"] = faces

        retval, buffer = cv2.imencode('.jpg', img_rectangle)

        return ret, base64.b64encode(buffer)
