from cnn import VGG19
from faceDetector import FaceDetector
import cv2
import numpy as np
from keras import backend as K
import pickle

def main():
    SIZE = 185
    EMOTIONS = ['angry', 'disgusted', 'fearful', \
            'happy', 'sad', 'surprised', 'neutral']

    EMOTIONS_pt = ['raiva', 'desgosto', 'medo', \
            'felicidade', 'tristeza', 'surpresa', 'neutralidade']

    PATH_NET_INPUT = "models/vgg.hdf5"
    PATH_MODEL_INPUT = "models/rf.model"

    INPUT_IMAGE = "input/anderson.jpg"

    net = VGG19()
    net = net.build_network((int(SIZE), int(SIZE), 3), len(EMOTIONS))
    net.load_weights(PATH_NET_INPUT)

    img = cv2.imread(INPUT_IMAGE)
    print(img.shape)

    # img_resize = cv2.resize(img, (SIZE, SIZE))

    faceDet = FaceDetector(mtcnn=True)
    coord = faceDet.detectMTCNN(img)
    if coord is not []:
        bb = coord[0]

        # cv2.rectangle(img_resize,
        #               (bb[3], bb[0]),
        #               (bb[1], bb[2]),
        #               (0,155,255),
        #               2)

        img_cropped = img[bb[0]:bb[2], bb[3]:bb[1]]

        img_resize = cv2.resize(img_cropped, (SIZE, SIZE))

        # cv2.imshow("window", img_resize)
        # # wait forever, if Q is pressed then close cv image window
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        img_resize = np.array(img_resize)
        img_resize = img_resize.reshape(1, SIZE, SIZE, 3)
        predictions = net.predict([np.array(img_resize)])

        index_emotion = np.argmax(predictions[0])
        print(np.around(predictions, decimals=2)[0])
        print('index: ', index_emotion)

        last_output = K.function([net.layers[0].input], [net.layers[-2].output])
        features = last_output([img_resize])
        print(features[0][0])

        with open(PATH_MODEL_INPUT, 'rb') as model_bn:
            model, emotions_ls = pickle.load(model_bn)
            print(emotions_ls)
            clazz = model.predict([features[0][0]])
            print("classe: ", clazz)
            probs = model.predict_proba([features[0][0]])
            print("probs: ", np.around(probs, decimals=2)[0])


if __name__ == '__main__':
    main()