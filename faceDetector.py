class FaceDetector:
    def __init__(self, mtcnn=False):
        if mtcnn == True:
            from mtcnn.mtcnn import MTCNN
            self.mtcnn = MTCNN()

    def detectMTCNN(self, img):
        result = self.mtcnn.detect_faces(img)

        if result == []:
            return []
        bbs = []
        for rt in result:
            if rt["confidence"] < 0.9:
                continue

            bb = rt["box"]
            #bb[1], bb[2], bb[3], bb[0]
            bbs.append((bb[1],  bb[2] + bb[0], bb[3] + bb[1], bb[0]))

        return bbs
