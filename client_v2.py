import requests
import json
import base64
import cv2
from PIL import Image
import io
import numpy as np

addr = 'http://localhost:5000/'
test_url = addr

content_type = 'application/octet-stream'
headers = {'content-type': content_type}

with open("input/", "rb") as image_file:
    encoded = base64.b64encode(image_file.read())
    print(encoded.decode('utf-8'))

    # # send http request with image and receive response
    # response = requests.post(test_url, data=encoded.decode('utf-8'), headers=headers)
    response = requests.post(test_url, data=encoded.decode('utf-8'), headers=headers)

    # # decode response
    print(json.loads(response.text))
    # print(encoded)

    rec = encoded.decode('utf-8')

    img = base64.b64decode(rec)
    print(type(img))
    t = open('dec.png', 'wb')
    t.write(img)
    t.close()

    # image = Image.open(io.BytesIO(rec.encode('utf-8')))
    # print(type(image))
    # return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    # sbuf = io.BytesIO()
    # sbuf.write(img)
    # pimg = Image.open(sbuf)
    # tttt = cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2RGB)


    nparr = np.frombuffer(img, np.uint8)
    tttt = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imshow("window", tttt)
    # wait forever, if Q is pressed then close cv image window
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
