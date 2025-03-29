# opencv_server/server.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Загружене зображення з Sender (тимчасово зберігаємо тут)
reference_image = None


def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_image
    data = request.get_json()
    base64_img = data.get('image')
    reference_image = base64_to_image(base64_img)
    return jsonify({'status': 'Reference image uploaded'})


@app.route('/analyze', methods=['POST'])
def analyze():
    global reference_image
    if reference_image is None:
        return jsonify({'error': 'No reference image uploaded'}), 400

    data = request.get_json()
    base64_img = data.get('image')
    frame = base64_to_image(base64_img)

    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(reference_image, None)
        kp2, des2 = sift.detectAndCompute(frame, None)

        if des1 is None or des2 is None:
            return jsonify({'match': False, 'reason': 'No descriptors found'})

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return jsonify({'match': len(good) > 10, 'matches': len(good)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
