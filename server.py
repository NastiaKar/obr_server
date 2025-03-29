# server.py — ORB з логами та зниженим порогом
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

reference_descriptors = None
reference_keypoints = None
orb = cv2.ORB_create(nfeatures=1000)


def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_descriptors, reference_keypoints
    data = request.get_json()
    base64_img = data.get('image')
    image = base64_to_image(base64_img)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    print(f"[REF] Keypoints: {len(keypoints)}, Descriptors: {descriptors.shape if descriptors is not None else None}")

    if descriptors is None:
        return jsonify({'error': 'No features found in reference image'}), 400

    reference_descriptors = descriptors
    reference_keypoints = keypoints
    return jsonify({'status': 'Reference descriptors stored'})


@app.route('/analyze', methods=['POST'])
def analyze():
    global reference_descriptors, reference_keypoints
    if reference_descriptors is None:
        return jsonify({'error': 'No reference uploaded'}), 400

    data = request.get_json()
    base64_img = data.get('image')
    image = base64_to_image(base64_img)

    kp2, des2 = orb.detectAndCompute(image, None)
    print(f"[FRAME] Keypoints: {len(kp2)}, Descriptors: {des2.shape if des2 is not None else None}")

    if des2 is None:
        return jsonify({'match': False, 'reason': 'No features in camera image'})

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(reference_descriptors, des2)

    # Відфільтрувати за відстанню
    good_matches = [m for m in matches if m.distance < 70] 
    print(f"[MATCHING] Good matches: {len(good_matches)}, Total: {len(matches)}")

    return jsonify({
        'match': len(good_matches) > 5,  # 🔽 з 15 до 5
        'good_matches': len(good_matches),
        'total_matches': len(matches)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
