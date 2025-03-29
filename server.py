from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

reference_descriptors = None
reference_keypoints = None
reference_image = None

orb = cv2.ORB_create(nfeatures=2000)

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_descriptors, reference_keypoints, reference_image
    data = request.get_json()
    base64_img = data.get('image')
    image = base64_to_image(base64_img)

    image = cv2.resize(image, (640, 480)) 

    keypoints, descriptors = orb.detectAndCompute(image, None)
    print(f"[REF] Keypoints: {len(keypoints)}, Descriptors: {descriptors.shape if descriptors is not None else None}")

    if descriptors is None:
        return jsonify({'error': 'No features found in reference image'}), 400

    reference_image = image
    reference_descriptors = descriptors
    reference_keypoints = keypoints
    return jsonify({'status': 'Reference descriptors stored'})

@app.route('/analyze', methods=['POST'])
def analyze():
    global reference_descriptors, reference_keypoints, reference_image
    if reference_descriptors is None:
        return jsonify({'error': 'No reference uploaded'}), 400

    try:
        data = request.get_json()
        base64_img = data.get('image')
        image = base64_to_image(base64_img)
        image = cv2.resize(image, (640, 480))

        kp2, des2 = orb.detectAndCompute(image, None)
        print(f"[FRAME] Keypoints: {len(kp2)}, Descriptors: {des2.shape if des2 is not None else None}")

        if des2 is None:
            return jsonify({'match': False, 'reason': 'No features in camera image'})

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(reference_descriptors, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        distances = [m.distance for m in matches]
        if not distances:
            return jsonify({'match': False, 'reason': 'No matches found'})

        median_distance = np.median(distances)
        threshold = median_distance * 1.2
        good_matches = [m for m in matches if m.distance < threshold]

        print(f"[MATCHING] Good matches: {len(good_matches)}, Median dist: {median_distance:.2f}, Threshold: {threshold:.2f}")

        if len(good_matches) < 20:
            return jsonify({'match': False, 'reason': 'Not enough good matches', 'good_matches': len(good_matches)})

        src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        print(f"[HOMOGRAPHY] Inliers: {inliers} / {len(good_matches)}")

        is_match = inliers > 15 and (inliers / len(good_matches)) > 0.5

        match_visual_base64 = None
        if mask is not None:
            match_img = cv2.drawMatches(
                reference_image, reference_keypoints,
                image, kp2,
                good_matches, None,
                matchesMask=mask.ravel().tolist(),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            match_visual_base64 = image_to_base64(match_img)

        return jsonify({
            'match': is_match,
            'good_matches': len(good_matches),
            'total_matches': len(matches),
            'inliers': inliers,
            'threshold': threshold,
            'median_distance': median_distance,
            'match_visual': match_visual_base64
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
