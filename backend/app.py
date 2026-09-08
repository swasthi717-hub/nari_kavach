from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import uuid
import json

from privacy_mirror import scan_sensitive, apply_blur
from full_cloak import cloak_face  # importing your function, not rewriting it


app = Flask(__name__)
CORS(app)


# =========================================================
# HOME
# =========================================================

@app.route('/')
def home():
    return "Server is running!"


# =========================================================
# FACE CLOAKING
# Existing functionality - kept unchanged
# =========================================================

@app.route('/cloak', methods=['POST'])
def cloak_endpoint():

    if 'photo' not in request.files:
        return {"error": "No photo uploaded."}, 400

    file = request.files['photo']

    try:
        img = Image.open(file.stream).convert('RGB')

        result_img = cloak_face(img)

        if result_img is None:
            return {"error": "No face detected in the photo."}, 400

        img_bytes = io.BytesIO()

        result_img.save(
            img_bytes,
            format='JPEG',
            quality=95
        )

        img_bytes.seek(0)

        return send_file(
            img_bytes,
            mimetype='image/jpeg'
        )

    except Exception as e:

        print("Cloaking error:", e)

        return {
            "error": "Something went wrong while processing the image."
        }, 500


# =========================================================
# PRIVACY MIRROR - SCAN
# Detect sensitive information WITHOUT blurring it
# =========================================================

@app.route('/privacy-scan', methods=['POST'])
def privacy_scan_endpoint():

    if 'photo' not in request.files:
        return jsonify({
            "success": False,
            "error": "No photo uploaded."
        }), 400

    file = request.files['photo']

    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No photo selected."
        }), 400

    try:

        # Create uploads folder if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Give uploaded image a unique filename
        filename = f"{uuid.uuid4().hex}_{file.filename}"

        image_path = os.path.join(
            "uploads",
            filename
        )

        # Save image temporarily
        file.save(image_path)

        # Run Privacy Mirror scan
        detections = scan_sensitive(image_path)

        # Return detection information to frontend
        return jsonify({
            "success": True,
            "detections": detections
        })

    except Exception as e:

        print("Privacy Mirror scan error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# PRIVACY MIRROR - APPLY BLUR
# Runs ONLY after user gives consent
# =========================================================

@app.route('/privacy-blur', methods=['POST'])
def privacy_blur_endpoint():

    if 'photo' not in request.files:
        return jsonify({
            "success": False,
            "error": "No photo uploaded."
        }), 400

    file = request.files['photo']

    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No photo selected."
        }), 400

    try:

        os.makedirs("uploads", exist_ok=True)

        # -------------------------------------------------
        # Get detections selected by the user
        # -------------------------------------------------

        detections_json = request.form.get(
            'detections',
            '[]'
        )

        try:
            detections = json.loads(
                detections_json
            )
        except json.JSONDecodeError:

            return jsonify({
                "success": False,
                "error": "Invalid detection data."
            }), 400

        # -------------------------------------------------
        # Save original image temporarily
        # -------------------------------------------------

        input_filename = (
            f"{uuid.uuid4().hex}_{file.filename}"
        )

        output_filename = (
            f"blurred_{uuid.uuid4().hex}.jpg"
        )

        input_path = os.path.join(
            "uploads",
            input_filename
        )

        output_path = os.path.join(
            "uploads",
            output_filename
        )

        file.save(input_path)

        # -------------------------------------------------
        # Apply blur ONLY to user-selected detections
        # -------------------------------------------------

        apply_blur(
            input_path,
            detections,
            output_path
        )

        # -------------------------------------------------
        # Return blurred image
        # -------------------------------------------------

        return send_file(
            output_path,
            mimetype='image/jpeg'
        )

    except Exception as e:

        print("Privacy Mirror blur error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == '__main__':

    app.run(
        debug=True,
        port=5000
    )