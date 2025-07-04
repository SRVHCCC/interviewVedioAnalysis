from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from main.videoAnalysis import analyze_video
import json

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    output_json = analyze_video(video_path)
    with open(output_json) as f:
        data = json.load(f)

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
