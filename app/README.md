# Clustered Face Identifier
This app identifies the number of unique faces in a video with YOLOv11 and DBSCAN-clustered Facenet512 embedding. It utilizes Gradio for deployment.

## Usage
1. Run `python app/app.py` from the root directory.
2. Open the local URL shown in the console on your web browser (http://127.0.0.1:7860).
3. In the app, upload a video file in the provided box.
4. Click "Identify People" to show the images and amount of unique faces in the uploaded video.

## Notes
- The app may take a longer time to show the results.
- You can tweak the parameters provided inside the app to enhance the identification quality.