import gradio as gr
import cv2

import detect_people

def process_video(
		video_path: str,
		track_conf: float = 0.6,
		track_iou: float = 0.5,
		qscore_conf_weight: float = 0.65,
		qscore_box_weight: float = 0.1,
		qscore_center_weight: float = 0.25,
		top_k_faces: int = 3,
		dbscan_eps: float = 0.25
	):
	if not video_path: return "Please upload a video first", None

	print(f"Processing video: {video_path}")
	
	try:
		faces_cls_best = detect_people.find_people(video_path, track_conf, track_iou, qscore_conf_weight, qscore_box_weight, qscore_center_weight, top_k_faces, dbscan_eps) 
	except Exception as e:
		return f"An error occurred during face processing: {e}", None
	
	gallery_images = []
	
	for i, face_obj in enumerate(faces_cls_best):
		face_img = face_obj.get_face() 
		face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
		caption = f"Person {i + 1} (FaceScore: {face_obj.score:.2f})"
		gallery_images.append((face_img_rgb, caption))

	num_unique_people = len(faces_cls_best)
	if num_unique_people == 0:
		summary = "Analysis complete. No faces were confidently identified."
	elif num_unique_people == 1:
		summary = "Analysis complete. 1 unique individual was identified."
	else:
		summary = f"Analysis complete. {num_unique_people} unique individuals were identified."
	
	return summary, gallery_images

if __name__ == "__main__":
	with gr.Blocks(title="Clustered Face Identifier") as demo:
		gr.Markdown(
			"<center><span style='font-size: 36px; font-weight: bold;'>Clustered Face Identifier</span></center>"
		)

		with gr.Row(equal_height=True) as main_row:
			with gr.Column(scale=2):
				video_input = gr.Video(label="Upload a Video", sources=["upload"])
				
				submit_button = gr.Button("Identify People", variant='primary')
				
				count_output = gr.Markdown(value="Upload a video and click 'Identify People'",label="Analysis Summary")
				
				image_output = gr.Gallery(label="Best Representative Face Images (One per Unique Person)", show_label=True, rows=1, columns=5, object_fit="contain", height="auto")

			with gr.Column(scale=1):
				gr.Markdown("## Parameters")
				gr.Markdown("### Face Detection and Tracking")

				track_conf = gr.Slider(minimum=0, maximum=1, value=0.6, step=0.01, label="Confidence Threshold")
				track_iou = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="IOU Threshold")
				
				gr.Markdown("### Face Quality Scoring")

				qscore_conf_weight = gr.Slider(minimum=0, maximum=1, value=0.65, step=0.01, label="Confidence Weight")
				qscore_box_weight = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.01, label="Box Size Weight")
				qscore_center_weight = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.01, label="Center Distance Weight")
				
				gr.Markdown("### Embedding & Clustering")

				top_k_faces = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top K Faces")
				dbscan_eps = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.01, label="DBSCAN Epsilon")

		submit_button.click(
			fn=process_video,
			inputs=[video_input, track_conf, track_iou, qscore_conf_weight, qscore_box_weight, qscore_center_weight, top_k_faces, dbscan_eps],
			outputs=[count_output, image_output]
		)

	demo.launch()