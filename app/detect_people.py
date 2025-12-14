from deepface import DeepFace
import dlib
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from tqdm import tqdm
from ultralytics import YOLO

DETECT_MODEL = YOLO("./outputs/models/yolo-detect.pt")
SHAPE_PREDICTOR = dlib.shape_predictor("./outputs/models/shape_predictor_5_face_landmarks.dat")

class Face:
	def __init__(self, orig_img: np.ndarray, box: np.ndarray, score: float):
		self.orig_img: np.ndarray = orig_img
		self.box: np.ndarray = box
		self.score: float = score
		self.face_img: np.ndarray = None
		self.embedding: np.ndarray = None
	
	def get_face(self) -> np.ndarray:
		if self.face_img is not None: return self.face_img
		x1, y1, x2, y2 = self.box
		shape = SHAPE_PREDICTOR(self.orig_img, dlib.rectangle(x1, y1, x2, y2))
		self.face_img = dlib.get_face_chip(self.orig_img, shape, size=160)
		return self.face_img
	
	def get_embedding(self) -> np.ndarray:
		if self.embedding is not None: return self.embedding
		self.embedding = DeepFace.represent(self.get_face(), model_name="Facenet512", enforce_detection=False, detector_backend="skip", align=False)[0]["embedding"]
		return self.embedding

def face_score(face_conf: float, orig_shape: tuple[int], box: tuple[int], conf_weight: float = 0.65, box_weight: float = 0.1, center_weight: float = 0.25) -> float:	
	# Face Confidence
	score_conf = conf_weight * face_conf

	# Box Size
	x1, y1, x2, y2 = box
	box_area = (y2 - y1) * (x2 - x1)
	frame_area = orig_shape[0] * orig_shape[1]
	score_boxsize =  box_weight * (box_area / frame_area)

	# Distance to Center
	center_box = (y1 + (y2 - y1) / 2, x1 + (x2 - x1) / 2)
	center_frame = np.array(orig_shape) / 2
	edge_dist = np.array((min(center_box[0], orig_shape[0] - center_box[0]), min(center_box[1], orig_shape[1] - center_box[1])))
	score_center = center_weight * np.min((edge_dist / center_frame) ** 2)

	score = (score_conf + score_boxsize + score_center) / (conf_weight + box_weight + center_weight)
	return float(score)

def find_people(
		video: str | np.ndarray,
		track_conf: float = 0.6,
		track_iou: float = 0.5,
		qscore_conf_weight: float = 0.65,
		qscore_box_weight: float = 0.1,
		qscore_center_weight: float = 0.25,
		top_k_faces: int = 3,
		dbscan_eps: float = 0.25
	):
	# Detect and track all faces in a video
	print("Tracking faces...")
	tracked_frames = DETECT_MODEL.track(video, conf=track_conf, iou=track_iou)

	# Organize faces based on tracker ID
	faces_trk: dict[int, list[Face]] = dict()
	for fno, frame in tqdm(enumerate(tracked_frames), desc="Calculating face quality scores"):
		if frame.boxes.id is None: continue
		track_ids = frame.boxes.id.int().tolist()
		confs = frame.boxes.conf.float().tolist()
		boxes = torch.round(frame.boxes.xyxy).int().tolist()

		for track_id, conf, box in zip(track_ids, confs, boxes):
			if track_id not in faces_trk: faces_trk[track_id] = []
			quality_score = face_score(conf, frame.orig_shape, box, conf_weight=qscore_conf_weight, box_weight=qscore_box_weight, center_weight=qscore_center_weight)
			faces_trk[track_id].append(Face(tracked_frames[fno].orig_img, box, quality_score))
	
	for track_id in faces_trk.keys():
		faces_trk[track_id].sort(key=lambda x: x.score, reverse=True)
	
	# Generate embeddings for each face and cluster them
	embeddings = []
	for faces_by_trk in tqdm(faces_trk.values(), desc=f"Generating embedding for top {top_k_faces} faces"):
		cur_embedding = np.array([face.get_embedding() for face in faces_by_trk[:top_k_faces]])		
		cur_embedding = np.mean(cur_embedding, axis=0)
		embeddings.append(cur_embedding)
	
	print("Clustering embeddings...")
	cluster_ids = DBSCAN(eps=dbscan_eps, min_samples=1, metric="cosine").fit_predict(embeddings)
	
	faces_cls: list[list[Face]] = []
	for cluster_id, faces_by_trk in zip(cluster_ids, faces_trk.values()):
		if cluster_id == -1: continue
		if cluster_id == len(faces_cls): faces_cls.append([])
		faces_cls[cluster_id].append(faces_by_trk[0])
	
	print("Filtering clustered faces...")
	faces_cls_best: list[Face] = []
	for faces_by_cls in faces_cls:
		faces_cls_best.append(max(faces_by_cls, key=lambda x: x.score))
	
	print("Face recognition complete.")
	return faces_cls_best