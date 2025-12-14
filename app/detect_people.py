from deepface import DeepFace
import dlib
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from ultralytics import YOLO

DETECT_MODEL = YOLO("./outputs/models/yolo-detect.pt")
SHAPE_PREDICTOR = dlib.shape_predictor("./outputs/models/shape_predictor_5_face_landmarks.dat")

class Face:
	def __init__(self, orig_img: np.ndarray, box: np.ndarray, conf: float):
		self.orig_img: np.ndarray = orig_img
		self.box: np.ndarray = box
		self.conf: float = conf
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

def find_people(
		video: str | np.ndarray,
		dbscan_eps: float = 0.25,
		track_conf: float = 0.6,
		track_iou: float = 0.5
	):
	# Detect and track all faces in a video
	tracked_frames = DETECT_MODEL.track(video, conf=track_conf, iou=track_iou)
	print("Face tracking finished.")

	# Organize faces based on tracker ID
	faces_trk: dict[int, list[Face]] = dict()
	for fno, frame in enumerate(tracked_frames):
		if frame.boxes.id is None: continue
		track_ids = frame.boxes.id.int().tolist()
		confs = frame.boxes.conf.float().tolist()
		boxes = torch.round(frame.boxes.xyxy).int().tolist()

		for track_id, conf, box in zip(track_ids, confs, boxes):
			if track_id not in faces_trk: faces_trk[track_id] = []
			faces_trk[track_id].append(Face(tracked_frames[fno].orig_img, box, conf))
	
	# Select one face with the best confidence score for each tracker ID
	print("Filtering tracked faces...")
	faces_trk_best: dict[int, Face] = dict()
	for track_id, faces in faces_trk.items():
		faces_trk_best[track_id] = max(faces, key=lambda x: x.conf)
	
	# Generate embeddings for each face and cluster them
	print("Generating embeddings...")
	embeddings = np.array([face.get_embedding() for face in faces_trk_best.values()])
	print("Clustering embeddings...")
	cluster_ids = DBSCAN(eps=dbscan_eps, min_samples=1, metric="cosine").fit_predict(embeddings)
	
	faces_cls: list[list[Face]] = []
	for cluster_id, face in zip(cluster_ids, faces_trk_best.values()):
		if cluster_id == len(faces_cls): faces_cls.append([])
		faces_cls[cluster_id].append(face)
	
	print("Filtering clustered faces...")
	faces_cls_best: list[Face] = []
	for faces in faces_cls:
		faces_cls_best.append(max(faces, key=lambda x: x.conf))
	
	print("Face recognition complete.")
	return faces_cls, faces_cls_best