from pathlib import Path
import importlib

import cv2


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo-12k-384p-augmented.pt"


def get_model_path() -> Path:
	if MODEL_PATH.exists():
		return MODEL_PATH
	raise FileNotFoundError(f"Could not find the model file: {MODEL_PATH}")


def main() -> None:
	model_path = get_model_path()
	YOLO = importlib.import_module("ultralytics").YOLO
	model = YOLO(str(model_path))

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open camera.")

	print("Press q to quit.")

	while True:
		ok, frame = cap.read()
		if not ok:
			print("Failed to read frame from camera.")
			break

		results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
		annotated_frame = results[0].plot()

		cv2.imshow("Hand detection", annotated_frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
