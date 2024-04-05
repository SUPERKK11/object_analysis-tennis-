from ultralytics import YOLO
import cv2
class FaceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model(frame)
        if 'person' in results.names:
            face_boxes = results.xyxy[results.names.index('person')]
            return face_boxes
        else:
            return []

    def draw_faces(self, frame, face_boxes):
        for box in face_boxes:
            x1, y1, x2, y2, conf = box.tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    def process_video(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_boxes = self.detect_faces(frame)
            if face_boxes:
                self.draw_faces(frame, face_boxes)

            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

