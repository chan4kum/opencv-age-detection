import argparse
import cv2

AGES = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
MEAN = (78.4263377603, 87.7689143744, 114.895847746)

face_net = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
age_net = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")


def annotate(frame, conf_thresh=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    dets = face_net.forward()
    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] < conf_thresh:
            continue
        x1, y1, x2, y2 = (dets[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN, swapRB=False)
        age_net.setInput(blob)
        age = AGES[age_net.forward()[0].argmax()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Age {age}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    args = p.parse_args()
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Could not read {args.image}")
        cv2.imshow("Age Detection", annotate(img))
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Age Detection", annotate(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
