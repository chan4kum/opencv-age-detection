"""Downloads the pretrained face + age models into ./models."""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/"
URLS = {
    "age_deploy.prototxt": BASE + "age_deploy.prototxt",
    "opencv_face_detector.pbtxt": BASE + "opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": BASE + "opencv_face_detector_uint8.pb",
    "age_net.caffemodel": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
}

os.makedirs("models", exist_ok=True)
for name, url in URLS.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        print("Downloading", name)
        urllib.request.urlretrieve(url, path)
print("Done.")
