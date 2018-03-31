import sys
from os import listdir

import cv2
import numpy as np
from os.path import isfile, join


def detect_shush(frame, location, ROI, cascade):
    mouths = cascade.detectMultiScale(ROI, 1.15, 3, 0, (20, 20))
    for (mx, my, mw, mh) in mouths:
        mx += location[0]
        my += location[1]
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    return len(mouths) == 0


def detect(frame, face_cascade, mouths_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #    gray_frame = cv2.equalizeHist(gray_frame)
    #    gray_frame = cv2.medianBlur(gray_frame, 5)

    faces = face_cascade.detectMultiScale(
        gray_frame, 1.15, 4, 0 | cv2.CASCADE_SCALE_IMAGE, (40, 40))
    detected = 0
    for (x, y, w, h) in faces:
        # ROI for mouth
        x1 = x
        h2 = int(h / 2)
        y1 = y + h2
        mouthROI = gray_frame[y1:y1 + h2, x1:x1 + w]

        if detect_shush(frame, (x1, y1), mouthROI, mouths_cascade):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if folder[-1] != "/":
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    window_name = None
    total_cnt = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            total_cnt += lCnt
            if window_name is not None:
                cv2.destroyWindow(window_name)
            window_name = f
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, img)
            cv2.waitKey(0)
    return total_cnt


def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showframe = True
    while showframe:
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False

    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + str(len(sys.argv) - 1) +
              "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('cascades/Mouth.xml')

    if len(sys.argv) == 2:  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, mouth_cascade)
