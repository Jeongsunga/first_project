import cv2
import os
import glob
import dlib
from datetime import datetime

detector = dlib.get_frontal_face_detector()

cnt = 10000
img_list_np = []
base_dir = './faces'
file_name = "Face_Detection"
file_name2 = "No_Face_Detection"
detect_dir = os.path.join(base_dir, file_name)
detect_dir2 = os.path.join(base_dir, file_name2)

if not os.path.exists(detect_dir):
    os.mkdir(detect_dir)

if not os.path.exists(detect_dir2):
    os.mkdir(detect_dir2)

now = datetime.now()
print(now)
dirs = [d for d in glob.glob(detect_dir2) if os.path.isdir(d)]
for dir in dirs:
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %d files' % (dir, len(files)))
    for file in files:
        image_cv2 = cv2.imread(file)
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:  # If faces are detected
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # bounding box
                color = (0, 0, 255)
                cv2.rectangle(image_cv2, (x, y), (x + w, y + h), color, 5)

                # confidence (not available in dlib)
                # confidence = "{:.2f}".format(confidence)
                # position = (x, y - 10)
                # cv2.putText(image_cv2, confidence, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3, cv2.LINE_AA)

            print("ok" + file)
            file_name_path = os.path.join(detect_dir, str(cnt) + '.jpg')
            cnt += 1
            cv2.imwrite(file_name_path, image_cv2)

        else:  # If no faces are detected
            print("no" + file)
            file_name_path = os.path.join(detect_dir2, str(cnt) + '.jpg')
            cnt += 1
            cv2.imwrite(file_name_path, image_cv2)

now = datetime.now()
print(now)