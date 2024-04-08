import cv2
import os, glob

detector = cv2.FaceDetectorYN.create("../First_Project/Resource/face_detection_yunet_2023mar.onnx", "", (0, 0))

eye_cascPath = "../First_Project/Resource/haarcascade_eye_tree_eyeglasses.xml"  #eye detect model
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

base_dir = './faces/Face_Detection'
dirs = [d for d in glob.glob(base_dir) if os.path.isdir(d)]
for dir in dirs:
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %dfiles'%(dir, len(files)))
    for file in files:
        image_cv2_yunet = cv2.imread(file)
        height, width, _ = image_cv2_yunet.shape
        detector.setInputSize((width, height))
        _, faces = detector.detect(image_cv2_yunet)
        frame = cv2.cvtColor(image_cv2_yunet, cv2.COLOR_BGR2GRAY)
        frame = frame[int(faces[0][1]):int(faces[0][1]) + int(faces[0][3]), int(faces[0][0]):int(faces[0][0]) + int(faces[0][2]):1]
        eyes = eyeCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        if len(eyes) == 0:
            print('no eyes!!!')
        else:
            print(file)
            print('eyes!!!')
        cv2.imshow('img', image_cv2_yunet)
        if cv2.waitKey(0) == ord('q') or cv2.waitKey(0) == ord('Q'):
            cv2.destroyAllWindows()
