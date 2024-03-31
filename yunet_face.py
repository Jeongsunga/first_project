import cv2
import os, glob
import matplotlib.pyplot as plt

eye_cascPath = "../First_Project/Resource/haarcascade_eye_tree_eyeglasses.xml"  #eye detect model
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
detector = cv2.FaceDetectorYN.create("../First_Project/Resource/face_detection_yunet_2023mar.onnx", "", (0, 0))

base_dir = './faces'
dirs = [d for d in glob.glob(base_dir) if os.path.isdir(d)]
for dir in dirs:
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %dfiles'%(dir, len(files)))
    for file in files:
        image_cv2_yunet = cv2.imread(file)
        height, width, _ = image_cv2_yunet.shape
        detector.setInputSize((width, height))
        _, faces = detector.detect(image_cv2_yunet)
        # if faces[1] is None, no face found
        if faces is not None:
            for face in faces:
                # parameters: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm

                # bouding box
                box = list(map(int, face[:4]))
                color = (0, 0, 255)
                cv2.rectangle(image_cv2_yunet, box, color, 5)

                # confidence
                confidence = face[-1]
                confidence = "{:.2f}".format(confidence)
                position = (box[0], box[1] - 10)
                cv2.putText(image_cv2_yunet, confidence, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3, cv2.LINE_AA)

            print("ok" + file)
            plt.figure()
            plt.imshow(image_cv2_yunet)
            plt.show()
            plt.axis('off')
            plt.close()

        else:
            print("no" + file)
            plt.figure()
            plt.imshow(image_cv2_yunet)
            plt.show()
            plt.axis('off')
            plt.close()
