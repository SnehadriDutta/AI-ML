import numpy as np
import time
import cv2
#import pandas
import os
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
import warnings
#import keras
warnings.filterwarnings('ignore')

np.random.seed(20)

class Detector:

    def __init__(self):
        pass

    def readclasses(self, classesPath):
        with open(classesPath, 'r') as f:
            self.classList = f.read().splitlines()
            f.close()
        self.colorList = np.random.uniform(low=0, high=255, size=len(self.classList))

    def downloadModel(self, modelUrl):
        filename = os.path.basename(modelUrl)
        self.modelName = filename[:filename.index('.')]
        self.cachedir = "./pretrained-model"
        os.makedirs(self.cachedir, exist_ok=True)
        get_file(fname=filename, origin=modelUrl, cache_dir=self.cachedir, extract=True)


    def loadModel(self):
        self.model = tf.saved_model.load(os.path.join(self.cachedir + "/datasets", self.modelName, "saved_model"))

    def createBoundingBox(self, image, threshold = 0.5):


        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]
        detection = self.model(inputTensor)
        bboxs = detection['detection_boxes'][0].numpy()
        classIndexes = detection['detection_classes'][0].numpy().astype(np.int32)
        classScores = detection['detection_scores'][0].numpy()
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIdx = classIndexes[i]
                classText = self.classList[classIdx]
                classColor = self.colorList[classIdx]
                disply_txt = '{} : {}%'.format(classText, classConfidence)
                imH, imW, imC = image.shape
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (int(xmin*imW), int(xmax*imW), int(ymin*imH), int(ymax*imH))

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color = classColor, thickness=1)
                cv2.putText(image, disply_txt, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        return image

    def predictImage(self, imagePath, threshold = 0.5):

        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingBox(image, threshold)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold = 0.5):

        cap = cv2.VideoCapture(videoPath)

        if videoPath == 0 and not cap.isOpened():
            print("Camera failed to open")

        while True:
            ret, frame = cap.read()
            bboxImage = self.createBoundingBox(frame, threshold)
            cv2.imshow("Result", bboxImage)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            #ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()

detector = Detector()
detector.readclasses("CocoNames.txt")
detector.downloadModel("http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz")
detector.loadModel()
#detector.predictImage("india.jpg")
detector.predictVideo(0)