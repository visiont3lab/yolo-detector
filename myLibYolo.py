import cv2 as cv
import numpy as np
import os

class detectorYolo:
    
    def __init__(self):
        pathData=os.path.join(".","yolo")
        self.config_path = os.path.join(pathData,"cfg", "yolov3.cfg")
        self.weights_path = os.path.join(pathData, "yolov3.weights")
        #self.config_path = os.path.join(pathData,"cfg", "yolov3-tiny.cfg")
        #self.weights_path = os.path.join(pathData, "yolov3-tiny.weights")
        
        #threshold when applying non-maxima suppression
        #self.nms = nms_th # set threshold for non maximum supression
        # loading all the class labels (objects)
        pathNames=os.path.join(pathData,"coco.names")
        self.classes = open(pathNames).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        COLORS = np.random.randint(0, 255, size=(len(self.classes), 3),dtype="uint8")
        # load the YOLO network
        #net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net = cv.dnn_DetectionModel(self.config_path, self.weights_path)
        #self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

    def doDetection(self, image,coordinates=[0,0],confidence_th=0.20,nms_th=0.2):
        #minimum probability to filter weak detections"
        self.c_threshold = confidence_th # set threshold for bounding box values
        #threshold when applying non-maxima suppression
        self.nms = nms_th # set threshold for non maximum supression
        #define the list of yolo coordinates
        yoloCoordinates=[]
        scoresCoordinates=[]
        #copy images
        image=image.copy()
        (H,W) = image.shape[:2]
        # Get the names of output layers
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # generate blob for image input to the network
        blob = cv.dnn.blobFromImage(image,1/255,(416,416),swapRB=True, crop=False)
        self.net.setInput(blob)
        #start = time.time()
        layersOutputs = self.net.forward(ln)        
        boxes = []
        confidences = []
        classIDs = []
        for output in layersOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]            
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.c_threshold :              
                    box = detection[0:4]* np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)              
        # Remove unnecessary boxes using non maximum suppression
        idxs = cv.dnn.NMSBoxes(boxes, confidences, self.c_threshold, self.nms)
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                #filtro solo alla classe persona
                #if(self.classes[classIDs[i]]=="person"):
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])            
                # draw a bounding box rectangle and label on the image
                #color = [int(c) for c in COLORS[classIDs[i]]]
                color=(0,255,0)
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
                
                #absolute coordinates
                #Controllo dimensioni roi per evitare falsi positivi. Elimina rettangoli con h molto maggiore di w. E
                #elimino rettangoli con score basso e Roi grande rispetto all'immagine elaborata
                if(w/h>0.3 and not (confidences[i]<0.5 and ((w/W>0.6) or (h/H>0.6)))):
                    yoloCoordinates.append([coordinates[0]+y,coordinates[1]+x,h,w])  
                    scoresCoordinates.append(confidences[i])  
                    #cv.imshow("im", image)
                    #cv.waitKey(0)
                    #print(h,w,H,W)
        #end = time.time()
        # print the time required
        #print("Time single image",end- start,"s")
        #return yoloCoordinates,scoresCoordinates
        return image

if __name__ == "__main__":
    #im = cv.imread("test_image.jpg",1)
    #cv.imshow("Image", im)
    #cv.waitKey(0)

    cap = cv.VideoCapture(0)
    dy = detectorYolo()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        #frame = cv.resize(frame, (256,256))
        image = dy.doDetection(frame)

        # Display the resulting frame
        cv.imshow('frame',image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()