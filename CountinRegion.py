import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cvzone
from collections import Counter  # Import Counter from collections module
import glob
import numpy as np
model = YOLO("C:/Users/Admin/PythonLession/YoloModel/yolov8s.pt")


pts = np.array([[0,300],[900,300],[900,620],[0,620]], np.int32)
pts = pts.reshape((-1,1,2))

Obj_count = 0
objcenter =[]
object_classes_region=[]


# Find key value in dictionery
def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"
#-----------------------------------------
def count_objects_in_image(object_classes,image):
    counter = Counter(object_classes)
    print("Object Count in Image: " + str(counter))

    n=0
    for obj, count in counter.items():
        print(f"{obj}: {count}")
        cvzone.putTextRect(image, f'{obj}', (50, 50+n), 1, 1)
        cvzone.putTextRect(image, f'{count}', (150, 50+n), 1, 1)
        n=n+50

        #cv2.imshow("img", image)
    return image
#  Counting object in the region

def count_objects_in_region(object_classes_region,image):
    counter = Counter(object_classes_region)
    print("Object Count in Image: " + str(counter))
    n=0
    for obj, count in counter.items():
        print(f"{obj}: {count}")
        cvzone.putTextRect(image, f'{obj}', (50, 300+n), 1,1, (0,255,0))
        cvzone.putTextRect(image, f'{count}', (150, 300+n), 1, 1,(0,255,0))
        n=n+50

        #cv2.imshow("img", image)
    return image
#  Counting object in the region


def object(frame):

    object_classes = []


    results = model.predict(frame)
    #  GPU / CPU extraction data from object
    result = results[0]
    #print(result)
    boxes = results[0].boxes.xyxy.tolist()
    #print(boxes)
    classes = results[0].boxes.cls.tolist()
    #print(classes)
    names = results[0].names
    #print(names)
    confidences = results[0].boxes.conf.tolist()
    annotator = Annotator(frame, line_width=2, example=str(names))
    counterobj = 0

    for box in result.boxes:
        label1 = result.names[box.cls[0].item()]
        # Get Key Value of dictionery Python
        # print(get_key(label1,names))
        cords1 = [round(x) for x in box.xyxy[0].tolist()]
        prob1 = round(box.conf[0].item(), 2)
        #print("Object type: ", label1)
        #print("Probablity: ", prob1)
        #print("Cordinate: ", cords1)
        #print("_")
        object_classes.append(label1)
        #print(object_classes)
        # Find center point of object and draw on the picture
        x= float((cords1[0]+cords1[2])/2)  # x and y is the center point
        y= float((cords1[1]+cords1[3])/2)
        center=[x,y]
        objcenter.append(center)
        #print(objcenter)

        dist = cv2.pointPolygonTest(pts, (int(x), int(y)), False)
        if dist == 1:
            cv2.circle(frame,(int(center[0]),int(center[1])),5,(0,0,255),3)
            counterobj +=1
            object_classes_region.append(label1)

        else:
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), 3)
    # Iterate through the results ANOTATOR ALL PICTURES ITEMS
    print('Object in the region: '+ str(counterobj))


    for box, cls, conf in zip(boxes, classes, confidences):
        annotator.box_label(box, names[int(cls)], (255, 42, 4))

    return object_classes

#-------------------------------------------------------




path = r'C:\Users\Admin\PythonLession\YoloV8\yolov8-object-count-in-imag\images\*.*'
for file in glob.glob(path):
    img = cv2.imread(file)
    # Detect in a region of picture / Frame
    #img = img[100:400, 0:300]
    img = cv2.resize(img, (1000, 650))
    cv2.polylines(img, [pts], True, (255, 50, 255), 4)

    object_classes = object(img)
    #Counting object for all area of image
    #print (object_classes) = ['car','car' .....]
    count_objects_in_image(object_classes,img)
    # Count object in a region
    count_objects_in_region(object_classes_region, img)

    cv2.imshow("img", img)
    # Clear all the MATRIX after finish one Frame reading
    objcenter.clear()
    object_classes_region.clear()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
