import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cvzone
from collections import Counter  # Import Counter from collections module
import glob

model = YOLO("C:/Users/Admin/PythonLession/YoloModel/yolov8s.pt")

#my_file = open("coco.txt", "r")
#data = my_file.read()
#class_list = data.split("\n")

def object(frame):
    object_classes = []
    results = model.predict(frame)
    #  GPU / CPU extraction data from object
    result = results[0]
    boxes = results[0].boxes.xyxy.tolist()
    #print(boxes)
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    annotator = Annotator(frame, line_width=2, example=str(names))
    #frame =result.plot()  # Box phai la boxes = results[0].boxes.xywh.cpu()
    for box in result.boxes:
        label1 = result.names[box.cls[0].item()]
        cords1 = [round(x) for x in box.xyxy[0].tolist()]
        prob1 = round(box.conf[0].item(), 2)
        print("Object type: ", label1)
        print("Probablity: ", prob1)
        print("Cordinate: ", cords1)
        print("_")
        object_classes.append(label1)

    # Iterate through the results

    for box, cls, conf in zip(boxes, classes, confidences):
        annotator.box_label(box, names[int(cls)], (255, 42, 4))

    return object_classes

def count_objects_in_image(object_classes,image):
    counter = Counter(object_classes)
    print("Object Count in Image:")
    print(counter)
    n=0
    for obj, count in counter.items():
        print(f"{obj}: {count}")
        cvzone.putTextRect(image, f'{obj}', (50, 50+n), 1, 1,colorT=(255,255,255),colorR=(0,0,0),border=2)
        cvzone.putTextRect(image, f'{count}', (150, 50+n), 1, 1)

        '''
        # Add a rectangle and put text inside it on the image
    img, bbox = cvzone.putTextRect(
        img, "CVZone", (50, 50),  # Image and starting position of the rectangle
        scale=3, thickness=3,  # Font scale and thickness
        colorT=(255, 255, 255), colorR=(255, 0, 255),  # Text color and Rectangle color
        font=cv2.FONT_HERSHEY_PLAIN,  # Font type
        offset=10,  # Offset of text inside the rectangle
        border=5, colorB=(0, 255, 0)  # Border thickness and color
    )
        '''
        n=n+50

        #cv2.imshow("img", image)
    return image
path = r'C:\Users\Admin\PythonLession\YoloV8\CountingOBJ in Image\images\*.*'
for file in glob.glob(path):
    img = cv2.imread(file)
    # Detect in a region of picture / Frame
    #img = img[100:400, 0:300]
    img = cv2.resize(img, (1020, 500))
    object_classes = object(img)
    #print (object_classes) = ['car','car' .....]
    count_objects_in_image(object_classes,img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
