import dlib
import numpy as np
import cv2
import os
import face_recognition
import datetime

class AttendanceMarker:
    def __init__(self,path):
        self.path = path
        
    def getdata(self):
        images = []
        classnames = []
        path = self.path
        mylist = os.listdir(path)
        for img in mylist:
            IMG = cv2.imread(f'{path}/{img}')
            images.append(IMG)
            classnames.append(os.path.splitext(img)[0])
        return images,classnames
    
    def encodingfunc(self,images):
        encodings = []
        for img in images:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodings.append(encode)
        print(f"Total {len(encodings)} Images found for Encoding !")
        return encodings
    
    
    def markattendance(self,name):
        with open('attendance_sheet.csv','r+') as f:
            datalist = f.readlines()
            namelist = []
            for line in datalist:
                entry = line.split(',')
                namelist.append(entry[0].upper()) # only names
            if name not in namelist:
                now = datetime.datetime.now()
                date = now.date()
                time = now.strftime("%H:%M:%S")
                f.writelines(f'\n{name},{date},{time}')
            f.close()
                
    def capture(self,classnames,encodings):
        cap = cv2.VideoCapture(0)
        while True:
            success,img = cap.read()
            img_small = cv2.resize(img,(0,0),None,0.25,0.25)
            img_small = cv2.cvtColor(img_small,cv2.COLOR_BGR2RGB)
            facesInCurrFrame = face_recognition.face_locations(img_small)
            encodeCurrFrame = face_recognition.face_encodings(img_small,facesInCurrFrame)

            for encodeface,faceloc in zip(encodeCurrFrame,facesInCurrFrame):
                match = face_recognition.compare_faces(encodings,encodeface)
                faceDistance = face_recognition.face_distance(encodings,encodeface)
                matchIdx = np.argmin(faceDistance)

                if match[matchIdx]:
                    name = classnames[matchIdx].upper()
                    y1,x2,y2,x1 = faceloc
                    y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    self.markattendance(name)

            cv2.imshow('Webcam',img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def runner(self):
        images,classnames = self.getdata()
        encodings = self.encodingfunc(images)
        self.capture(classnames,encodings)

if __name__=="__main__":
    atten = AttendanceMarker("./Images")
    atten.runner()
