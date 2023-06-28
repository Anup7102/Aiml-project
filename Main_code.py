import pandas as pd
import cv2
import os
import numpy as np
import tensorflow as tf
import csv
import datetime
import time
#from py_send_mail import send_mail
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image                  
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
#img = cv2.cvtColor(cv2.imread("3.jpg"), cv2.COLOR_BGR2RGB)
#detector=detector.detect_faces(img)

#harcascadePath = "haarcascade_frontalface_default.xml"
#detector = cv2.CascadeClassifier(harcascadePath)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0
ALARM_ON = False




# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)



from glob import glob
#Mask_model = tf.keras.models.load_model('trained_model_CNN1.h5')
Mask_model = load_model('VGG16.h5')
Present_Absent=["present","absent"]
s_list= [item[7:-1] for item in sorted(glob("./data/*/"))]#['Anupama','Jayshri','Ramesh','Vijaylaxmi','Anant','venkat']
def get_index_positions_2(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list

def face_detection_realtime():
    
    cnt=1
    hit1=1
    
    cap = cv2.VideoCapture(0)
    COUNTER = 0
    vecharrar=np.zeros((len(s_list),))
    while True:
        cap_not, image = cap.read()
        if cap_not:
            if len(image.shape)==3:
            # gray scale
                gray =image# cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray=image
        else:
            return
        # Get faces into webcam's image
        
        face_cordinates = []
        # For each detected face
        
       
        #faces = detector.detectMultiScale(gray, 1.3, 5)
        faces=detector.detect_faces(gray)
        for result in faces:
            x1, y1, w, h = result['box']
            x2=x1+w
            y2=y1+h
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face_cordinates.append((x1, y1, w, h))
            
            face_image =image[y1:y1+h,x1:x1+w]
            cv2.imwrite('./Images/'+str(cnt)+'.jpg',face_image)
            cv2.imwrite('temp.jpg',face_image)
            cnt+=1
            face_image=cv2.resize(face_image,(256,256))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


                #cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #Clasification
            test_tensors = paths_to_tensor('temp.jpg')/255
            #test_tensors=np.expand_dims(face_image, axis=0)
            pred=Mask_model.predict(test_tensors)
            p1=np.argmax(pred)
            print(np.amax(pred))

            vecharrar[p1]=1
                
            #vecharrar=[s0,s1,s2,s3,s4,s5]
            if p1<=len(s_list) and np.amax(pred)>0.9:
                text=s_list[p1]
            else:
                text=' Not Recognize'
            # Put text above face
            cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 100, 180), 2)

        # Display image
        cv2.imshow("Output", image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    return vecharrar

if __name__ == "__main__":

    
    run1=True
    # LPB feature based face detecton
       
    
    # Start if subject Code is correct
    if run1==True:
        attadance1=face_detection_realtime()
    else:
        
        print("Enter Correct Subject details")

    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    
    Student=[]
    itr=1
    for name1 in s_list:
        if attadance1[itr-1]>=1:
            AT_present=Present_Absent[0]
        else:
            AT_present=Present_Absent[1]
            
        T_S = [itr, name1, timeStamp ,date, AT_present]
        Student.append(T_S)
        itr+=1

    print(np.array(Student))
##    fields = ['ID','Name', 'Time','Data', 'Attandance']
##    
##    # name of csv file  
##    filename = './Attandance_Data/' + str(date) + "Attandance.csv"
##    
##    # writing to csv file  
##    with open(filename, 'w') as csvfile:
##        # creating a csv writer object  
##        csvwriter = csv.writer(csvfile)  
##        # writing the fields  
##        csvwriter.writerow(fields)  
##        # writing the data rows  
##        csvwriter.writerows(Student)
##
##
##    X=[]    
##    cw_directory = os.getcwd()
##    folder = cw_directory+'/Attandance_Data'
##    attandance_count=0
##    for filename in os.listdir(folder):
##        sub_dir=(folder+'/'+filename)
##        CSV_DATA = pd.read_csv(sub_dir)
##        attandance_count+=1
##        cnt=0
##        idx1= CSV_DATA.iloc[:,0]
##        #CSV_DATA.iloc[i,:].values
##        for i in idx1:
##            if str(i) != 'nan':
##                X.append(CSV_DATA.iloc[cnt,:].values)
##            cnt+=1
##                          
##    X1=np.array(X)
##    Attandance_count=[]
##    #print(X1)
##    
##    name1=X1[:,1]
##    at_pr_ab=X1[:,4]
##    for s1 in s_list:
##        index1 = get_index_positions_2(name1, s1)
##        atcnt=0
##        for id1 in index1:
##            pr_ab =at_pr_ab[id1]
##            if pr_ab=='present':
##                atcnt+=1
##
##        #atcnt=np.double(atcnt)
##        if atcnt!=0:
##            Attandance_percetn=round((atcnt/len(os.listdir(folder)))*100);
##        else:
##            Attandance_percetn=atcnt
##            
##        Attandance_count.append([s1,Attandance_percetn])        
##            
##        
##        
##    print("****************************over all percentage of Attancance*********************************\n")   
##    print(np.array(Attandance_count))
##    send_mail('./Attandance_Data/' + str(date) + "Attandance.csv") 
##
##
##
##
##
##
##
##
##
##






    
