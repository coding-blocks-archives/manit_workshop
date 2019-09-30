#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Mapping between names & labels
idx2name = {
}
files = os.listdir()
pics = []
Y = []

cnt = 0

for f in files:
    if f.endswith(".npy"):
        data = np.load(f)
        labels = np.ones(data.shape[0],dtype='int32')*cnt
        pics.append(data)
        idx2name[cnt] = f[:-4] 
        cnt += 1
        Y.append(labels)


# In[3]:


X = np.vstack(pics)
print(X.shape)


# In[4]:


Y = np.asarray(Y)
Y = Y.reshape((40,))
Y.shape


# In[5]:


X.shape,Y.shape


# In[6]:


idx2name


# In[7]:


def dist(a,b):
    return np.sum((a-b)**2)**.5    

def knn(X,Y,test_point,k=5):
    
    # 1 Step - Find dist of test_point from all points
    d = []
    m = X.shape[0]
    
    for i in range(m):
        current_dis = dist(X[i],test_point)
        d.append((current_dis,Y[i]))
    
    # Sort
    d.sort()
    
    # Take the first k elements after sorting (slicing)
    d = np.array(d[0:k])
    d = d[:,1]
    uniq,occ = np.unique(d,return_counts=True)
    #print(uniq,occ)
    idx = np.argmax(occ)
    pred = uniq[idx]
    
    return idx2name[int(pred)]


# In[8]:


#test_point = X[5]


# In[9]:


import cv2
import numpy as np

camera = cv2.VideoCapture(0)
facedetector = cv2.CascadeClassifier('../Day-1/face_template.xml')



while True:
    b,img = camera.read()
    
    if b==False:
        continue
    # Detect Faces
    faces = facedetector.detectMultiScale(img,1.2,5)
    
    # No face is detected
    if(len(faces)==0):
        continue

    
    # Draw bounding box around each face
    for f in faces:
        x,y,w,h = f
        green = (0,255,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),green,5)
        
        # Get the Pred for Cropped Face
        cropped_face = img[y:y+h,x:x+w]
        cropped_face = cv2.resize(cropped_face,(100,100))
        pred = knn(X,Y,cropped_face)
        cv2.putText(img, pred, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    
    # Show the New Image
    cv2.imshow("Title",img)
    #Add some delay 1 ms between 2 frames
    key = cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




