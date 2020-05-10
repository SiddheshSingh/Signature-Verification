import os
import cv2
from zipfile import ZipFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations

def imagePreprocess(image,size):# Size in format img_width,img_height
    image=cv2.resize(image, size) 
    #(thresh, image) = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # grayscale to binary using threshold
    image = image/255
    return image
def getData(loc,name_file,size,dic):
    img_list = []
    l = len(name_file)
    counter = 0
    for name in name_file:
        if counter==int(l/4):
          print("25% Completed..")
        elif counter==int(l/2):
          print("50% Completed..")
        elif counter==int(3*l/2):
          print("75% Completed..")
        counter+=1

        try:
            img = cv2.imread(os.path.join(loc,name),0)
            img = imagePreprocess(img,size)
            img = img.reshape((size[0],size[1],1))
            img_list.append(name)
            dic[name] = img
        except:
            print("Couldn't import ",name,"in Location:",loc)
            continue
    print("100% Completed")
    return img_list    

   
# Dataset 1
path_dataset1 = 'E:\Code data\Signature\Signatures\sample_Signature\sample_Signature'

def getNames1(loc):
    real_loc = os.path.join(loc,'genuine')
    forge_loc= os.path.join(loc,'forged')
    real_names = os.listdir(real_loc)
    forge_names= os.listdir(forge_loc)
    # Sorting forged list because it is not in order of elements
    # Sort it in order of "Last 2 Digits" (Excluding ".png") which denote who's sign it is 
    forge_names = sorted(forge_names,key= lambda x: int(x[-6:-4]))
    return real_names,forge_names

def getImages(loc,size,dic):
    print("Getting Dataset-1 Data and Saving inside the Dictionary..")
    real_names,forge_names = getNames1(loc)
    print("Getting Genuine Images..")
    real_img = getData(os.path.join(loc,'genuine'),real_names,size,dic)
    print("Getting Forged Images..")
    forge_img= getData(os.path.join(loc,'forged'),forge_names,size,dic)
    return np.asarray(real_img),np.asarray(forge_img)



path_dataset2 = 'E:\Code data\Signature\Signature2\signatures'

def getImages2(loc,size,dic):
    print("Getting Dataset2 Data..")
    real_names = os.listdir(os.path.join(loc,'full_org'))
    forg_names = os.listdir(os.path.join(loc,'full_forg'))
    img_real = getData(os.path.join(loc,'full_org'),real_names,size,dic)
    img_forg = getData(os.path.join(loc,'full_forg'),forg_names,size,dic)
    print("Data Import Complete!")
    return (np.asarray(img_real),np.asarray(img_forg))


def getDataset3(data,zipobject,dic,size):
    lis = []
    for c in range(0,len(data)):
        if c == len(data)//2:print("50% Complete")
        if c == len(data)//4:print("25% Complete")
        if c == 3*len(data)//4:print("75% Complete")
        i = data[c]
        img = np.asarray(Image.open(zipobject.open('BHSig260/Hindi/'+ i)))
        img = imagePreprocess(img ,size)
        img = img.reshape((size[0],size[1],1))
        lis.append(i)
        dic[i] = img
    print('100% Complete')
    return lis

def returnPairList(pairfile):
  x1,x2,y=[],[],[]
  for i in pairfile:
    t = i.split(' ')
    x1.append(t[0])
    x2.append(t[1])
    y.append(int(t[2]))
  return x1,x2,y

def getHindi(path,size,images_dictionary):
  real_list=[]
  forge_lis=[]
  with ZipFile(path, 'r') as z: 
    Fdata = z.read('BHSig260/Hindi/list.forgery').decode("utf-8").split("\n")
    Gdata = z.read('BHSig260/Hindi/list.genuine').decode("utf-8").split("\n")
    Fdata = Fdata[0:-1]
    Gdata = Gdata[0:-1]
    pairs = z.read('BHSig260/Hindi/Hindi_pairs.txt').decode("utf-8").split("\n")
    pairs=pairs[0:-1]
    print("Getting Genuine Data..")
    real_list=getDataset3(Gdata,z,images_dictionary,size)
    print("Getting Forged Data..")
    forge_list=getDataset3(Fdata,z,images_dictionary,size)
  return real_list,forge_list

def makeHindiPairs(real,forged):
  x1,x2,y = [],[],[]
  for i in range(0,160):
    fstart = i*30
    gstart = i*24
    for j in range(gstart,gstart+24):
      for k in range(j+1,gstart+24):
        x1.append(real[j])
        x2.append(real[k])
        y.append(1)
      for k in range(fstart,fstart+30):
        x1.append(real[j])
        x2.append(forged[k])
        y.append(0)
  return x1,x2,y


def makePairs(real_img,forged_img,no_of_writers):
    y=[]
    x1=[]
    x2=[]
    length = len(real_img) # Length of both is supposed to be same
    for i in range(0,length,no_of_writers): # Real-Real samples
        combs = list(combinations(range(i,i+no_of_writers),2))
        for each in combs:
            x1.append(real_img[each[0]])
            x2.append(real_img[each[1]])
            y.append(1)
            x1.append(real_img[each[0]])
            x2.append(forged_img[each[1]])
            y.append(0)
    return [np.asarray(x1),np.asarray(x2),np.asarray(y)]