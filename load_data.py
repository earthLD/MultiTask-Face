import numpy as np
import os
import threading
from PIL import Image
from keras.applications.mobilenet import preprocess_input

def Image_label_generator_train(imgs_name , Idxs , labels, train=True):
    i = 0
    ageGenerator = []
    imgGenerator = []
    genderGenerator = []
    emotionGenerator = []
    ethnicityGenerator = []
    kpGenerator = []
        
    num = 0
    batch_size = 16
    train_path = '/data/img_celeba_ssd/'
    if train:
        Idx = shuffle_balance(Idxs, imgs_name, labels)
    else:
        Idx = Idxs
    img_num = len(Idx)
    while 1:
        if num >= img_num:
            if train:
                Idx = shuffle_balance(Idxs, imgs_name, labels)
            else:
                np.random.shuffle(Idxs)
                Idx = Idxs.copy()
            img_num = len(Idx)
            num = 0
        img_name = imgs_name[Idx[num]]
        #labelProcess
        attr = labels[img_name]
        
        age = float(attr['age'])/10.0
        kp = np.array(attr['loc'])*1.0
        
        gender = np.zeros((2))
        gender[int(attr['gender'])] = 1
        
        emotion = np.zeros((3))
        if int(attr['emotion'])==1:
            emotion[int(attr['emotion'])] = 1
        elif int(attr['emotion'])==6:
            emotion[2] = 1
        else:
            emotion[0] = 1
        
        ethnicity = np.zeros((4))
        ethnicity[int(attr['ethnicity'])] = 1
        
        #imgProcess
        #img = imread(train_path+img_name)
        img = Image.open(train_path+img_name).convert('RGB')
        w, h = img.size
        img = np.array(img.resize((192, 192), Image.ANTIALIAS))
        img = preprocess_input(img)
        kp[::2] = kp[::2]/float(h)
        kp[1::2] = kp[1::2]/float(w)
        ageGenerator.append(age)
        kpGenerator.append(kp)
        genderGenerator.append(gender)
        emotionGenerator.append(emotion)
        ethnicityGenerator.append(ethnicity)
        imgGenerator.append(img)

        if i>=batch_size-1:
            yield (np.array(imgGenerator), {'pred_age':np.array(ageGenerator), 
                                            'pred_kp':np.array(kpGenerator), 
                                            'pred_g':np.array(genderGenerator), 
                                            'pred_emo':np.array(emotionGenerator), 
                                            'pred_eth':np.array(ethnicityGenerator)})
            i = -1
            ageGenerator = []
            imgGenerator = []
            genderGenerator = []
            emotionGenerator = []
            ethnicityGenerator = []
            kpGenerator = []
        i += 1
        num += 1
def point_rotate(center,point,degree):
    ##rotate along counter-clockwise direction,center should be (x,y) coordinate ,x is col/2,y is row/2
    radian = np.pi / 180 * degree
    trans = np.array([[np.cos(radian),-np.sin(radian)],[np.sin(radian),np.cos(radian)]])
    tmp = point - center
    point_T = np.expand_dims(tmp,axis = 1)
    new_point = np.dot(trans,point_T)
    new_point = center + new_point.T
    return new_point[0]

def rotate(img, label):
    prob = random.random()
    if prob > 0.4:
        degree = random.randint(-90, 90)
        img1 = transform.rotate(np.array(Image.fromarray(img)),angle = degree,mode = 'wrap')
        center = np.array([img.shape[0]/2,img.shape[1]/2])
        temp = label.copy()
        for k in range(int(len(temp)/2)):
            point = np.array([temp[2*k+1],temp[2*k]])
            new_point = point_rotate(center,point,degree)
            temp[2*k] = int(new_point[1])
            temp[2*k+1] = int(new_point[0])
        return img1, temp
    else: 
        return img, label


def shuffle_balance(Idx, imgs_name, labels):
    age_num = np.zeros(8)
    emo_num = np.zeros(3)
    eth_num = np.zeros(4)
    Idx_res = []
    np.random.shuffle(Idx)
    for i in Idx:
        img_name = imgs_name[i]
        attr = labels[img_name]
        age = int(attr['age'])
        age = 11 if age<=10 else age
        age = 89 if age>=90 else age
        emo = int(attr['emotion'])
        if int(attr['emotion'])==1:
            emo = 1
        elif int(attr['emotion'])==6:
            emo = 2
        else:
            emo = 0
        if age_num[int(age/10)-1] < 6000 and emo_num[emo] < 8000 and eth_num[int(attr['ethnicity'])] < 6000:
            age_num[int(age/10)-1] += 1
            emo_num[emo] += 1
            eth_num[int(attr['ethnicity'])] += 1
            Idx_res.append(i)
    return Idx_res
        
        
def load_data():
    label_path = 'face++ssdloc.npy'
    labels = np.load(label_path).item()
    img_name = list(labels.keys())
    Idx = [i for i in range(len(labels))]
    np.random.shuffle(Idx)

    train_img_num = int(len(img_name)*0.996)
    Idx_train = Idx[:train_img_num]
    Idx_test = Idx[train_img_num:]
    
    print (len(Idx_train))
    print (len(Idx_test))
    
    train_generator = Image_label_generator_train(img_name, Idx_train, labels)
#     train_generator = Threadsafe_iter(train_generator)
    test_generator = Image_label_generator_train(img_name , Idx_test, labels, train=False)
#     test_generator = Threadsafe_iter(test_generator)
    return train_generator,test_generator
