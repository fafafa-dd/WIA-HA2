import numpy as np
import time
import json
import tensorflow as tf
import keras

# function opens json file and returns it in data
def open_js_file(name):
    #t1 = time.time()
    json_ = open(name)
    data = json.load(json_)
    #t2 = time.time()
    #print('Needed {} sec = {} min'.format(t2-t1, (t2-t1)/60))
    return data

# function gets json file (data) as input and returns acceleration data, vehicle-ID, crossing speed and labels
def get_acceleration_fid_v_labels(data, va=False, ha=False): 
    # data = js FILE
    # va=False, ha=False-> roi not separated
    # va=True -> acceleration data of roi_va
    # ha=True -> acceleration data of roi_ha
    
    # dataframe for acceleration measurement series
    shape = len(data['0th crossing'][2]['accz'])
    df = np.zeros((len(data), shape, 3))
    
    # label vorderachse, label hinterachse, fahrzeug-id, geschwindigkeit
    l_va, l_ha, fid, v =[], [], [], []
    
    # idx1 = index of label va, idx2 = index pf label ha
    # index of label va/ label ha depends on chosen dataset (9 or 10)
    if va == True: 
        idx1 = 9
    elif ha == True:
        idx2 = 9
    elif va == False and ha == False: 
        idx1 = 9
        idx2 = 10
        
    # save acceleration data of every crossing (len(data)=number of crossings)    
    for i in range(len(data)):
        
        # acceleration data in x, y, z direction
        df[i,:,0] = data[str(i)+'th crossing'][0]['accx'] 
        df[i,:,1] = data[str(i)+'th crossing'][1]['accy'] 
        df[i,:,2] = data[str(i)+'th crossing'][2]['accz'] 
        
        # vehicle id
        fid = np.append(fid, data[str(i)+'th crossing'][4]['fid'])
        # speed of the crossing
        v = np.append(v, data[str(i)+'th crossing'][7]['geschwindigkeit']) 
        
        # differentiate between datasets 
        # separated va -> label va is needed only 
        # separated ha -> label ha is needed only
        # not separated -> label va and label ha are needed both
        if va == True: 
            l_va = np.append(l_va, data[str(i)+'th crossing'][idx1]['label_va'])
        elif ha == True:
            l_ha = np.append(l_ha, data[str(i)+'th crossing'][idx2]['label_ha'])
        elif va == False and ha == False: 
            l_va = np.append(l_va, data[str(i)+'th crossing'][idx1]['label_va'])
            l_ha = np.append(l_ha, data[str(i)+'th crossing'][idx2]['label_ha'])
    
    # return statements (depend ondataset)
    if va == False and ha == False: 
        return df, fid, v, l_va, l_ha
    elif va == True:
        return df, fid, v, l_va
    elif ha == True:
        return df, fid, v, l_ha

# label 0 = defect, label 1 = intact
def labels_roi(lva, lha):
    l_roi=[]
    #class0: va defect & ha defect
    #class0: va defect & ha intact
    #class0: va intact & ha defect
    #class0: va intact & ha intact
    for i in range(len(lva)):
        if lva[i]==0.0 and lha[i]==0.0:
            l_roi = np.append(l_roi, 0)
        elif lva[i]==0.0 and lha[i]==1.0:
            l_roi = np.append(l_roi, 1)
        elif lva[i]==1.0 and lha[i]==0.0:
            l_roi = np.append(l_roi, 2)
        elif lva[i]==1.0 and lha[i]==1.0:
            l_roi = np.append(l_roi, 3)
    return l_roi

def get_fids(fid):
    f = []
    #idx = []
    
    # go through all fids
    for i in range(1,len(fid)):
        
        # save the current (i-1) fid, if the next one is a new one
        if fid[i-1] != fid[i]:
            #idx = np.append(idx, i)
            f = np.append(f, fid[i-1])
            
    # delete the fids which are doubled
    f = np.unique(f)
    return f

# compute the distribution of the labels
def label_distr(y, va_ha_sep=False):
    count0, count1, count2, count3 = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i]==0:
            count0 += 1
        elif y[i]==1:
            count1 += 1
        elif y[i]==2:
            count2 += 1
        elif y[i]==3:
            count3 += 1
            
    #print('Distribution labels:',round(count0/len(y),4), round(count1/len(y),4), round(count2/len(y),4), round(count3/len(y),4))  
    if va_ha_sep==False:
        return np.array([count0/len(y), count1/len(y), count2/len(y), count3/len(y)])
    elif va_ha_sep==True:
        return np.array([count0/len(y), count1/len(y)])

# compare the distribution of the testset and the trainset with the original distribution of the labels in whole dataset
#it is allowed to have a deviation of 'percent'%
def compare_distr(distr_test, distr_train, distr_roi, percent=10):
    
    # does the testset has a deviation less than 'percent'% from the original distribution?
    d_test = (distr_test < (1+percent/100)*distr_roi).all() and (distr_test > (1-percent/100)*distr_roi).all()
    
    # does the trainset has a deviation less than 'percent'% from the original distribution?
    d_train = (distr_train < (1+percent/100)*distr_roi).all() and (distr_train > (1-percent/100)*distr_roi).all()
    
    # if test- and trainset both have a deviation less than 'percent'%, set return statement True
    same_distr = False
    if d_test==True and d_train==True:
        same_distr = True    
    return same_distr

def train_test_split(fid, df, l_roi, va_ha_sep=False): 
    # call get_fids -> f contains the fid's
    f = get_fids(fid)
    # 1. condition to exit the loop (testdata should not contain more than 21 % of the whole data)
    testdata_21per=False
    # 2. condition to exit the loop (test- and traindata should have approximatly the same label distribution as the whole dataset)
    same_distr = False
    # while test_data contains less than 21 % and the distribution is not closely the same do ...
    while testdata_21per==False and same_distr==False:
        # number of fid's in dataset
        n = fid.shape[0]  
        # list to save the indices of the testdata
        test_idx = []
        # list which will contain test data
        x_test = []
        
         # condition that testdata should contain at least 20 percent of whole data
        testdata_20per = False
        # do the following loop as long as testdata contain less than 20 percent of whole data
        while testdata_20per==False:
            # random choice of a fid
            random_fid = np.random.choice(f)
            # Every fid should be chosen one time. Save the indices where random_fid==f, such that these cant be deleted later 
            r_idx = np.where(random_fid==f)
            r_idx = np.array(r_idx)
            # reshape it and save it as integer
            r_idx = (np.reshape(r_idx, (r_idx.shape[1],))).astype(int)
            # delete randomly chosen fid such that it can't be chosen in next loop round
            f = np.delete(f, r_idx)
            
            # x = indices of crossings with the randomly chosen fid of this loop round 
            x = np.where(random_fid==fid)
            x = np.array(x)
            # reshape x and save it as int
            x = (np.reshape(x, (x.shape[1],))).astype(int)
            # save the indices of the crossings which are in testdataset now
            test_idx = (np.append(test_idx, x)).astype(int)
            
            # testset
            x_test = np.append(x_test, df[x,:,:])
            x_test = np.reshape(x_test, (-1, df.shape[1], df.shape[2]))
            
            # condition that testdata should contain at least 20% of whole data
            if x_test.shape[0] >= 0.2*n: testdata_20per = True
        
        # crossings = indices of all crossings (is needed to built traindataset in next step)
        crossings = np.arange(n)
        # to separate test from train, overwrite the indices of test with -1
        crossings[test_idx] = -1
        # train_idx will contain indices of train crossings
        train_idx = []
        # fill train_dix with the values of crossings where crossings isn't -1 (crossings[i]!=-1) and save them as integer
        for i in range(n):
            if crossings[i]!=-1:
                train_idx = (np.append(train_idx, crossings[i])).astype(int)
        
        # save the corresponding labels into y_train and y_test
        y_test = l_roi[test_idx]
        y_train = l_roi[train_idx]
        
        # calculate the distribution of the labels in test and traindata
        distr_test = label_distr(y_test, va_ha_sep)
        distr_train = label_distr(y_train, va_ha_sep)
        distr_roi = label_distr(l_roi, va_ha_sep)
        
        # compare the distributions of the labels in test and traindata with the distribution of the labels in whole data
        same_distr = compare_distr(distr_test, distr_train, distr_roi)
        # condition to stop the loop, if testdataset gets bigger than 21% of the whole dataset
        if x_test.shape[0]<0.21*n: testdata_21per = True
    
    # save the traindataset        
    x_train = df[train_idx,:,:]
    return x_train, y_train, x_test, y_test

def train_test_split_cross_validation(fid, df, l_roi, before, va_ha_sep=False): 
    # call get_fids -> f contains the fid's
    f = get_fids(fid)
    # 1. condition to exit the loop (testdata should not contain more than 21 % of the whole data)
    testdata_21per=False
    test_crossvalidation  = False
    # 2. condition to exit the loop (test- and traindata should have approximatly the same label distribution as the whole dataset)
    same_distr = False
    # while test_data contains less than 21 % and the distribution is not closely the same do ...
    while testdata_21per==False and same_distr==False:
        # number of fid's in dataset
        n = fid.shape[0]  
        # list to save the indices of the testdata
        test_idx = []
        # list which will contain test data
        x_test = []
        test = []
        # condition that testdata should contain at least 20 percent of whole data
        testdata_20per = False
        # do the following loop as long as testdata contain less than 20 percent of whole data
        while testdata_20per==False:
            # random choice of a fid
            random_fid = np.random.choice(f)
            test.append(int(random_fid))
            # Every fid should be chosen one time. Save the indices where random_fid==f, such that these cant be deleted later 
            r_idx = np.where(random_fid==f)
            r_idx = np.array(r_idx)
            # reshape it and save it as integer
            r_idx = (np.reshape(r_idx, (r_idx.shape[1],))).astype(int)
            # delete randomly chosen fid such that it can't be chosen in next loop round
            f = np.delete(f, r_idx)           
            # x = indices of crossings with the randomly chosen fid of this loop round 
            x = np.where(random_fid==fid)
            x = np.array(x)
            # reshape x and save it as int
            x = (np.reshape(x, (x.shape[1],))).astype(int)
            # save the indices of the crossings which are in testdataset now
            if x not in test_idx:
                test_idx = (np.append(test_idx, x)).astype(int)
                # testset
                x_test = np.append(x_test, df[x,:,:])
                x_test = np.reshape(x_test, (-1, df.shape[1], df.shape[2]))
            # condition that testdata should contain at least 20% of whole data
            if x_test.shape[0] >= 0.2*n: testdata_20per = True
        for i in range(len(test)):
            swap = i + np.argmin(test[i:])
            (test[i], test[swap]) = (test[swap], test[i])
        # crossings = indices of all crossings (is needed to built traindataset in next step)
        crossings = np.arange(n)
        # to separate test from train, overwrite the indices of test with -1
        crossings[test_idx] = -1
        # train_idx will contain indices of train crossings
        train_idx = []
        # fill train_dix with the values of crossings where crossings isn't -1 (crossings[i]!=-1) and save them as integer
        for i in range(n):
            if crossings[i]!=-1:
                train_idx = (np.append(train_idx, crossings[i])).astype(int)
        
        # save the corresponding labels into y_train and y_test
        y_test = l_roi[test_idx]
        y_train = l_roi[train_idx]
        
        # calculate the distribution of the labels in test and traindata
        distr_test = label_distr(y_test, va_ha_sep)
        distr_train = label_distr(y_train, va_ha_sep)
        distr_roi = label_distr(l_roi, va_ha_sep)
        
        # compare the distributions of the labels in test and traindata with the distribution of the labels in whole data
        same_distr = compare_distr(distr_test, distr_train, distr_roi)
        # condition to stop the loop, if testdataset gets bigger than 21% of the whole dataset
        if x_test.shape[0]<0.21*n: testdata_21per = True
        if test not in before: test_crossvalidation = True
    before.append(test)
    # save the traindataset        
    x_train = df[train_idx,:,:]
    return x_train, y_train, x_test, y_test, before

def bring_in_right_shape_self(x_train, y_train, x_test, y_test, num_classes=4, num_features=512*3):
    #from tensorflow import to_categorical
    #reshape y
    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))
    
    #cast to np.float32
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Compute the categorical classes (doenst requested when binary classification)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
     
    # reshape x 
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]* x_train.shape[2], order='F')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2], order='F') 
    return x_train, y_train, x_test, y_test

def sigmoid(x):    
    return 1.0/(1.0 + np.exp(-x))

def forward_pass(x):    
#forward pass - preactivation and activation    
    x1, x2 = x    
    a1 = w1*x1 + w2*x2 + b1    
    h1 = sigmoid(a1)    
    a2 = w3*x1 + w4*x2 + b2    
    h2 = sigmoid(a2)   
    a3 = w5*self.h1 + self.w6*self.h2 + self.b3    
    h3 = self.sigmoid(self.a3)    
    return self.h3