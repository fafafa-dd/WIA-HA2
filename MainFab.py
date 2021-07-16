import numpy as np
import json
import reading_splitting_dataset_functions
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import keras
import keras_metrics

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Zum Ausführen in der Shell:::

import numpy as np
import json
import sys
sys.path.append("C:\\Users\\fabia\\Desktop\Seminar\\Hausaufgabe\\Programmcode")
import reading_splitting_dataset_functions


Für Reload:::
import importlib
importlib.reload(MainFab)


"""

"""
Datenaufbau:
   
data_roi                            : Dictionary
data_roi["0th crossing"]            : List
data_roi["0th crossing"][0]         : Dictionary

Ab hier: es gibt immer nur einen key im Dictionary
data_roi["0th crossing"][0]["accx"] : List
data_roi["0th crossing"][1]["accy"] : List
.
.
.
data_roi["0th crossing"][3]["hersteller"] : String

die 12 keys sind :
[0]"accx"             : List(float)
[1]"accy"             : List(float)
[2]"accz"             : List(float)
[3]"hersteller"       : String
[4]"fid"              : float (sind aber eigentlich ints)
[5]"id_"              : float (sind aber eigentlich ints)
[6]"winkel"           : float
[7]"geschwindigkeit"  : float
[8]"modell"           : string
[9]"label_va"         : float (sind aber eigentlich ints)
[10]"label_ha"        : float (sind aber eigentlich ints)
[11]"roi_t2"          : float


"""

def getRawData():
    data_roi = reading_splitting_dataset_functions.open_js_file('data_preprocessed_roi.JSON')
    print("getRawData() : Einlesen der Rohdaten erfolgreich (",len(data_roi), "/ 33676 ) Datensätze")
    return data_roi

"""
df_roi  : accx, accy, accz Daten
fid_roi : fids
v_roi   : Überfahrgeschwindigkeiten
lva_roi : Label Vorderachse
lha_roi : Label Hinterachse
l_roi   : Label Gesamt
"""
def getData():
    df_roi, fid_roi, v_roi, lva_roi, lha_roi = reading_splitting_dataset_functions.get_acceleration_fid_v_labels(getRawData())
    l_roi = reading_splitting_dataset_functions.labels_roi(lva_roi, lha_roi)
    #print(l_roi)
    print("getData() : Verarbeiten der Rohdaten erfolgreich (", len(v_roi), ") Datensätze")
    return df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi

def getPartialRawData(anzahl):
    print("getPartialRawData() : getRawData() wird aufgerufen")
    return dict(list(getRawData().items())[:anzahl])

def getPartialData(anzahl):
    df_roi, fid_roi, v_roi, lva_roi, lha_roi = reading_splitting_dataset_functions.get_acceleration_fid_v_labels(getPartialRawData(anzahl))
    l_roi = reading_splitting_dataset_functions.labels_roi(lva_roi, lha_roi)
    print("getPartialData() : Verarbeiten der Rohdaten erfolgreich (", len(v_roi), ") Datensätze")
    return df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi

"""
Geschwindigkeiten:::
np.mean(v_roi) = 8.043573554557916
np.min(v_roi) = 0.0
np.max(v_roi) = 17.472413793103446
"""
def getTruncatedDataAsymmetrical(erlaubteAbweichungRelativLinks, erlaubteAbweichungRelativRechts):
    print("getTruncatedData() : getData() wird aufgerufen")
    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getData()

    Durchschnitt = np.mean(v_roi)
    l            = erlaubteAbweichungRelativLinks*Durchschnitt
    r            = erlaubteAbweichungRelativRechts*Durchschnitt
    
    mask       = [ [True if ( (v_roi[i] > Durchschnitt - l) & (v_roi[i] < Durchschnitt + r) )\
                         else False for i in range(len(v_roi)) ] ]
    c = sum([1 if mask[0][i]==True else 0 for i in range(len(mask[0]))])

    print("Truncation:", Durchschnitt-l, "|", Durchschnitt, "|", Durchschnitt+r, "führt zu ", c, "/", len(v_roi), "Datensätzen")

    return df_roi[tuple(mask)], fid_roi[tuple(mask)], v_roi[tuple(mask)], \
           lva_roi[tuple(mask)], lha_roi[tuple(mask)], l_roi[tuple(mask)]



def getTruncatedData(erlaubteAbweichungRelativ):
    return getTruncatedDataAsymmetrical(erlaubteAbweichungRelativ, erlaubteAbweichungRelativ)

# AUFGABE 1+2
def seqNet(df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi):
    
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()

    CrossValDurchläufe = 4
    lossVec          = np.zeros(CrossValDurchläufe)
    accuracyVec      = np.zeros(CrossValDurchläufe)
    val_lossVec      = np.zeros(CrossValDurchläufe)
    val_accuracyVec  = np.zeros(CrossValDurchläufe)
    precisionVec     = np.zeros(CrossValDurchläufe)
    val_precisionVec = np.zeros(CrossValDurchläufe)
    recallVec        = np.zeros(CrossValDurchläufe)
    val_recallVec    = np.zeros(CrossValDurchläufe)
    before = []

    for i in range(CrossValDurchläufe):
        print("Durchlauf "+str(i+1)+ "/"+str(CrossValDurchläufe))

        before = []
        
        x_train, y_train, x_test, y_test, before  =  \
                 reading_splitting_dataset_functions.train_test_split_cross_validation(fid_roi, df_roi, l_roi, before)

        x_train, y_train, x_test, y_test  =  \
                 reading_splitting_dataset_functions.bring_in_right_shape_self(x_train, y_train, x_test, y_test)
        
        # define the 1536-256-128-10 architecture using Keras
        model = Sequential()

        #Ürsprüngliche Version
        #model.add(Dense(256, activation="sigmoid")) # oder 'softmax', 'tanh', 'relu'
        #model.add(Dense(128, activation="sigmoid"))

        #Verbesserung 1 Kernelregularizer
        model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(), activation="tanh"))
        model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(), activation="tanh"))

        
        #Verbesserung 2 Dropoutlayer
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(Dense(4, activation="softmax"))


        # train the model using SGD
        # print("[INFO] training network...")
        sgd = SGD(0.01)
        numberOfEpochs = 50
        model.compile(loss="categorical_crossentropy", optimizer=sgd,
                metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
        H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                epochs=numberOfEpochs, batch_size=128)
        #ursprünglich 100 epochen

        # evaluate the network
        # print("[INFO] evaluating network...")
        predictions = model.predict(x_test, batch_size=128)
        """
        print(classification_report(y_test.argmax(axis=1),
                predictions.argmax(axis=1),
                target_names=[str(x) for x in lb.classes_]))
        """

        lossVec[i]          = H.history["loss"][-1]
        accuracyVec[i]      = H.history["accuracy"][-1]
        val_lossVec[i]      = H.history["val_loss"][-1]
        val_accuracyVec[i]  = H.history["val_accuracy"][-1]
        precisionVec[i]     = H.history["precision"][-1]
        val_precisionVec[i] = H.history["val_precision"][-1]
        recallVec[i]        = H.history["recall"][-1]
        val_recallVec[i]    = H.history["val_recall"][-1]

        #print("ACCUARCY TEST PPRINT\n"+str(H.history["accuracy"])+"\n\n\n")
        
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, numberOfEpochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, numberOfEpochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, numberOfEpochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, numberOfEpochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("Aufgabe1_"+str(numberOfEpochs)+"Epo_D"+str(i+1)+".png")
        
    print("loss der "+str(numberOfEpochs)+" Epochen: "+ str(lossVec) + "  ~~~~~~~> "+str(np.mean(lossVec)))
    print("val_loss der "+str(numberOfEpochs)+" Epochen: "+ str(val_lossVec) + "  ~~~~~~~> "+str(np.mean(val_lossVec)))
    print("accuracy der "+str(numberOfEpochs)+" Epochen: "+ str(accuracyVec) + "  ~~~~~~~> "+str(np.mean(accuracyVec)))
    print("val_accuracy der "+str(numberOfEpochs)+" Epochen: "+ str(val_accuracyVec) + "  ~~~~~~~> "+str(np.mean(val_accuracyVec)))
    print("precision der "+str(numberOfEpochs)+" Epochen: "+ str(precisionVec) + "  ~~~~~~~> "+str(np.mean(precisionVec)))
    print("val_precision der "+str(numberOfEpochs)+" Epochen: "+ str(val_precisionVec) + "  ~~~~~~~> "+str(np.mean(val_precisionVec)))
    print("recall der "+str(numberOfEpochs)+" Epochen: "+ str(recallVec) + "  ~~~~~~~> "+str(np.mean(recallVec)))
    print("val_recall der "+str(numberOfEpochs)+" Epochen: "+ str(val_recallVec) + "  ~~~~~~~> "+str(np.mean(val_recallVec)))
    
    return H



if(__name__ == "__main__"):
    startTime = time.time()

    print("Main beginnt")
    """
    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getData()
    
    Durchläufe = 3
    test_loss = np.zeros(Durchläufe)
    test_acc = np.zeros(Durchläufe)
    training_loss = np.zeros(Durchläufe)
    training_acc = np.zeros(Durchläufe)
    before = []
    
    #TODO BEFORE LISTE FÜLLEN
    #hier schleifenbeginn
    
    x_train, y_train, x_test, y_test, before  =  \
             reading_splitting_dataset_functions.train_test_split_cross_validation(fid_roi, df_roi, l_roi, before)

    x_train, y_train, x_test, y_test  =  \
             reading_splitting_dataset_functions.bring_in_right_shape_self(x_train, y_train, x_test, y_test)

    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getTruncatedData(2.0, 2.0)
    print("len(v_roi)=",len(v_roi))
    """




    """
    #Generiert plot der Geschwindigkeiten
    sorted_v_roi=np.sort(v_roi)
    plt.plot(sorted_v_roi, label='Messdaten', color='blue')
    plt.ylabel('Geschwindigkeiten')
    plt.axhline(y=np.mean(sorted_v_roi), xmin=0, xmax=len(sorted_v_roi), label='Durchschnittsgeschwindigkeit '+str(np.mean(sorted_v_roi)), color='red')
    plt.grid()
    plt.legend()
    plt.show()
    """
    
    print('seqNet Anfang')
    #ALLE DATEN
    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getData()

    #TRUNCATED DATA
    #df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getTruncatedData(0.1)

    
    H = seqNet(df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi)
    print('seqNet Ende')

    endTime = time.time()
    print("ExecTime: ", endTime - startTime)
    
    
