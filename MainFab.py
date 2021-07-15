import numpy as np
import json
import reading_splitting_dataset_functions
import time
import matplotlib.pyplot as plt
    

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
def getTruncatedData(erlaubteAbweichungRelativLinks, erlaubteAbweichungRelativRechts):
    print("getTruncatedData() : getData() wird aufgerufen")
    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getData()

    Durchschnitt = np.mean(v_roi)
    l            = erlaubteAbweichungRelativLinks*Durchschnitt
    r            = erlaubteAbweichungRelativRechts*Durchschnitt
    
    IndexListe = []
    for i in range(len(v_roi)):
        if ( (v_roi[i] > Durchschnitt - l) & (v_roi[i] < Durchschnitt + r) ):
            IndexListe.append(i)

    c = len(IndexListe)

    print("Truncation:", Durchschnitt-l, "|", Durchschnitt, "|", Durchschnitt+r, "führt zu ", c, "/", len(v_roi), "Datensätzen")

    return np.ndarray( (c,),buffer=np.array([df_roi[i] for i in IndexListe]) ),\
           np.ndarray( (c,),buffer=np.array([fid_roi[i] for i in IndexListe]) ),\
           np.ndarray( (c,),buffer=np.array([v_roi[i] for i in IndexListe]) ),\
           np.ndarray( (c,),buffer=np.array([lva_roi[i] for i in IndexListe]) ),\
           np.ndarray( (c,),buffer=np.array([lha_roi[i] for i in IndexListe]) ),\
           np.ndarray( (c,),buffer=np.array([l_roi[i] for i in IndexListe]) )

def seqNet():
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import classification_report

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras import backend as K
    #import argparse

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
            help="path to the output loss/accuracy plot")
    args = vars(ap.parse_args())
    """

    # grab the MNIST dataset (if this is your first time using this
    # dataset then the 11MB download may take a minute)
    # print("[INFO] accessing MNIST...")
    #((trainX, trainY), (testX, testY)) = mnist.load_data()

    # each image in the MNIST dataset is represented as a 28x28x1
    # image, but in order to apply a standard neural network we must
    # first "flatten" the image to be simple list of 28x28=784 pixels
    #trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
    #testX = testX.reshape((testX.shape[0], 28 * 28 * 1))
    # scale data to the range of [0, 1]
    #trainX = trainX.astype("float32") / 255.0
    #testX = testX.astype("float32") / 255.0


    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    #trainY = lb.fit_transform(trainY)
    #testY = lb.transform(testY)
    #trainY = lb.fit_transform(trainY)
    #testY = lb.transform(testY)
    
    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getData()
    before = []
    x_train, y_train, x_test, y_test, before  =  \
             reading_splitting_dataset_functions.train_test_split_cross_validation(fid_roi, df_roi, l_roi, before)

    x_train, y_train, x_test, y_test  =  \
             reading_splitting_dataset_functions.bring_in_right_shape_self(x_train, y_train, x_test, y_test)

    # define the 784-256-128-10 architecture using Keras
    model = Sequential()
    #model.add(Dense(256, input_shape=(len(x_train[0]),), activation="sigmoid")) ursprünglich
    model.add(Dense(256, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(4, activation="softmax"))


    # train the model using SGD
    # print("[INFO] training network...")
    sgd = SGD(0.01)
    numberOfEpochs = 100
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
            metrics=["accuracy"])
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
    plt.savefig("PLOTPLOT.png")
    
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
    H= seqNet()
    print('seqNet Ende')

    endTime = time.time()
    print("ExecTime: ", endTime - startTime)
    
    
