import numpy as np
import json
import reading_splitting_dataset_functions
import time
import tensorflow as tf

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

if(__name__ == "__main__"):
    startTime = time.time()

    print("Main beginnt")
    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getData()
    
    Durchläufe = 3
    test_loss = np.zeros(Durchläufe)
    test_acc = np.zeros(Durchläufe)
    training_loss = np.zeros(Durchläufe)
    training_acc = np.zeros(Durchläufe)
    before = []

    #hier schleifenbeginn
    
    x_train, y_train, x_test, y_test, before  =  \
             reading_splitting_dataset_functions.train_test_split_cross_validation(fid_roi, df_roi, l_roi, before)

    x_train, y_train, x_test, y_test  =  \
             reading_splitting_dataset_functions.bring_in_right_shape_self(x_train, y_train, x_test, y_test)

    df_roi, fid_roi, v_roi, lva_roi, lha_roi, l_roi = getTruncatedData(2.0, 2.0)
    print("len(v_roi)=",len(v_roi))






    #Generiert plot der Geschwindigkeiten
    import matplotlib.pyplot as plt
    sorted_v_roi=np.sort(v_roi)
    plt.plot(sorted_v_roi, label='Messdaten', color='blue')
    plt.ylabel('Geschwindigkeiten')
    plt.axhline(y=np.mean(sorted_v_roi), xmin=0, xmax=len(sorted_v_roi), label='Durchschnittsgeschwindigkeit '+str(np.mean(sorted_v_roi)), color='red')
    plt.grid()
    plt.legend()
    plt.show()

    #baut unser Modell
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1536,)))
    model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu'))
    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(4))
        #https://keras.io/api/layers/core_layers/dense/
    	#tf.keras.layers.Flatten(input_shape=(DIM, DIM)),
    	#kernel_regularizer=tf.keras.regularizers.l2()
    	#tf.keras.layers.Dropout(DROPOUT_RATE),
    print("Modell gebaut")
    model.summary()
    
    #definiert unsere Lossfunktion
    loss_fn= tf.keras.losses.MeanAbsoluteError()  	#https://keras.io/api/losses/
    print("Lossfunktion definiert")

    #Compillierung unseres Modells
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        #https://keras.io/api/optimizers/
    print("Modell compilliert")

    
    #Visualisierung im Browser
    # trainingSurface = tfvis.visor().surface(tab: modelType, name: ’Model Training’)	#syntax aus JS kopiert
    # my_callbacks    = tfvis.show.fitCallbacks(	#syntax aus JS kopiert
    #			trainingSurface,
    #			[’loss’, ’val loss’], 
    #			{callbacks: [’onBatchEnd’, ’onEpochEnd’]}
    #			)

    # Modelltraining			
    model.fit(x_train, y_train, epochs=20)
        #, callbacks=my_callbacks
    print("Modell trainiert")

    model.evaluate(x_test, y_test, verbose=1) #verbose: 0=silent, 1=progress bar
    

    endTime = time.time()
    print("ExecTime: ", endTime - startTime)
    
    
