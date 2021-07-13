from Kreuzval import StratifiedGroupKFold
from reading_splitting_dataset_functions import *
import numpy as np


###### "Dokumentation" hier #####
# https://scikit-learn.org/dev/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py

###### Daten laden, wie gehabt #####
data_roi=open_js_file('data_preprocessed_roi.JSON')
print('length ROI:', len(data_roi))
df_roi, fid_roi, v_roi, lva_roi, lha_roi = get_acceleration_fid_v_labels(data_roi)
l_roi = labels_roi(lva_roi, lha_roi)
print(l_roi)

##### Splits erzeugen #####

n_samples = len(data_roi)

# Dieses Objekt kann später mit irgendwelchen
# Daten Splits erzeugen:
cv = StratifiedGroupKFold(n_splits=5)

# Hier geben wir dem Objekt die Daten nach dem
# Muster cv.split(X,y,group)
# Es ist egal, was in X steht: nur wegen Kompatibilität drin
# y sind die Klassen, die gleichmäßig in Test und Training auftauchen sollen {1,2,3,4}.
# group sind die Samples, die nicht aufgeteilt werden dürfen.
splits = cv.split(np.zeros([n_samples]), l_roi, fid_roi)

# Anwendung:
for train_idxs, test_idxs in splits:
    x_test = df[train_idxs,:,:]
    x_train = df[train_idxs,:,:]
    y_test = l_roi[test_idxs]
    y_train = l_roi[train_idxs]

    # ... Modelltraining ...