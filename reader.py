#!/usr/bin/python

import aifc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
import numpy as np
import preprocess






def drawFrames(frames, n=7):
    ctr = 0
    f, axarr = plt.subplots(n, n)
    for ii in range(0,n):
        for jj in range(0,n):
            axarr[ii, jj].plot( frames[ctr] )
            ctr += 1
    plt.show()


def transform( features, labels ):

#    for ff, ll in zip(features, labels):
#        print ll, ff
#    for rr in range(0, len(features) ):
#        features[rr] = scaler.fit_transform( features[rr] )

    print "transforming features via pca"
    pca = PCA(n_components = 30)
    features = pca.fit_transform( features )

    envelope = EllipticEnvelope()
    envelope.fit( features )
    print envelope.predict( features )

    scaler = MinMaxScaler()
    features = scaler.fit_transform( features )



    return features, labels



def classify( features, labels ):

    clf = svm.SVC(kernel="linear")#DecisionTreeClassifier()
    kf  = KFold(len(labels), n_folds=4, shuffle=True)

    for train_index, test_index in kf:
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        clf.fit( features_train, labels_train )
        pred = clf.predict( features_test )
        print "training labels: ", labels_train
        print "predictions: ", pred
        print "test labels: ", labels_test
        acc = accuracy_score( pred, labels_test )
        print acc
    print "fitting classifier"
    print "classification done"





def main():
    #raw    = aifc.open("/Users/katiemalone/Desktop/data_podcast/fisherfaces/fisherfaces.aif", "r")
    #edited = aifc.open("/Users/katiemalone/Desktop/data_podcast/fisherfaces/fisherfaces_edited.aif", "r")
    raw    = aifc.open("fisherfaces.aif", "r")
    edited = aifc.open("fisherfaces_edited.aif", "r")

    print raw.getparams()

    s_markers, b_markers = preprocess.signalBackgroundMarkers( raw ) 
    window_length = preprocess.findMaxWindow(s_markers)

    s_frames = preprocess.normedFrames( raw, window_length, start_points=s_markers)
#    drawFrames( s_frames )


    b_frames = preprocess.normedFrames( edited, window_length, n_samples=200 )
#    drawFrames( b_frames )

    features = []
    labels = []
    for ii in range(0, len(s_frames)):
        features.append(s_frames[ii])
        labels.append(1)
    for jj in range(0, len(b_frames)):
        features.append(b_frames[jj])
        labels.append(0)
    print "data acquisition done"

    features = np.asarray( features )
    labels = np.asarray( labels )

    features, labels = transform( features, labels )


#    f, axarr = plt.subplots(2, 2)
#    ctr = 0
    for ff, ll in zip(features, labels):
        if ll==0:
            #axarr[0, 0].scatter( ff[0], ff[1], color="red")
            plt.scatter( ff[0], ff[1], color="red")
        if ll==1:
            plt.scatter( ff[0], ff[1], color="blue")
#    for ii in range(0,n):
#        for jj in range(0,n):
#            axarr[ii, jj].plot( frames[ctr] )
#            ctr += 1
    plt.show()



    classify( features, labels )



if __name__=="__main__":
    main()

