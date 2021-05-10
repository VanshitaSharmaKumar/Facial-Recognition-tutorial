import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

##Helper functions. Use when needed. 
def show_orignal_images(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(6, 10, figsize=(11, 7),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
	plt.show()

def show_eigenfaces(pca):
	#Displaying Eigenfaces
	fig, axes = plt.subplots(3, 8, figsize=(9, 4),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
	    ax.set_title("PC " + str(i+1))
	plt.show()

#PCA: aiming to preserve variance in the data
## Step 1: Read dataset and visualize it.
df = pd.read_csv("face_data.csv")
#print(df.head())
labels = df["target"] #seperating labels and features
pixels = df.drop(["target"],axis = 1) #dropping the column "target"

#show_orignal_images(pixels)

## Step 2: Split Dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(pixels,labels)
#x=pixels, and y=labels:x_train and x_test will have our features. y_train and y_test will have our targets/labels

## Step 3: Perform PCA. Using matplotlib to plot the data
pca = PCA(n_components=200).fit(x_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_)) #sum is getting captured as we keep increasing the number of components

#plt.show() # showing the data
#print(show_eigenfaces(pca)) # which direction we have the maximum variation in the data. What is being highlghted are the specific facial features.

## Step 4: Project Training data to PCA
#getting the training data and tranforming it in PCA
x_train_pca = pca.transform(x_train) #transforming the data

##############

## Step 5: Initialize Classifer and fit training data
clf = SVC(kernel='rbf',C=1000,gamma=0.001)  #parameters from sklearn
# support vector classifier
#kernal = non-linear kernal
# C = regulairzation parameter, telling the clasifeir how muh erroer it allows and not allows
# gamma = the smoothness of the code the classifer prints.
#changing parameters change the classifiers performance

clf = clf.fit(x_train_pca, y_train)

## Step 6: Perform testing and get classification report
x_test_pca = pca.transform(x_test) #trnforming x_test
y_pred = clf.predict(x_test_pca)
print(classification_report(y_test,y_pred))
# the prediction we are trying to predict. We predict
# final output: from 1000+ images, it is reduced down to 39 faces.
