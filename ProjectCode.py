import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
from sklearn.preprocessing import PolynomialFeatures, label_binarize


df = pd.read_csv("/Users/Eddy/Desktop/Machine Learning/Project/GeoData.csv", header=None, comment='#')
X1 = df.iloc[:,0] #Distance
X2 = df.iloc[:,1] #Time
X3 = df.iloc[:,2] #Avg Speed
X4 = df.iloc[:,3] #Min Speed
X5 = df.iloc[:,4] #Max Speed
y = df.iloc[:,5]

X1Mean = np.mean(X1)
X1Std = np.std(X1)
X1Norm = (X1 - X1Mean)/X1Std

X2Mean = np.mean(X2)
X2Std = np.std(X2)
X2Norm = (X2 - X2Mean)/X2Std

X = np.vstack([X1Norm, X2Norm, X3, X4, X5]).T
splitSize=10

#Finding best number of splits
splits = [2,5,10,25,50,100]
mean_error=[]; std_error=[]; temp=[]; scoreTemp=[]; avgScore=[]

for n in splits:
    kf = KFold(n_splits=n, shuffle=True)
    temp=[]; scoreTemp=[]
    for train, test in kf.split(X):
        model = LogisticRegression(penalty='l2', max_iter=1000, C=0.01)
        model.fit(X[train], y[train])
        yPred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],yPred))
        scoreTemp.append(model.score(X[test],y[test]))
    avgScore.append(np.array(scoreTemp).mean())
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
print(avgScore)
print(mean_error)
print(std_error)

print('\n','Support Vector Machine')
#Finding the best C value
C_range = [0.00001,0.0001,0.001,0.01]
kf = KFold(n_splits=splitSize, shuffle=True)
mean_error=[]; bestC=[]; temp=[]

for l in range(50):
    avgScore=[]
    for cTemp in C_range:
        temp=[];
        for train, test in kf.split(X):
            model = LinearSVC(penalty='l2', max_iter=10000, C=cTemp)
            model.fit(X[train], y[train])
            yPred = model.predict(X[test])
            temp.append(model.score(X[test],y[test]))
        avgScore.append(np.array(temp).mean())
    bestC.append(avgScore.index(max(avgScore)))
print(bestC)

print('\n','Logistic Regression')
#Finding best C
C_range = [0.0001,0.001,0.01,0.1,1,10,100]
kf = KFold(n_splits=splitSize, shuffle=True)
bestC=[]; temp=[]

for l in range(50):
    avgScore=[]
    for cTemp in C_range:
        temp=[];
        for train, test in kf.split(X):
            model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=cTemp)
            model.fit(X[train], y[train])
            yPred = model.predict(X[test])
            temp.append(model.score(X[test],y[test]))
        avgScore.append(np.array(temp).mean())
    bestC.append(avgScore.index(max(avgScore)))
print(avgScore)
print(bestC)

#Plotting the scores of different C values for Logistic Regression model
std_error=[]; temp=[]; scoreTemp=[]; avgScore=[]
fig = plt.figure(figsize=(14,7))
for cTemp in C_range:
    temp=[]; scoreTemp=[]
    for train, test in kf.split(X):
        model = LogisticRegression(penalty='l2', max_iter=1000, C=cTemp, solver='lbfgs')
        model.fit(X[train], y[train])
        yPred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],yPred))
        scoreTemp.append(model.score(X[test],y[test]))
    avgScore.append(np.array(scoreTemp).mean())
    std_error.append(np.array(temp).std())
print(avgScore)
plt.errorbar(C_range,avgScore,yerr=std_error,linewidth=3)
plt.xlabel('C', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.title('Graph 1 - Average Score (Accuracy) with varying C; LR Model', fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.show()

print('\n','Decision Tree')
X = np.column_stack((X4,X5))
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
TreeModel = tree.DecisionTreeClassifier()
TreeModel.fit(Xtrain,ytrain)
print(TreeModel.score(Xtest,ytest))
fig = plt.figure(figsize=(20,10))
tree.plot_tree(TreeModel)

print('\n','kNN Model')
#Finding best k
kRange=[3,5,7,9,11,13,15,17]
for p in kRange:
    temp=[]
    for i in range(100):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True)
        neigh = KNeighborsClassifier(n_neighbors=p)
        neigh.fit(Xtrain, ytrain)
        temp.append(neigh.score(Xtest,ytest))
    print(p, ' = ', np.array(temp).mean())

#Building models and comparing accuracies
LRTemp=[];SVMTemp=[];TreeTemp=[];kNNTemp=[];FreTemp=[];StraTemp=[]
for i in range(100):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True)

    LRModel = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=0.01)
    LRModel.fit(Xtrain, ytrain)
    SVMModel = LinearSVC(penalty='l2', C=0.01, max_iter=10000)
    SVMModel.fit(Xtrain, ytrain)
    TreeModel = tree.DecisionTreeClassifier()
    TreeModel.fit(Xtrain,ytrain)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(Xtrain, ytrain)
    FreModel = DummyClassifier(strategy='most_frequent')
    FreModel.fit(Xtrain, ytrain)
    StraModel = DummyClassifier(strategy='stratified')
    StraModel.fit(Xtrain, ytrain)
    
    LRTemp.append(LRModel.score(Xtest,ytest))
    SVMTemp.append(SVMModel.score(Xtest,ytest))
    TreeTemp.append(TreeModel.score(Xtest,ytest))
    kNNTemp.append(neigh.score(Xtest,ytest))
    FreTemp.append(FreModel.score(Xtest,ytest))
    StraTemp.append(StraModel.score(Xtest,ytest))
    
print('LR Accuracy = ', np.array(LRTemp).mean())
print('SVM Accuracy = ', np.array(SVMTemp).mean())
print('DT Accuracy = ', np.array(TreeTemp).mean())
print('kNN Accuracy = ', np.array(kNNTemp).mean())
print('MF Accuracy = ', np.array(FreTemp).mean())
print('Stra Accuracy = ', np.array(StraTemp).mean())

#Confusion Matrices
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
X = np.column_stack((X3,X5))

print('\n','Logistic Regression Matrix')
LRModel = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=0.01)
LRModel.fit(Xtrain, ytrain)
yPredLR = LRModel.predict(Xtest)
print(pd.crosstab(ytest, yPredLR, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n','SVM Matrix')
SVMModel = LinearSVC(penalty='l2', C=0.01, max_iter=10000)
SVMModel.fit(Xtrain, ytrain)
yPredSVM = SVMModel.predict(Xtest)
print(pd.crosstab(ytest, yPredSVM, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n','Decision Tree Matrix')
TreeModel = tree.DecisionTreeClassifier()
TreeModel.fit(Xtrain,ytrain)
yPredTree = TreeModel.predict(Xtest)
print(pd.crosstab(ytest, yPredTree, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n', 'kNN Matrix')
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(Xtrain, ytrain)
yPredNeigh = neigh.predict(Xtest)
print(pd.crosstab(ytest, yPredNeigh, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n', 'Most Frequent Predictor Matrix')
FreModel = DummyClassifier(strategy='most_frequent')
FreModel.fit(Xtrain, ytrain)
yPredFre = FreModel.predict(Xtest)
print(pd.crosstab(ytest, yPredFre, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n', 'Stratified Predictor Matrix')
StraModel = DummyClassifier(strategy='stratified')
StraModel.fit(Xtrain, ytrain)
yPredStra = StraModel.predict(Xtest)
print(pd.crosstab(ytest, yPredStra, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n')

#Results
print('Regression Report')
print(metrics.classification_report(ytest, yPredLR, digits=4))
print('\n', 'SVM Report')
print(metrics.classification_report(ytest, yPredSVM, digits=4))
print('\n', 'Tree Report')
print(metrics.classification_report(ytest, yPredTree, digits=4))
print('\n', 'kNN Report')
print(metrics.classification_report(ytest, yPredNeigh, digits=4))
print('\n', 'Frequent Report')
print(metrics.classification_report(ytest, yPredFre, digits=4))
print('\n', 'Stratified Report')
print(metrics.classification_report(ytest, yPredStra, digits=4))


#Plotting the ROC curves
y = label_binarize(y, classes=[1,2,3,4])
n_classes = 4; lw=3
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
fprLR = dict(); tprLR = dict(); roc_aucLR = dict()
fprSVM = dict(); tprSVM = dict(); roc_aucSVM = dict()
fprkNN = dict(); tprkNN = dict(); roc_auckNN = dict()
fprTree = dict(); tprTree = dict(); roc_aucTree = dict()
fprStra = dict(); tprStra = dict(); roc_aucStra = dict()
fprFre = dict(); tprFre = dict(); roc_aucFre = dict()

LRmodel = OneVsRestClassifier(LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=1000))
LRScore = LRmodel.fit(Xtrain, ytrain).decision_function(Xtest)
SVMmodel = OneVsRestClassifier(LinearSVC(penalty='l2', C=0.01, max_iter=10000))
SVMScore = SVMmodel.fit(Xtrain, ytrain).decision_function(Xtest)
neigh = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
kNNScore = neigh.fit(Xtrain, ytrain).predict_proba(Xtest)
TreeModel = OneVsRestClassifier(tree.DecisionTreeClassifier())
TreeScore = TreeModel.fit(Xtrain,ytrain).predict_proba(Xtest)
StraModel = OneVsRestClassifier(DummyClassifier(strategy='stratified'))
StraScore = StraModel.fit(Xtrain, ytrain).predict_proba(Xtest)
FreModel = OneVsRestClassifier(DummyClassifier(strategy='most_frequent'))
FreScore = FreModel.fit(Xtrain, ytrain).predict_proba(Xtest)

for i in range(n_classes):
    fprLR[i], tprLR[i], _ = roc_curve(ytest[:,i], LRScore[:,i])
    roc_aucLR[i] = auc(fprLR[i], tprLR[i])
    fprSVM[i], tprSVM[i], _ = roc_curve(ytest[:,i], SVMScore[:,i])
    roc_aucSVM[i] = auc(fprSVM[i], tprSVM[i])
    fprkNN[i], tprkNN[i], _ = roc_curve(ytest[:,i], kNNScore[:,i])
    roc_auckNN[i] = auc(fprkNN[i], tprkNN[i])
    fprTree[i], tprTree[i], _ = roc_curve(ytest[:,i], TreeScore[:,i])
    roc_aucTree[i] = auc(fprTree[i], tprTree[i])
    fprStra[i], tprStra[i], _ = roc_curve(ytest[:,i], StraScore[:,i])
    roc_aucStra[i] = auc(fprStra[i], tprStra[i])
    fprFre[i], tprFre[i], _ = roc_curve(ytest[:,i], FreScore[:,i])
    roc_aucFre[i] = auc(fprFre[i], tprFre[i])
    
all_fprLR = np.unique(np.concatenate([fprLR[i] for i in range(n_classes)]))
mean_tprLR = np.zeros_like(all_fprLR)
all_fprSVM = np.unique(np.concatenate([fprSVM[i] for i in range(n_classes)]))
mean_tprSVM = np.zeros_like(all_fprSVM)
all_fprkNN = np.unique(np.concatenate([fprkNN[i] for i in range(n_classes)]))
mean_tprkNN = np.zeros_like(all_fprkNN)
all_fprTree = np.unique(np.concatenate([fprTree[i] for i in range(n_classes)]))
mean_tprTree = np.zeros_like(all_fprTree)
all_fprStra = np.unique(np.concatenate([fprStra[i] for i in range(n_classes)]))
mean_tprStra = np.zeros_like(all_fprStra)
all_fprFre = np.unique(np.concatenate([fprFre[i] for i in range(n_classes)]))
mean_tprFre = np.zeros_like(all_fprFre)

for i in range(n_classes):
    mean_tprLR   += np.interp(all_fprLR, fprLR[i], tprLR[i])
    mean_tprSVM  += np.interp(all_fprSVM, fprSVM[i], tprSVM[i])
    mean_tprkNN  += np.interp(all_fprkNN, fprkNN[i], tprkNN[i])
    mean_tprTree += np.interp(all_fprTree, fprTree[i], tprTree[i])
    mean_tprStra += np.interp(all_fprStra, fprStra[i], tprStra[i])
    mean_tprFre  += np.interp(all_fprFre, fprFre[i], tprFre[i])
mean_tprLR   /= n_classes
mean_tprSVM  /= n_classes
mean_tprkNN  /= n_classes
mean_tprTree /= n_classes
mean_tprStra /= n_classes
mean_tprFre  /= n_classes

fprLR["micro"], tprLR["micro"], _ = roc_curve(ytest.ravel(), LRScore.ravel())
roc_aucLR["micro"] = auc(fprLR["micro"], tprLR["micro"])
fprSVM["micro"], tprSVM["micro"], _ = roc_curve(ytest.ravel(), SVMScore.ravel())
roc_aucSVM["micro"] = auc(fprSVM["micro"], tprSVM["micro"])
fprkNN["micro"], tprkNN["micro"], _ = roc_curve(ytest.ravel(), kNNScore.ravel())
roc_auckNN["micro"] = auc(fprkNN["micro"], tprkNN["micro"])
fprTree["micro"], tprTree["micro"], _ = roc_curve(ytest.ravel(), TreeScore.ravel())
roc_aucTree["micro"] = auc(fprTree["micro"], tprTree["micro"])
fprStra["micro"], tprStra["micro"], _ = roc_curve(ytest.ravel(), StraScore.ravel())
roc_aucStra["micro"] = auc(fprStra["micro"], tprStra["micro"])
fprFre["micro"], tprFre["micro"], _ = roc_curve(ytest.ravel(), FreScore.ravel())
roc_aucFre["micro"] = auc(fprFre["micro"], tprFre["micro"])

# Plot all ROC curves
plt.figure(figsize=(14,7))
plt.plot(fprLR["micro"], tprLR["micro"],
         label='Logistic Regression (area = {0:0.2f})'
               ''.format(roc_aucLR["micro"]),
         color='deeppink', linestyle=':', linewidth=lw)
plt.plot(fprSVM["micro"], tprSVM["micro"],
         label='SVM (area = {0:0.2f})'
               ''.format(roc_aucSVM["micro"]),
         color='dodgerblue', linestyle='-', linewidth=lw)
plt.plot(fprkNN["micro"], tprkNN["micro"],
         label='kNN (area = {0:0.2f})'
               ''.format(roc_auckNN["micro"]),
         color='chartreuse', linestyle='--', linewidth=lw)
plt.plot(fprTree["micro"], tprTree["micro"],
         label='Decision Tree (area = {0:0.2f})'
               ''.format(roc_aucTree["micro"]),
         color='gold', linestyle='-.', linewidth=lw)
plt.plot(fprStra["micro"], tprStra["micro"],
         label='Stratified Baseline (area = {0:0.2f})'
               ''.format(roc_aucStra["micro"]),
         color='dimgrey', linestyle=':', linewidth=lw)
plt.plot(fprFre["micro"], tprFre["micro"],
         label='Most Frequent Baseline (area = {0:0.2f})'
               ''.format(roc_aucFre["micro"]),
         color='orangered', linestyle='-', linewidth=lw)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Micro-Averaged ROC Curves', fontsize=14)
plt.legend(loc="lower right")
plt.grid()
plt.show()