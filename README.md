# PR

#### 1 a. Implement a Decision Tree

In [1] from sklearn import tree  
In [2] clf = tree. DecisionTreeClassifier()  
In [3] # height, hair length, voice pitch   
X = [[180, 15, 0],  
[167, 42, 1],  
[136, 35, 1],  
[174, 15, 0],  
[141, 28, 1]]  
In [4] Y = [ 'Man', 'Woman', 'Woman', 'Man', 'Woman']  
In [5]: dtclf = clf.fit(X,Y)  
In [6]: prediction =dtclf.predict([[133,37,1]])  
In [7]: print (prediction)  
['Woman']  
In [8] from matplotlib import pyplot as plt

#### 1b. Visualise decision tree  
In [8]: from matplotlib import pyplot as plt  
In [9] fig= plt.figure(figsize=(4,4), facecolor="white")  
tree.plot_tree(dtclf, feature_names["Height", "Hair length", "Voice pitch"], class_names=["Man", "Woman"], filled=True)

#### 2. Implement SVM
  
In [1] from sklearn.svm import SVC import matplotlib.pyplot as plt  
In [2]: clf = SVC()  
In [3] # height, hair length, voice pitch   
X = [[180, 15, 0],  
[167, 42, 1],  
[136, 35, 1],  
[174, 15, 0],  
[141, 28, 1]]  
In [4] Y = ['Man', 'Woman', 'Woman', 'Man', 'Woman']  
In [5] svcClassifier = clf.fit(X,Y)  
In [6] prediction = svcClassifier.predict([[133,37,1]])  
In [7] print (prediction)
  
  ##### MNIST Dataset
    
    In [8]: from sklearn.datasets import load_digits  
    from sklearn.model_selection import train_test_split  
    from sklearn.preprocessing import StandardScaler  
    from sklearn.pipeline import Pipeline  
    import seaborn as sns  
    from sklearn.metrics import classification_report, confusion_matrix  
    In [9]: x, y = load_digits (return_X_y=True)  
    In [10]: x_train, x_test, y_train, y_test train_test_split(x, y, test_size=0.2, train_size=0.8)  
    In [11]: steps [('scaler', StandardScaler()), ('SVM', SVC())]  
    In [12]: pipeline=Pipeline (steps)  
    In [13]: svcClassifier=pipeline.fit(x_train, y_train)  
    In[14]:print(classification_report(y_test,svcClassifier.predict(x_test)))  
    In [15]: train_ace float (svcClassifier.score (x_train, y_train)*100) print("train accuracy score:", train_acc)  
    test_acc float (svcClassifier.score (x_test, y_test)*100) print("test accuracy score:", test_acc)  
    In [16] cf matrix=confusion_matrix(y_test, svcClassifier.predict(x_test)) 
             print (cf matrix)  
    In [17] fig = plt.figure(figsize=(7,7), facecolor="white")  
    cf_matrix=confusion_matrix(y_test, svcClassifier.predict(x_test))  
    sns.heatmap(cf_matrix, annot=True)  
    plt.show()

#### 3. Implement Agglomerative Hierarchical Clustering.  
In [1]: import numpy as np  
import matplotlib.pyplot as plt  
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.cluster import AgglomerativeClustering  
In [2] x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]  
       y=[21, 19, 24, 17, 16, 25, 24, 22, 21, 21]  
       
In [3] data = list(zip(x, y))  
In [4] linkage_data= linkage (data, method='ward', metric='euclidean')   
dendrogram (linkage_data) 
plt.show()  
In [5] hierarchical_cluster=AgglomerativeClustering (n_clusters=2, affinity='euclidean’, linkage='ward')  
labels=hierarchical_cluster.fit_predict(data)  
In [6]: print(labels)  
In (7) plt.scatter=(x, y, c=labels, cmap=’coolwarm')  
plt.show()
       
       
  #### 4. Write a program to implement k-means Clustering from scratch.  
In [1]: import pandas as pd   
import numpy as np   
import matplotlib.pyplot as plt  
In [2]: data = pd. read_excel (" K Means Clustering.xIsx")  
In [3]: data.head()  
In [4]: pit.style.use("seaborn")  
plt. figure (0)  
plt. scatter (data [ 'Age' ], data[ ' Income' ]) plt.show ()  
In [5]: X1 = data [ "Age" ] .values  
X2 = data [ "Income" ] .values
  
XI = (X1-np.mean (X1) ) /np. std(X1 )  
X2 = (X2-np. mean (X2) ) /np. std (X2)  

X = np. column_stack ( (X1, X2) )  
print (X. shape)    

In (6]: plt. figure(0)  
pit.scatter (XI,X2)  
plt.show ()  

1. Random Initialisation  

In [7): k = 3 # specifying how many clusters we want to have  
colour= ["green", "red", "purple") # specifying colour for each of the cluster  
clusters = {} # dictionary of all the clusters  

for i in range(k):  

#initialising one random center for each cluster in the range (-1.5,1.5)  
center • 1.5* (2*np.random.random( (X.shape (1], ))-1) # generating one data point having 2 features  
#we also need an assignment list (so with each cluster center, we associate a list)  
points = [ ] # contains points assigned to / associated with that cluster  
#for every cluster we create one dictionary (this is the ith cluster)  
cluster = {  
'center': center,  
'points': points,  
'colour': colour [i]  
}  
clusters[i] - cluster  

In [8]: clusters  

2. Implementing E Step  

In [9]: # helper function to compute distance between 2 vectors or points (we use euclidean distance)def distance (v1,v2):
return np.sqrt (np.sum( (v1-v2)** 2))  

In [10]: def assignPointToClusters (clusters) :  
#iterating over all the training data points  
for ix in range (X. shape[0]):  
dist = [ ]  
curr x = X[ix] # current data point  

#computing the distance of current data point from the center of the kth cluster  
for kx in range (k):  
d = distance (curr_ x, clusters [kx][ 'center ' ])  
dist.append(d)  

#choosing the cluster which has the minimum distance  
current cluster = np.argmin(dist)  
clusters[current_cluster]['points'].append(curr_x)  
return  
In (11]: assignPointToClusters (clusters)  
In [12]: clusters  
In [13]: #function to see if we are able to plot the cluster centers and the points associated with them  
def plotCluster (clusters):  
#iterate over every cluster  
for kx in range (k):  

#converting the list to numpy array so that we can give it to matplotlib function 
pts = np.array (clusters [kx]['points' ])  

#PLOTTING THE POINTS  
try:  
pit.scatter(pts[:,0],pts[:,1],color=clusters[kx]['colour'])  
except:  
pass 

#PLOTTING THE CLUSTER CENTERS  
uk = clusters [kx][ 'center']  
plt.scatter(uk[0],uk[1],color='black' ,marker='*’)  

In [14]: plotCluster (clusters)  

3. Implementing M Step  


In [15]: def updateClusters (clusters) :  
for kx in range (k):  
pts = np.array (clusters [kx][ 'points' ])  
#if a cluster has non-zero points, we will take the mean and update that cluster  
if pts.shape[0]>0:  
new_uk = pts.mean (axis=0) # new cluster center  
clusters [kx]( 'center' ] = new_uk # updating the center  
#emptying the list of points of that cluster as will repeat E-Step to get the points  
clusters [kx]['points'| = [ ]   

4. Repeating E and M steps again and again until there is no change in the value of cluster centers  
In [16]: while(1):    
cnt = 0    
prev_center = []  
for ix in range (k):  
prev_center.append(clusters [ix]['center' ])  
assignPointToClusters (clusters)  
updateClusters(clusters)  

for ix in range (k):  
1f(all (prev_center[ix]==clusters[ix]['center' ])):  
cnt+=1  

if (cnt==k):  
break  

assignPointToClusters(clusters)  
plotCluster (clusters)


#### 5. Implement Maximum-Likelihood Estimation  
In [1]: import numpy as np  
from scipy.stats import expon  
import matplotlib.pyplot as plt  
In [2]: #population with exponential distribution  
population_rate=3 sample_size = 100  
get_sample=lambda n: np.random.exponential (population_rate, n)  
xs= np.arange(0, 20, 0.001)  
ys = expon.pdf (xs, scale=population_rate)  
In [3]: plt.plot(xs, ys, label='population')  
sample = get_sample (sample_size)  
plt.hist (sample, density=True, label='sample')  
plt.legend()  
plt.show()  

In [4]: #estimate λ(rate) parameter of the actual population by having a sample from this population  
log_likelihood=lambda rate: sum([np.log(expon.pdf (v, scale=rate)) for v in sample])  
rates=np.arange(1, 8, 0.01)  
estimates=[log_likelihood(r) for r in rates]  
plt.xlabel('parameter')  
plt.plot(rates, estimates)  
print('parameter value:’, rates [estimates.index (max (estimates))])

#### 6. Implement Principal Component Analysis and use it for unsupervised learning.  
In [1] import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.datasets import load_breast_cancer   
from sklearn.preprocessing import StandardScaler   
from sklearn.decomposition import PCA  
In [2] cancer = load_breast_cancer()  
df = pd.DataFrame (cancer ['data'], columns=cancer['feature_names'])  
df.head()  
In [3]: scalar=StandardScaler()   
scalar.fit(df)  
scaled_data = scalar.transform(df)  
In [4]: pca=PCA (n_components = 2)   
pca.fit(scaled_data)  
x_pca=pca.transform(scaled_data)  
x_pca.shape  
In [5]: plt.scatter (x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='coolwarm')  
plt.xlabel('First Principal Component')  
plt.ylabel('Second Principal Component')  
plt.show()  
In[6]: pca.components


# END










