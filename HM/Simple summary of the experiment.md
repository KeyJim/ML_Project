# Simple summary of the experiment



---

# 1.Label

* Manually tag each waveform
* To calculate Easily, label value 0-4 corresponds to 140-148


# 2. Import Data
```python
dframe = pd.read_excel("test.xlsx")
test = DataFrame(dframe)
test=test.T
X = test.iloc[ : , :-1].values 
#X = finalData  # Using PCA compressed data, the test results are not ideal
Y = test.iloc[ : , 302].values 
```

# 3. Data Preprocessing

### 1. For Multi-layer Neural Network
```python

# Using sklearn to split dataset 1:4
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# shape each y to [0,0,1,0,0]....
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

```
### 2. For other Classifiers
> Random tree, random forest, SVM...

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
```


# 4. Training model

### 1. Multi-layer Neural Network

```python
# 3-layers
model = Sequential()
model.add(Dense(90, activation='relu', input_dim=302))
model.add(Dense(30, activation='relu'))
model.add(Dense(5, activation='softmax'))

# using Adam to optumize
rms = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])
# add validation data to reduce overfitting
model.fit(x_train, y_train,validation_split=0.2,
          epochs=300,
          batch_size=10)
score = model.evaluate(x_test, y_test, batch_size=10)
print(model.metrics_names)
print(score)

```

**Results: 98%**

> Can continue to optimize and improve  accuracy.



### 2. Other Classifiers 

* Using tpot package to select models and optimize Automatically 
> A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

```python

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline.py')
```

**Results: 89.15%**

<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/result.png" width="800"/> </div><br>







  
 
* According to this , using SVC(linear) + PCA 
```python
from sklearn.svm import SVC
clf = SVC(C=1,kernel='linear', probability = True,random_state=0)
clf.fit(x_train, y_train)

```

**Results : 90%**

* The number of features is much larger than the number of samples 


# 5. Conclution

* Better to Use Multi-layer Neural Network
* Better to have more data
* PCA  is not useful for this dataset, and may be useful when factor is lager.
  
----

2018-11-10

