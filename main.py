import cv2
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x = np.load("image.npz")
y = pd.read_csv("labels.csv")
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=3500, test_size=500)
x_trainscale = x_train/255.0
x_testscale = x_test/255.0
lr = LogisticRegression().fit(x_trainscale, y_train)
something = lr.predict(x_testscale)
acc = accuracy_score(y_test, something)
print(acc)