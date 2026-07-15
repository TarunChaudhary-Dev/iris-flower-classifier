import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

x = iris.data
y = iris.target

x_train , x_test , y_train , y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
    )
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(x_train,y_train)

test_ans = KNN.predict(x_test)

accuracy = accuracy_score(y_test,test_ans)

userinput = input("Enter all 4 values sprated by comma: ")
input_list = [float(x) for x in userinput.split(",")]
new_data = np.array([input_list])

result = KNN.predict(new_data)

flowername = result[0]

if flowername == 0:
    print("🌸 Setosa")
elif flowername == 1:
    print("🌺 Versicolor")
elif flowername == 2:
    print("🌹 Virginica")


