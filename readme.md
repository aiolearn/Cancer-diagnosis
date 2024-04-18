# Cancer diagnosis

In this project, we have written a cancer detection program

## Modules

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
```

## Usage
Reading the dataset file

```python
data = pd.read_csv("cancer.csv")
data.head()
data.info()
data.describe()
```

With this command, we delete the two columns that we do not need for the model

```python
data = data.drop(['Unnamed: 32', 'id'], axis = 1)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
```

It gives 1 for benign cancers and 0 for malignant cancers

```python
c = 0
for k in data.diagnosis:
    if k == 'M':
        data.diagnosis[c] = 1
    else:
        data.diagnosis[c] = 0
    c += 1
```

This code first extracts the "diagnosis" column from the data as the response variable "y". Then it converts "y" values to integer data type. Finally, it removes the data without the "diagnosis" column from the "x_data" variable.

```python
y = data.diagnosis

y = y.astype('int')

x_data = data.drop(['diagnosis'], axis = 1)
 
y
```

Normalize the data

```python
d = np.array([0,10,20,100,50,200,40])
d2 = (d - np.min(d)) / (np.max(d) - np.min(d))
print(d2)
```

This code converts the input data to values between 0 and 1 using the min-max scaling formula.

```python
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
```

This code uses the train_test_split function in sklearn to split the input data (x and y) into two training and test sets. The data is divided into 85% for training and 15% for testing. Also, by using the random_state parameter, the data is divided randomly, and by assigning a fixed number to this parameter, it is possible to perform multiple tests with the same data divisions.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)
```

This code creates a logistic regression model using the scikit-learn library and can then apply specific commands to the specified model, such as setting the maximum number of iterations for the optimization algorithm (max_iter) and training the model on the training data (x_train and y_train ) by calling the fit method.

```python
from sklearn import linear_model
robot = linear_model.LogisticRegression()


robot.max_iter=1000000
robot.fit(x_train, y_train)
```

This code performs model predictions on the test data (x_test) and then checks whether the predictions match the actual values by comparing y_test to the predictions. Then it calculates and prints the number of cases where the model's prediction was wrong.

```python
t=robot.predict(x_test)
v= y_test ==t 
v=v.reset_index()
v=v.diagnosis
n=0
for i in v:
    if i==False : n=n+1

print(n,'/',len(v))
```

## Result

This project was written by Majid Tajanjari and the Aiolearn team, and we need your support!❤️  

# تشخیص سرطان

در این پروژه یک برنامه تشخیص سرطان نوشته ایم

## ماژول

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
```

## نحوه استفاده

خواندن فایل دیتاست

```python
data = pd.read_csv("cancer.csv")
data.head()
data.info()
data.describe()
```

با این دستور دو ستونی که برای مدل نیاز نداریم را حذف می کنیم

```python
data = data.drop(['Unnamed: 32', 'id'], axis = 1)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
```

برای سرطان های خوش خیم 1 و برای سرطان های بدخیم 0 می دهد

```python
c = 0
for k in data.diagnosis:
    if k == 'M':
        data.diagnosis[c] = 1
    else:
        data.diagnosis[c] = 0
    c += 1
```

این کد ابتدا ستون "تشخیص" را به عنوان متغیر پاسخ "y" از داده ها استخراج می کند. سپس مقادیر "y" را به نوع داده عدد صحیح تبدیل می کند. در نهایت، داده های بدون ستون "تشخیص" را از متغیر "x_data" حذف می کند.

```python
y = data.diagnosis

y = y.astype('int')

x_data = data.drop(['diagnosis'], axis = 1)
 
y
```

عادی سازی دیتاها

```python
d = np.array([0,10,20,100,50,200,40])
d2 = (d - np.min(d)) / (np.max(d) - np.min(d))
print(d2)
```

این کد داده های ورودی را با استفاده از فرمول مقیاس حداقل حداکثر به مقادیر بین 0 و 1 تبدیل می کند.

```python
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
```

این کد از تابع train_test_split در sklearn برای تقسیم داده های ورودی (x و y) به دو مجموعه آموزشی و آزمایشی استفاده می کند. داده ها به 85 درصد برای آموزش و 15 درصد برای آزمایش تقسیم می شوند. همچنین با استفاده از پارامتر random_state داده ها به صورت تصادفی تقسیم می شوند و با اختصاص یک عدد ثابت به این پارامتر می توان تست های متعدد با تقسیم های داده مشابه انجام داد.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)
```

این کد یک مدل رگرسیون لجستیک با استفاده از کتابخانه scikit-learn ایجاد می کند و سپس می تواند دستورات خاصی را برای مدل مشخص شده اعمال کند، مانند تنظیم حداکثر تعداد تکرار برای الگوریتم بهینه سازی (max_iter) و آموزش مدل بر روی داده های آموزشی (x_train و y_train ) با فراخوانی متد fit.


```python
from sklearn import linear_model
robot = linear_model.LogisticRegression()


robot.max_iter=1000000
robot.fit(x_train, y_train)
```

این کد پیش‌بینی‌های مدل را روی داده‌های آزمون (x_test) انجام می‌دهد و سپس با مقایسه y_test با پیش‌بینی‌ها بررسی می‌کند که آیا پیش‌بینی‌ها با مقادیر واقعی مطابقت دارند یا خیر. سپس تعداد مواردی که پیش‌بینی مدل اشتباه بوده را محاسبه و چاپ می‌کند.

```python
t=robot.predict(x_test)
v= y_test ==t 
v=v.reset_index()
v=v.diagnosis
n=0
for i in v:
    if i==False : n=n+1

print(n,'/',len(v))
```

## نتیجه

این پروژه توسط مجید تجن جاری و تیم Aiolearn نوشته شده است و ما به حمایت شما نیازمندیم!❤️