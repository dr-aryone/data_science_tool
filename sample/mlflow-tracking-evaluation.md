
![mlflow - Tracking](https://www.recordedfuture.com/assets/machine-learning-cybersecurity-applications.png)

Source: https://www.recordedfuture.com/machine-learning-cybersecurity-applications/

In data science work, Jupyter notebook is a well known tools. Other than, we may use databricks's notebook or Colab( by Google). How about productization? 

How can deploy our model to production? We can use TensorFlow Serving if you are using tensorflow library to build model. Data Science Workbench can be used if you use Azure platform. What if we do use the above solution? Some company built an internal system such as michaelangelo (by Uber), FBLearner Flow (by Facebook). However, those platform is designed for their internal system. 

Do we have open source? Before that, we have modeldb. We have new option, mlflow,  which just released alpha version in last week (5 Jun 2018). Let have a tour to this product.

### Data Preparation


```python
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

data = load_iris()

feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
y_name = 'target'

data_df = pd.DataFrame(
    data=np.c_[data['data'], data['target']],
    columns=feature_names + [y_name])
# data_df.head()
```


```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(data_df)
```


```python
x_train = train[feature_names]
y_train = train[y_name]

x_test = test[feature_names]
y_test = test[y_name]
```


```python
# Copy from mlflow example

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
```

![mlflow - Tracking](https://sdtimes.com/wp-content/uploads/2018/06/mlflow-490x245.png)

Source: https://sdtimes.com/ai/databricks-releases-mlflow-runtime-for-ml-and-databricks-delta-at-spark-ai-summit/


```python
import mlflow
import mlflow.sklearn
```

##### Since cannot retrieve user from window environment. I override the deafult name. (Not recommend)


```python
mlflow.tracking._DEFAULT_USER_ID = 'https://github.com/makcedward/'
```

##### Set the tracking destination


```python
mlflow.set_tracking_uri("http://localhost:5000")
```


```python
# First of all, you have to execute the following command in command line to start the UI service
# mlflow ui
```

### Tracking


```python
from sklearn.linear_model import LogisticRegression

tuned_parameters = [
    {'penalty': 'l2', 'solver': 'newton-cg', 'max_iter': 10},
    {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 10},
    {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 30},
    {'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 50},
]


cnt = 0 
for param in tuned_parameters:
    with mlflow.start_run(source_name='https://medium.com/@makcedward/', source_version='1.0.' + str(cnt)):
        lr = LogisticRegression(max_iter=param['max_iter'], penalty=param['penalty'], solver=param['solver'])
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("max_iter", param['max_iter'])
        mlflow.log_param("penalty", param['penalty'])
        mlflow.log_param("solver", param['solver'])

        mlflow.sklearn.log_model(lr, "model")
        
    cnt += 1
```

    req_body: {
      "experimentId": "0",
      "userId": "https://github.com/makcedward/",
      "sourceType": "LOCAL",
      "sourceName": "https://medium.com/@makcedward/",
      "startTime": "1528651944413",
      "sourceVersion": "1.0.0"
    }
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\optimize.py:203: ConvergenceWarning: newton-cg failed to converge. Increase the number of iterations.
      "number of iterations.", ConvergenceWarning)
    

    req_body: {
      "experimentId": "0",
      "userId": "https://github.com/makcedward/",
      "sourceType": "LOCAL",
      "sourceName": "https://medium.com/@makcedward/",
      "startTime": "1528651953631",
      "sourceVersion": "1.0.1"
    }
    req_body: {
      "experimentId": "0",
      "userId": "https://github.com/makcedward/",
      "sourceType": "LOCAL",
      "sourceName": "https://medium.com/@makcedward/",
      "startTime": "1528651962805",
      "sourceVersion": "1.0.2"
    }
    req_body: {
      "experimentId": "0",
      "userId": "https://github.com/makcedward/",
      "sourceType": "LOCAL",
      "sourceName": "https://medium.com/@makcedward/",
      "startTime": "1528651971968",
      "sourceVersion": "1.0.3"
    }
    
