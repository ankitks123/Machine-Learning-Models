import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    raw_data = []
    #Loading the json file
    with open('/home/ankitsingh/Desktop/datasets/customersdata.json') as f:
        for line in f:
            raw_data.append(json.loads(line))

    #Creating Dataframe
    records = []
    for i in range(len(raw_data)):
        orders = {}
        payments = {}
        transactions = {}
        fraudulent = raw_data[i]['fraudulent']
        customer = [raw_data[i]['customer']['customerEmail'], raw_data[i]['customer']['customerPhone'], raw_data[i]['customer']['customerDevice'], raw_data[i]['customer']['customerIPAddress'], raw_data[i]['customer']['customerBillingAddress']]
        for order in raw_data[i]['orders']:
            orders[order['orderId']] = [order['orderId'], order['orderAmount'], order['orderState'], order['orderShippingAddress']]
        for payment in raw_data[i]['paymentMethods']:
            payments[payment['paymentMethodId']] = [payment['paymentMethodId'], payment['paymentMethodRegistrationFailure'], payment['paymentMethodType'], payment['paymentMethodProvider'], payment['paymentMethodIssuer']]
        for transaction in raw_data[i]['transactions']:
            transactions[transaction['transactionId']] = [transaction['transactionId'], transaction['orderId'], transaction['paymentMethodId'], transaction['transactionAmount'], transaction['transactionFailed']]
        for k,v in transactions.items():
            records.append([fraudulent] + customer + orders[v[1]] + payments[v[2]] + [k] + v[3:])
    columns = ['fraudulent', 'customerEmail', 'customerPhone', 'customerDevice', 'customerIPAddress', 'customerBillingAddress', 'orderId', 'orderAmount', 'orderState', 'orderShippingAddress', 'paymentMethodId', 'paymentMethodRegistrationFailure', 'paymentMethodType', 'paymentMethodProvider', 'paymentMethodIssuer', 'transactionId', 'transactionAmount', 'transactionFailed']
    customer_df = pd.DataFrame(records, columns=columns)
    customer_df.set_index('customerEmail', inplace=True) #customerEmail is the unique identifier for each customer
    customer_df.dropna(inplace=True)

    #To drop features not useful at all
    customer_df.drop(['customerPhone', 'customerDevice', 'customerIPAddress', 'customerBillingAddress', 'orderId', 'orderShippingAddress', 'paymentMethodId', 'transactionId', 'transactionAmount'], 1, inplace=True)
    customer_df['orderState'] = customer_df['orderState'].astype('category')
    customer_df['paymentMethodType'] = customer_df['paymentMethodType'].astype('category')
    customer_df['paymentMethodProvider'] = customer_df['paymentMethodProvider'].astype('category')
    customer_df['paymentMethodIssuer'] = customer_df['paymentMethodIssuer'].astype('category')

    #To make sure Logistic Regression and SVM are able to handle categorical data.
    le = LabelEncoder()
    customer_df['orderState'] = le.fit_transform(customer_df['orderState'])
    customer_df['paymentMethodType'] = le.fit_transform(customer_df['paymentMethodType'])
    customer_df['paymentMethodProvider'] = le.fit_transform(customer_df['paymentMethodProvider'])
    customer_df['paymentMethodIssuer'] = le.fit_transform(customer_df['paymentMethodIssuer'])

    #Separate out feature set and target
    X = customer_df.drop('fraudulent', 1).values
    y = customer_df['fraudulent'].values

    #Creating train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)

    #Considering baseline model to be Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    print('Logistic Regression model accuracy: ', logistic_model.score(X_test, y_test))
    pred = logistic_model.predict(X_test)
    print('Logistic Regression model F1 score: ', f1_score(y_test, pred))

    #Support Vector Classifier
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)

    print('SVC model accuracy: ', svm_model.score(X_test, y_test))
    pred = svm_model.predict(X_test)
    print('SVC model F1 score: ', f1_score(y_test, pred))

    #Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators = 1000, random_state = None)
    rf_model.fit(X_train, y_train)

    print('Random Forest Classifier model accuracy: ', rf_model.score(X_test, y_test))
    pred = rf_model.predict(X_test)
    print('Random Forest Classifier model F1 score:', f1_score(y_test, pred))

    #Finding the most influencial features according to the Random Forest Model
    feature_importance = {i:j for i,j in zip(customer_df.columns[1:], rf_model.feature_importances_)}
    print('Feature importance according to Random Forest Classifier: ', feature_importance)

    #Using the correlation among the features to find the most important feature
    plt.figure(figsize=(12,10))
    cor = customer_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
