import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle


def get_clean_data():
    data = pd.read_csv('data/data.csv')
    
    # drop col Unnamed: 32, id
    data.drop(['Unnamed: 32','id'],axis=1, inplace=True)
    
    # replace M with 1 and B with 0 in col diagnosis
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    print(data.head())
    return data

def create_model(data):
    X = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']
    
    # sclae
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  
    
    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    # train the model
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    # test the model
    y_pred = model.predict(X_test)
    print('Accuracy: ',accuracy_score(y_test,y_pred))
    print('Classification report: \n', classification_report(y_test,y_pred))
    return model, scaler
    
  
    
def main():
    data = get_clean_data()
    print(data.info())
    
    model,scaler = create_model(data)
    
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

    
if __name__ == '__main__':
    main()
