import numpy as np
from sklearn.tree import DecisionTreeClassifier
def tree(x_train,y_train):
    print('Decision Tree')
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model

from sklearn.ensemble import RandomForestClassifier
def forest(x_train,y_train):
    print('Random Forest')
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    return model

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
def adaboost_clf(x_train,y_train):
    print('Adaboost Regression')
    rng = np.random.RandomState(1)
    model = AdaBoostClassifier(DecisionTreeClassifier(),random_state=rng,n_estimators=200)
    model.fit(x_train, y_train)
    return model

from sklearn.ensemble import GradientBoostingClassifier
def gradient_boosting(x_train, y_train):
    print('Gradient Boosting Regression')
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01,"random_state":0}
    model = GradientBoostingClassifier(**params)
    model.fit(x_train, y_train)
    return model

#%%

from sklearn.metrics import classification_report
from desc_plot import plot_confusion_matrix

def show_clf_result(model, x_test, y_test):    
    y_pred = model.predict(x_test)
    print('Classification_report: \n', 
      classification_report(
          y_true=y_test,
          y_pred=y_pred))
    plot_confusion_matrix(y_test, y_pred)

#%%

def pipeline_clf(x_train,y_train,x_test,y_test):
    y_train = y_train.values.reshape(-1,)
    # Linear regression
    model = tree(x_train,y_train)
    show_clf_result(model,x_test,y_test)
    # linear svr
    model = forest(x_train,y_train)
    show_clf_result(model,x_test,y_test)
    # adaboost regressor
    model = adaboost_clf(x_train,y_train)
    show_clf_result(model,x_test,y_test)
    # gradient boosting regression  
    model = gradient_boosting(x_train,y_train)
    show_clf_result(model,x_test,y_test)