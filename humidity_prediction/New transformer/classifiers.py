import numpy as np
from sklearn.tree import DecisionTreeClassifier
def tree(x_train,y_train,verbose=True):
    if verbose:
        print('Decision Tree')
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model

from sklearn.ensemble import RandomForestClassifier
def forest(x_train,y_train,verbose=True):
    if verbose:
        print('Random Forest')
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    return model

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
def adaboost_clf(x_train,y_train,verbose=True):
    if verbose:
        print('Adaboost Regression')
    rng = np.random.RandomState(1)
    model = AdaBoostClassifier(DecisionTreeClassifier(),random_state=rng,n_estimators=200)
    model.fit(x_train, y_train)
    return model

from sklearn.ensemble import GradientBoostingClassifier
def gradient_boosting(x_train, y_train,verbose=True):
    if verbose:
        print('Gradient Boosting Regression')
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01,"random_state":0}
    model = GradientBoostingClassifier(**params)
    model.fit(x_train, y_train)
    return model

#%%

from sklearn.metrics import classification_report, f1_score
from desc_plot import plot_confusion_matrix

def show_clf_result(model, x_test, y_test,plot=True):    
    y_pred = model.predict(x_test)
    if plot:
        print('Classification_report: \n', 
          classification_report(
              y_true=y_test,
              y_pred=y_pred))
        plot_confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred,average="weighted")
    return f1
#%%

def pipeline_clf(x_train,y_train,x_test,y_test,
                 models=['tree','forest','ada','grad'],
                 plot=True):
    y_train = y_train.values.reshape(-1,)
    model_result_dict = {}
    if 'tree' in models:
        # Decision tree
        model = tree(x_train,y_train,verbose=plot)
        result = show_clf_result(model,x_test,y_test,plot)
        model_result_dict['tree'] = (model,result)
        
    if 'forest' in models:
        # Random forest
        model = forest(x_train,y_train,verbose=plot)
        result = show_clf_result(model,x_test,y_test,plot)
        model_result_dict['forest'] = (model,result)
        
    if 'ada' in models:
        # Adaboost classifier
        model = adaboost_clf(x_train,y_train,verbose=plot)
        result = show_clf_result(model,x_test,y_test,plot)
        model_result_dict['ada'] = (model,result)
        
    if 'grad' in models:
        # gradient boosting classifier
        model = gradient_boosting(x_train,y_train,verbose=plot)
        result = show_clf_result(model,x_test,y_test,plot)
        model_result_dict['grad'] = (model,result)
        
    return model_result_dict