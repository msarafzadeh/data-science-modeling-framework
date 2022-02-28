from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import matplotlib.cm
from sklearn import  metrics
from sklearn.metrics import roc_curve

import seaborn as sns
import pandas as pd 

import xgboost as xgb

models = {}


def get_best_model(models,abs_score=False):
    if abs_score:
        best_model = max(models, key=lambda k: abs(models[k]['metric_score']))
    else:
        best_model = max(models, key=lambda k: models[k]['metric_score'])
    
    return best_model, models[best_model]

    

def plot_custom_confusion_matrix(cm,labels,confusion_matrix_values_format=None):
    
    cmap = matplotlib.cm.get_cmap('Blues')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    return disp.plot(cmap=cmap,values_format=confusion_matrix_values_format)




def generate_classification_report(y_true,y_pred,labels,confusion_matrix_normalize=None,confusion_matrix_values_format=None):
    
    selected_metric = None
    
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: %.2f%%" % (acc * 100.0))
    
    prec_mic = precision_score(y_true, y_pred,average='micro')
    print("Precision (micro): %.2f%%" % (prec_mic * 100.0))
    
    prec_mac = precision_score(y_true, y_pred,average='macro')
    print("Precision (macro): %.2f%%" % (prec_mac * 100.0))
    
    prec_wei = precision_score(y_true, y_pred,average='weighted')
    print("Precision (weighted): %.2f%%" % (prec_wei * 100.0))
    
    rec_mic = recall_score(y_true, y_pred,average='micro')
    print("Recall (micro): %.2f%%" % (rec_mic * 100.0))
    
    rec_mac = recall_score(y_true, y_pred,average='macro')
    print("Recall (macro): %.2f%%" % (rec_mac * 100.0))
    
    rec_wei = recall_score(y_true, y_pred,average='weighted')
    print("Recall (weighted): %.2f%%" % (rec_wei * 100.0))
    
    f1_mic = f1_score(y_true, y_pred,average='micro')
    print("F1 (micro): %.2f%%" % (f1_mic * 100.0))
    
    f1_mac = f1_score(y_true, y_pred,average='macro') 
    print("F1 (macro): %.2f%%" % (f1_mac * 100.0))
    
    f1_wei = f1_score(y_true, y_pred,average='weighted')
    print("F1 (weighted): %.2f%%" % (f1_wei * 100.0))
    
    f1_bin = f1_score(y_true, y_pred,average='binary')
    print("F1 (binary): %.2f%%" % (f1_bin * 100.0))
    
    f2_bin = fbeta_score(y_true, y_pred,average='binary',beta=2)
    print("F2 (binary): %.2f%%" % (f2_bin * 100.0))
    
    fd5_bin = fbeta_score(y_true, y_pred,average='binary',beta=0.5,)
    print("F1/2 (binary): %.2f%%" % (fd5_bin * 100.0))
    
    
    mcc_result = matthews_corrcoef(y_true, y_pred)
    print("MCC: %.4f%%" % (mcc_result))
    
    selected_metric = fd5_bin
    
    
    print()
    print()
    
    cm = confusion_matrix(y_true, y_pred, normalize=confusion_matrix_normalize)
    print(cm)
    print()
    print()
    print(classification_report(y_true, y_pred))

    # saving classification report to dataframe as well 
    output_classification_report = classification_report(y_true, y_pred, output_dict=True)
    output_classification_report = pd.DataFrame(output_classification_report).transpose()
    output_classification_report['F2'] = f2_bin
    output_classification_report['F1/2'] = fd5_bin
    output_classification_report['MCC'] = mcc_result
    
    
    # removed since shows cm of overfit training and not of CV 
    # disp = plot_confusion_matrix(model, X, y_pred,cmap=plt.cm.Blues,)
    # see: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/metrics/_plot/confusion_matrix.py#L185
    disp = plot_custom_confusion_matrix(cm,labels, confusion_matrix_values_format=confusion_matrix_values_format)
    plt.title("Confusion Matrix", fontsize =16)
    plt.show()
    
    
    #plot_rates(y_true,y_pred,labels)
    #plt.title("Rates", fontsize =16)
    #plt.show()
    
    return round(selected_metric,4), output_classification_report





def predict_model(model,X,y,validate=True,threshold=None,model_name=None, confusion_matrix_normalize=None, confusion_matrix_values_format=None):
    
    y_preds = model.predict(X)
    
    if threshold:
        y_preds_orig = y_preds
        y_preds= (y_preds>threshold)
    
    
    try:
        labels=model.classes_
    except AttributeError:
        labels = list(set(y))
    
    
    score, output_classification_report = generate_classification_report(y,y_preds,labels,confusion_matrix_normalize, confusion_matrix_values_format)

        
    model_results = {}
    model_results['model']=model
    model_results['metric_score']=score
    model_results['classification_report']=output_classification_report
    model_results['y_preds']=y_preds    
    if threshold:
        model_results['y_preds_orig']=y_preds_orig
    
    
    if validate and model_name:
        models[model_name] = model_results
    
    
    return model_results






def plot_loss(history,train_loss_path='loss',val_loss_path='val_loss'):
    # Plot training & validation loss values
    #plt.plot(history.history[train_loss])
    #plt.plot(history.history[val_loss])
        
    plt.plot(train_loss_path)
    plt.plot(val_loss_path)    
    
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    


def xgboost_feat_importance(model,X_data, reduced_plotting=False,max_num_feat = 10):
    #xgboost_importance_types=['weight','gain','cover']

    #for xgboost_importance_type in xgboost_importance_types:
        #print(xgboost_importance_type)
   
    print('Feature Importance (weight)')
    if reduced_plotting:
        xgb.plot_importance(model, max_num_features=max_num_feat,show_values =False)
    else:
        xgb.plot_importance(model) 

    plt.show()
    print()
    print()

    
    print('Feature Importance (gain)')
    results=pd.DataFrame()
    results['features']=X_data.columns
    results['importances'] = model.feature_importances_
    results.sort_values(by='importances',ascending=False,inplace=True)
    
    if reduced_plotting:
        return results.head(10)
    
    return results
###########


def _plot_roc(fpr, tpr,):
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='estimator')
    display.plot()  
    plt.show()
    return 

def _cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def calculate_roc(y_valid,y_preds):
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_preds,)
    
    _plot_roc(fpr, tpr,)
    best_threshold= _cutoff_youdens_j(fpr,tpr,thresholds)
    print('Best threshold: '+str(best_threshold))
    return best_threshold
###########

def _plot_pr_curve(actual,predicted,):
    precision, recall, thresholds = precision_recall_curve(actual, predicted)
    average_precision = average_precision_score(actual, predicted)

    pr_auc = metrics.auc(recall, precision)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall,average_precision=average_precision, estimator_name = '(PR AUC: ' + str(round(pr_auc,2))+')' )
    
    disp.plot()
    return precision,recall,thresholds
    
    
def _optimal_pr_threshold(precision,recall,thresholds,beta=1):
    
    metric = 'f'+str(beta)
    pr_results = pd.DataFrame({'threshold':thresholds, 'precision':precision[1:], 'recall':recall[1:]})
    pr_results['_f1']=  2 * (pr_results.precision * pr_results.recall) / (pr_results.precision + pr_results.recall)
    pr_results[metric]=  (1+beta**2) * (pr_results.precision * pr_results.recall) / ((beta**2) * pr_results.precision + pr_results.recall)
    
    pr_results = pr_results.sort_values(by=metric,ascending=False)
    best_pr = pr_results.head(1)
    optimal_threshold = best_pr['threshold'].values[0]
    
    print(pr_results.head(3))
    
    return optimal_threshold
    
def _plot_pr_threshold_chart(precision, recall, thresholds):
    
    #plt.clf()
    plt.figure()
    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0,1])
    
    
def calculate_pr(actual, predicted, beta=1):
    
    metric = 'f'+str(beta)
        
    precision,recall,thresholds = _plot_pr_curve(actual,predicted,)
    optimal_threshold = _optimal_pr_threshold(precision, recall, thresholds, beta)
    print('\nBest %s is at threshold: %s' %(metric, optimal_threshold))
    
    _plot_pr_threshold_chart(precision, recall, thresholds)

    return optimal_threshold
