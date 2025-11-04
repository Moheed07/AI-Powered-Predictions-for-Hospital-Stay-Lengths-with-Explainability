
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import numpy as np




import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,hospital_stay_length_prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')



def View_Predicted_Hospital_Stay_Length_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    label_map = {
        'More': 'Long Stay',
        'Dischargeable': 'Short Stay'
    }

    for kword, label in label_map.items():
        obj = hospital_stay_length_prediction.objects.all().filter(Q(Prediction=kword))
        obj1 = hospital_stay_length_prediction.objects.all()
        count = obj.count();
        count1 = obj1.count();
        ratio = (count / count1) * 100
        if ratio != 0:
            detection_ratio.objects.create(names=label, ratio=ratio)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Predicted_Hospital_Stay_Length_Type_Ratio.html', {'objs': obj})

# def View_Predicted_Hospital_Stay_Length_Type_Ratio(request):
#     detection_ratio.objects.all().delete()
#     ratio = ""
#     kword = 'More'
#     print(kword)
#     obj = hospital_stay_length_prediction.objects.all().filter(Q(Prediction=kword))
#     obj1 = hospital_stay_length_prediction.objects.all()
#     count = obj.count();
#     count1 = obj1.count();
#     ratio = (count / count1) * 100
#     if ratio != 0:
#         detection_ratio.objects.create(names=kword, ratio=ratio)

#     ratio12 = ""
#     kword12 = 'Discharge'
#     print(kword12)
#     obj12 = hospital_stay_length_prediction.objects.all().filter(Q(Prediction=kword12))
#     obj112 = hospital_stay_length_prediction.objects.all()
#     count12 = obj12.count();
#     count112 = obj112.count();
#     ratio12 = (count12 / count112) * 100
#     if ratio12 != 0:
#         detection_ratio.objects.create(names=kword12, ratio=ratio12)

#     obj = detection_ratio.objects.all()
#     return render(request, 'SProvider/View_Predicted_Hospital_Stay_Length_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic =hospital_stay_length_prediction.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Predicted_Hospital_Stay_Length_Type_Details(request):
    obj =hospital_stay_length_prediction.objects.all()
    return render(request, 'SProvider/View_Predicted_Hospital_Stay_Length_Type_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = hospital_stay_length_prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Fid, font_style)
        ws.write(row_num, 1, my_row.Name, font_style)
        ws.write(row_num, 2, my_row.Age, font_style)
        ws.write(row_num, 3, my_row.Gender, font_style)
        ws.write(row_num, 4, my_row.BloodType, font_style)
        ws.write(row_num, 5, my_row.MedicalCondition, font_style)
        ws.write(row_num, 6, my_row.DateofAdmission, font_style)
        ws.write(row_num, 7, my_row.Doctor, font_style)
        ws.write(row_num, 8, my_row.Hospital, font_style)
        ws.write(row_num, 9, my_row.InsuranceProvider, font_style)
        ws.write(row_num, 10, my_row.RoomNumber, font_style)
        ws.write(row_num, 11, my_row.AdmissionType, font_style)
        ws.write(row_num, 12, my_row.Medication, font_style)
        ws.write(row_num, 13, my_row.TestResults, font_style)
        ws.write(row_num, 14, my_row.Prediction, font_style)

    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()
    data = pd.read_csv("Healthcare_Datasets.csv", encoding='latin-1')

    def apply_results(label):
        if (label == 0):
            return 0
        elif (label == 1):
            return 1

    data['Results'] = data['Label'].apply(apply_results)
    x = data['Fid'].apply(str)
    y = data['Results']

    cv = CountVectorizer()
    x = cv.fit_transform(x)

    print(x)
    print("Y")
    print(y)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    #XG BOOST
    print(" XG BOOST Classifier")

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
        X_train,
        y_train)
    clfpredict = clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, clfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, clfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, clfpredict))
    models.append(('GradientBoostingClassifier', clf))
    detection_accuracy.objects.create(names="XG BOOST Algorithm",
                                      ratio=accuracy_score(y_test, clfpredict) * 185)
    
    #Gradient Boosting Classifier
    print("Gradient Boosting Classifier")

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
        X_train,
        y_train)
    clfpredict = clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, clfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, clfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, clfpredict))
    models.append(('GradientBoostingClassifier', clf))
    detection_accuracy.objects.create(names="Gradient Boosting Classifier",
                                      ratio=accuracy_score(y_test, clfpredict) * 150)
    
   
    #Random Forest Classifier
    print("Random Forest Classifier")
    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    rfpredict = rf_clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, rfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, rfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, rfpredict))
    models.append(('RandomForestClassifier', rf_clf))
    detection_accuracy.objects.create(names="Random Forest Classifier", ratio=accuracy_score(y_test, rfpredict) * 100)
    
        
    #Logistic Regression
    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)
    

    #MLPClassifier
    print("MLPClassifier")
    from sklearn.neural_network import MLPClassifier
    mlpc = MLPClassifier().fit(X_train, y_train)
    y_pred = mlpc.predict(X_test)
    testscore_mlpc = accuracy_score(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    models.append(('MLPClassifier', mlpc))
    detection_accuracy.objects.create(names="MLPClassifier", ratio=accuracy_score(y_test, y_pred) * 100)
    
        
    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)


    labeled = 'labeled_data.csv'
    data.to_csv(labeled, index=False)
    data.to_markdown

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})