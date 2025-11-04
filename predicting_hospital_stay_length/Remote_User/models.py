from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)


class hospital_stay_length_prediction(models.Model):

    Fid= models.CharField(max_length=3000)
    Name= models.CharField(max_length=3000)
    Age= models.CharField(max_length=3000)
    Gender= models.CharField(max_length=3000)
    BloodType= models.CharField(max_length=3000)
    MedicalCondition= models.CharField(max_length=3000)
    DateofAdmission= models.CharField(max_length=3000)
    Doctor= models.CharField(max_length=3000)
    Hospital= models.CharField(max_length=3000)
    InsuranceProvider= models.CharField(max_length=3000)
    RoomNumber= models.CharField(max_length=3000)
    AdmissionType= models.CharField(max_length=3000)
    Medication= models.CharField(max_length=3000)
    TestResults= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



