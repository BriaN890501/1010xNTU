from django.db import models

# Create your models here.
class Method(models.Model):
    Name = models.CharField(max_length=100, default= "default")
    MA_S = models.BooleanField(default=True) #moving average seasonal
    SE_S = models.BooleanField(default=True) #single exponential seasonal
    DE_S = models.BooleanField(default=True) #double exponential seasonal
    MA_NS = models.BooleanField(default=True) #moving average non-seasonal
    SE_NS = models.BooleanField(default=True) #single exponential non-seasonal
    DE_NS = models.BooleanField(default=True) #double exponential non-seasonal

    def __str__(self):
        return self.Name