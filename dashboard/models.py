from django.db import models

class Data(models.Model):
    name = models.CharField(max_length=200, default='data')
    file = models.FileField()

    def __str__(self):
        return str(self.name)