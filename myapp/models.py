from django.db import models

class DogBreed(models.Model):
    breed = models.CharField(max_length=255)
    image = models.ImageField(upload_to='dog_images/')
