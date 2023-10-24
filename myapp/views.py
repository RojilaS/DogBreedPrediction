# dog_breed_prediction/views.py
from django.conf import settings 
from django.shortcuts import render
\
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from . models import *

def predict_dog_breed(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_file.name, uploaded_file)

        with open(os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg'), 'wb+') as temp_image:
            for chunk in uploaded_file.chunks():
                temp_image.write(chunk)

        # Load and preprocess the uploaded image
        img_width, img_height = 224, 224
        img = image.load_img(os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg'), target_size=(img_width, img_height))

        # Load the pre-trained model
        model = load_model('dog_breed.h5')
        dogs=['australian_terrier',
 'beagle',
 'boxer',
 'chow',
 'german_shepherd',
 'labrador_retriever',
 'pomeranian']

        # Load and preprocess the uploaded image
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = dogs[predicted_class_index]
        new = DogBreed.objects.create(breed=predicted_class,image=uploaded_file)
        new.save()

        return render(request, 'upload.html', {'prediction': predicted_class,'result':new})

    return render(request, 'upload.html')



