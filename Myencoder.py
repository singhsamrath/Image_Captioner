
#Encode the Images using VGG16 and store it 
from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


def encode_extract_features(directory):
    model = VGG16() 
    model.layers.pop() 
    model = Model(inputs = model.inputs , outputs=model.layers[-1].output)
    
    # print(model.summary())
    final_ftr = dict() 
    
    for name in listdir(directory):
        filename = directory + '/' + name 
        image = load_img(filename , target_size=(224,224))
        image = img_to_array(image)
        t = image.shape
        image = image.reshape((1,t[0],t[1],t[2]))
        image = preprocess_input(image)
        
        feature = model.predict(image,verbose=0)
        image_id = name.split('.')[0]
        final_ftr[image_id] = feature 
        
    return final_ftr


        
    
    

