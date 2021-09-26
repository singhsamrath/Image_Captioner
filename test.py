
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import preprocess

def word2idx(i , tokenizer):
	for word , index in tokenizer.word_index.items():
		if index == i : 
			return word

	return None 

def generate_caption(model , tokenizer , photo , max_length):
	first_word = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([first_word])[0]
		# print(sequence)
		sequence = pad_sequences([sequence] , maxlen = max_length)
		# print(sequence)
		prediction = model.predict([photo , sequence] , verbose = 0)
		prediction = argmax(prediction)

		word = word2idx(prediction , tokenizer)

		if word is None :
			break 

		first_word += ' ' + word 

		if word == 'endseq':
			break 

		

	return first_word


def extract_feature(img_filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs , outputs = model.layers[-1].output)
	image = load_img(img_filename,target_size=(224,224))
	image = img_to_array(image)
	x,y,z=image.shape[0],image.shape[1],image.shape[2]
	image = image.reshape((1,x,y,z))
	image = preprocess_input(image)
	features = model.predict(image,verbose=0)
	return features

def modelgencptn(filename , photoname):
	model = load_model(filename)
	with open('tokenizer.pickle' , 'rb') as handle:
		tokenizer = load(handle)
	max_length = 33
	photo = extract_feature(photoname)

	Ans = generate_caption(model,tokenizer,photo,max_length)
	return Ans

def test_model():
	act , prd = list() , list()
	model = load_model('./model_9.h5')
	with open('tokenizer.pickle' , 'rb') as handle:
		tokenizer = load(handle)
	max_length = 33

	test_set = preprocess.make_dataset('D:/cs229/Flickr_tokens/Flickr_8k.testImages.txt')
	descriptions = preprocess.load_dataset_descriptions('D:/cs229/descriptions.txt' , test_set)
	test_images = preprocess.load_dataset_photo('D:/cs229/Final_Features.pkl' , test_set)
	for key , desc_list in descriptions.items():
		reference = [d.split() for d in desc_list]
		yhat = generate_caption(model,tokenizer,test_images[key] , max_length)
		act.append(reference)
		prd.append(yhat.split())

	print('BLEU-1: %f' % corpus_bleu(act, prd, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(act, prd, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(act, prd, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(act, prd, weights=(0.25, 0.25, 0.25, 0.25)))
		

# test_model()


