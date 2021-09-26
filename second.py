
#Main File 
import Myencoder
import preprocess 
import pickle
from numpy import array
import tensorflow
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

home_dir = 'D:/cs229'
image_dir = 'D:/cs229/Flicker8k_Dataset'
# Final_Features = Myencoder.encode_extract_features(image_dir) 

print("** Encoding Finished **")

# pickle.dump(Final_Features , open(home_dir + '/Final_Features.pkl' , 'wb'))

#%%
description_dir = 'D:/cs229/Flickr_tokens/Flickr8k.token.txt'

description_doc = preprocess.load_document(description_dir)
descriptions = preprocess.descriptions2Dict(description_doc)
preprocess.clean_descriptions(descriptions)

Vocab = preprocess.build_vocab(descriptions)
preprocess.save_final_descriptions(descriptions , home_dir + '/descriptions.txt')

print('length of descriptions  = %d \nVocab size = %d'  %(len(descriptions) , len(Vocab)))

train_file = 'D:/cs229//Flickr_tokens/Flickr_8k.trainImages.txt'
train_identifiers = preprocess.make_dataset(train_file)
train_descriptions = preprocess.load_dataset_descriptions( home_dir + '/descriptions.txt', train_identifiers)
train_img_features = preprocess.load_dataset_photo(home_dir + '/Final_Features.pkl' , train_identifiers)

print("Train dataset fully complete ")
print(len(train_descriptions))
print(len(train_img_features))

#%%

tokenizer = preprocess.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1 

lines = preprocess.to_line(train_descriptions)
maxlen = max(len(d.split()) for d in lines)
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	for desc in desc_list:
		seq = tokenizer.texts_to_sequences([desc])[0]
		for i in range(1, len(seq)):
			in_seq, out_seq = seq[:i], seq[i]
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)


def data_generator(descriptions , photos , tokenizer , max_length):
	while 1 :
		for key , desc_list in descriptions.items() :
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]



def define_model(vocab_size, max_length):
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())
	return model


model = define_model(vocab_size, maxlen)
epochs = 5
steps = len(train_descriptions)
for i in range(epochs):
	generator = data_generator(train_descriptions, train_img_features, tokenizer, maxlen)
	model.fit_generator(generator, epochs= 5, steps_per_epoch=steps, verbose=1)
	model.save(home_dir + '/model_' + str(i) + '.h5')

