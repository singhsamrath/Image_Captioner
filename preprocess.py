
#Preprocess the vocab etcetera
import string
import pickle
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#Open the Description file 
def load_document(filename):
	file = open(filename,'r')
	text = file.read() 
	file.close()
	return text 

def descriptions2Dict(text):
	Ans = dict() 
	for a in text.split('\n'):
		letters = a.split()
		if len(letters) < 2 :
			continue 

		imgid , desc = letters[0] , letters[1:]
		imgid = imgid.split('.')[0]
		desc = ' '.join(desc)

		if imgid not in Ans :
			Ans[imgid] = list() 
		Ans[imgid].append(desc)

	return Ans

def clean_descriptions(descriptions):
	extra = str.maketrans('','', string.punctuation)
	for key , desc in descriptions.items():
		for i in range (len(desc)):
			temp = desc[i].split() 
			clean = list()

			for word in temp :
				word = word.lower().translate(extra)
				if len(word) > 1 and word.isalpha():
					clean.append(word) 

			clean = ' '.join(clean) 
			desc[i] = clean


def build_vocab(descriptions):
	Vocab = set() 

	for key , desc_list in descriptions.items():
		for i in range (len(desc_list)):
			for word in desc_list[i].split() :
				Vocab.add(word)

	return Vocab

def save_final_descriptions(descriptions , filename):
	line = list() 
	for key , desc_list in descriptions.items():
		for i in range (len(desc_list)):
			line.append(key + ' ' + desc_list[i])

	final = '\n'.join(line)
	file = open(filename , 'w')
	file.write(final)
	file.close()

def make_dataset(filename):
	dataset = set() 
	file = load_document(filename)
	for line in file.split('\n'):
		temp = line.split('.')
		if len(line) < 1 :
			continue 
		image_id = temp[0]  
		dataset.add(image_id)

	return dataset

def load_dataset_descriptions(filename , dataset):
	final_ans = dict() 
	descriptions = load_document(filename)
	for line in descriptions.split('\n') :
		temp = line.split()
		key , desc = temp[0] , temp[1:]
		if key in dataset :

			if key not in final_ans :
				final_ans[key] = list()

			caption = 'startseq' + ' '.join(desc) + ' endseq'
			
			final_ans[key].append(caption)

	return final_ans

def load_dataset_photo(filename , dataset):
	images = pickle.load(open(filename,'rb'))
	final_ans = dict()

	for key in dataset :
		final_ans[key] = images[key]

	return final_ans

def to_line(descriptions):
	all_desc = list() 
	for key in descriptions.keys():
		for desc in descriptions[key] :
			all_desc.append(desc) 


	return all_desc

def create_tokenizer(descriptions):
	lines = to_line(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer




















