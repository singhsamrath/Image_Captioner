
import preprocess 
import pickle

# print(123)
train = preprocess.make_dataset('./Flickr_tokens/Flickr_8k.trainImages.txt')
descriptions = preprocess.load_dataset_descriptions('./descriptions.txt',train)
tokenizer = preprocess.create_tokenizer(descriptions)
# print("jk")
with open('./tokenizer.pickle' , 'wb') as handle :
	print("Saving started")
	pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

