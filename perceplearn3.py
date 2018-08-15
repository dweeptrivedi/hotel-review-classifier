# perceplearn3.py will learn perceptron classifier from the training data to identify hotel reviews as either true or fake, and either positive or negative

#prior prob: p(c)


#likelihood prob: p(fi/c)


import math
import sys
from collections import defaultdict
import copy
import re
import numpy as np
import operator 
import random

priorProb1 = defaultdict(float)
priorProb2 = defaultdict(float)
likelihoodProb1 = defaultdict(float)
likelihoodProb2 = defaultdict(float)

#store count of documents/class
priorDic1 = defaultdict(int)
priorDic2 = defaultdict(int)

#store count of a feature in a perticular class
likelihoodDic1 = defaultdict(int)
likelihoodDic2 = defaultdict(int)

#set of vocabulary words in training data
vocabList1 = set()
vocabList2 = set()

#store sum of count of all words in a class
classLength1 = defaultdict(int)
classLength2 = defaultdict(int)

#counts word frequency
wordFreq = defaultdict(int)
df = defaultdict(int)
dfb = defaultdict(int)

#List of possible classes:
classList = [("True","Pos"),("True","Neg"),("Fake","Pos"),("Fake","Neg")]
classList1 = ["True","Fake"]
classList2 = ["Pos","Neg"]

#with stopwords: 0.9139, without stopwords: 0.9045
#FIXME: If I do something for unknown words in test dataset then I need to use the same list in nbclassify3.py also
#stopWords = ['few', 'how', 'such', 'doing', 'through', 'should', 'your', 'yourselves', 'again', "let's", 'whom', 'because', "i'm", 'both', "shouldn't", 'are', 'is', "he'll", "we've", 'been', 'at', 'here', "they'll", "she'll", 'into', 'below', 'under', 'those', 'be', 'them', "you'll", 'about', 'am', 'her', 'him', "mustn't", "what's", 'ourselves', 'against', 'hers', "you've", "wasn't", 'no', "we'll", 'yours', '', 'themselves', 'than', 'being', 'above', 'in', 'these', 'then', "couldn't", 'between', 'not', 'on', "he'd", 'up', 'itself', "i've", 'to', 'very', "aren't", "you'd", 'only', 'any', 'for', 'from', 'ought', "can't", 'it', 'do', "they'd", 'would', 'further', "we're", "when's", 'some', 'have', 'why', 'does', 'too', "we'd", 'if', "there's", 'the', 'so', 'their', "don't", "weren't", "you're", 'did', 'after', 'myself', 'they', 'i', 'which', "they're", 'all', 'down', "i'd", 'with', 'own', "they've", 'our', 'he', 'or', 'you', 'an', "hadn't", 'cannot', 'its', 'who', "won't", "didn't", 'when', 'a', 'during', "hasn't", 'ours', 'had', 'where', 'but', 'himself', "where's", 'until', 'was', 'nor', 'his', 'and', "she's", 'this', "why's", "how's", "wouldn't", 'each', 'off', "isn't", 'herself', "it's", 'same', 'out', 'has', "that's", "she'd", 'what', 'theirs', 'by', "here's", 'of', 'as', 'over', 'having', 'yourself', 'we', 'most', 'once', 'could', 'she', 'that', 'while', "doesn't", "haven't", 'my', 'there', 'were', "i'll", 'me', "who's", "he's", 'before', "shan't", 'other', 'more']
stopWords = []


trainFile = "coding-2-data-corpus/train-labeled.txt"

documentList = []

positionDic = defaultdict(int)
bPositionDic = defaultdict(int)


def getFeatureVec(doc, X):
	for i in range(3,len(doc)):
		X[positionDic[doc[i]]] += 1



def trainPercep(itrV=30,itrA=30):
	global trainFile

	if len(sys.argv)>1:
		trainFile = sys.argv[1]

	inputFile = open(trainFile,"r",encoding="utf-8")
	lines = inputFile.readlines()
	#print("No. of documents available for training:",len(lines))
	inputFile.close()

	for line in lines:
		document = line.strip()
		[class1,class2] = document.split(' ')[1:3]
		tempDoc = document.split(' ')[0:3]

		document = document.lower()
		document = re.sub('[?.!/;:,"]', '', document)
		document = re.sub("[{}()']", '', document)
		document = document.split(' ')
		wordList = document[3:]
		
		#increase the count of documents for found class
		priorDic1[class1] += 1
		priorDic2[class2] += 1

		prev_word = "_START_"
		bigrams = set()

		for word in wordList:
			if word in stopWords:
				continue
			tempDoc.append(word)
			likelihoodDic1[(word,class1)] += 1
			likelihoodDic2[(word,class2)] += 1
			classLength1[class1] += 1
			classLength2[class2] += 1
			#FIXME: doesn't make sense to have vocabList1 and vocabList2, revist
			vocabList1.add(word)
			vocabList2.add(prev_word+word)
			bigrams.add(prev_word+word)
			wordFreq[word] += 1
			prev_word = word

		for word in set(wordList):
			df[word] += 1

		for b in bigrams:
			dfb[b] += 1


		#print(tempDoc)
		documentList.append(tempDoc)

	#print(dfb)
	
	vocabList = list(vocabList1)
	vocabList.sort()
	#sorted_x = sorted(dfb.items(), key=operator.itemgetter(1))
	#print(sorted_x)
	len_vocabList = len(vocabList)

	#construct bigram vocab
	bigramList = list(vocabList2)
	bigramList.sort()
	len_bigramList = len(bigramList)


	for i in range(len_vocabList):
		positionDic[vocabList[i]] = i

	#create position list for bigrams
	for i in range(len_bigramList):
		bPositionDic[bigramList[i]] = i

	#print("length of positionDic:",len(positionDic))
	assert(len(positionDic)==len(vocabList))

	df1 = np.zeros(len_vocabList,dtype=np.float64)
	dfb1 = np.zeros(len_bigramList,dtype=np.float64)
	#print("Total Docs:",len(lines),len(df1),len(dfb1))
	for i in range(len_vocabList):
		df1[positionDic[vocabList[i]]] = 1+math.log((len(lines)+1)/(1+df[vocabList[i]]))

	for i in range(len_bigramList):
		dfb1[bPositionDic[bigramList[i]]] = 1+math.log((len(lines)+1)/(1+dfb[bigramList[i]]))

	#print(bigramList)

	np.save("df1.npy", df1)
	np.save("dfb1.npy", dfb1)

	df_final = np.concatenate([df1,dfb1])

	W1 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	W2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	b1 = 0
	b2 = 0
	maxIter = itrV

	Wa1 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	Wa2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	ba1 = 0
	ba2 = 0
	c1 = 1
	c2 = 1
	beta1 = 0
	beta2 = 0
	u1 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	u2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)	
	maxIter = itrA
	#print(W,len(W))

	for it in range(maxIter):
		#print(it)
		for doc in documentList:
			#print(doc)
			X = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
			prev_word = "_START_"
			for i in range(3,len(doc)):
				X[positionDic[doc[i]]] += 1
				X[bPositionDic[prev_word+doc[i]]+len_vocabList] += 1
				prev_word = doc[i]

			X = np.multiply(X,df_final)
			X = np.divide(X,np.linalg.norm(X))

			van1 = np.dot(W1,X) + b1
			van2 = np.dot(W2,X) + b2
			
			y1 = 1 if doc[1] == "True" else -1
			y2 = 1 if doc[2] == "Pos" else -1

			if y1*van1 <= 0:
				W1 = W1 + np.multiply(y1,X)
				b1 = b1 + y1 

			if y2*van2 <= 0:
				W2 = W2 + np.multiply(y2,X)
				b2 = b2 + y2


			avg1 = np.dot(Wa1,X) + ba1
			avg2 = np.dot(Wa2,X) + ba2

			if y1*avg1 <= 0:
				Wa1 = Wa1 + y1*X
				ba1 = ba1 + y1 
				u1 = u1 + y1*c1*X
				beta1 = beta1 + y1*c1

			if y2*avg2 <= 0:
				Wa2 = Wa2 + y2*X
				ba2 = ba2 + y2
				u2 = u2 + y2*c2*X
				beta2 = beta2 + y2*c2
			c1 = c1 + 1
			c2 = c2 + 1

	Wa1 = Wa1 - (u1/c1)
	b1 = b1 - (beta1/c1)
	Wa2 = Wa2 - (u2/c2)
	b2 = b2 - (beta2/c2)

	#print("length of positionDic:",len(positionDic))
	assert(len(positionDic)==len(vocabList))

	outputFile = open("vanillamodel.txt","w",encoding="utf-8")
	W1 = W1.tolist()
	W2 = W2.tolist()
	#print(len(W1))
	#print("POS:",len(positionDic))
	for wi in W1:
		outputFile.write(str(wi)+",")
	outputFile.write("\n")	
	outputFile.write(str(b1)+"\n")
	for wi in W2:
		outputFile.write(str(wi)+",")
	outputFile.write("\n")
	outputFile.write(str(b2)+"\n")
	for word in positionDic:
		outputFile.write(word+":"+str(positionDic[word])+"\n")
	outputFile.close()



	#print(W,len(W))

#	for it in range(maxIter):
#		#print(it)
#		for doc in documentList:
#			X = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
#			prev_word = "_START_"
#			for i in range(3,len(doc)):
#				X[positionDic[doc[i]]] += 1
#				X[bPositionDic[prev_word+doc[i]]+len_vocabList] += 1
#				prev_word = doc[i]
#
#			X = np.multiply(X,df_final)
#			X = np.divide(X,np.linalg.norm(X))
#			
#			a1 = np.dot(Wa1,X) + ba1
#			a2 = np.dot(Wa2,X) + ba2
#			#print(it, doc[0],a1,Wa1,ba1,y1)
#			
#			y1 = 1 if doc[1] == "True" else -1
#			y2 = 1 if doc[2] == "Pos" else -1
#
#			if y1*a1 <= 0:
#				Wa1 = Wa1 + y1*X
#				ba1 = ba1 + y1 
#				u1 = u1 + y1*c1*X
#				beta1 = beta1 + y1*c1
#
#			if y2*a2 <= 0:
#				Wa2 = Wa2 + y2*X
#				ba2 = ba2 + y2
#				u2 = u2 + y2*c2*X
#				beta2 = beta2 + y2*c2
#			c1 = c1 + 1
#			c2 = c2 + 1
#
#	Wa1 = Wa1 - (u1/c1)
#	b1 = b1 - (beta1/c1)
#	Wa2 = Wa2 - (u2/c2)
#	b2 = b2 - (beta2/c2)

	#print("length of positionDic:",len(positionDic))
	assert(len(positionDic)==len(vocabList))
	Wa1 = Wa1.tolist()
	Wa2 = Wa2.tolist()

	outputFile = open("averagedmodel.txt","w",encoding="utf-8")
	for wi in Wa1:
		outputFile.write(str(wi)+",")
	outputFile.write("\n")	
	outputFile.write(str(ba1)+"\n")
	for wi in Wa2:
		outputFile.write(str(wi)+",")
	outputFile.write("\n")
	outputFile.write(str(ba2)+"\n")
	for word in positionDic:
		outputFile.write(word+":"+str(positionDic[word])+"\n")
	outputFile.close()

	outputFile = open("bigrammodel.txt","w",encoding="utf-8")
	for word in bPositionDic:
		outputFile.write(word+":"+str(bPositionDic[word])+"\n")
	outputFile.close()

	#np.savetxt('vanillamodel.txt', W1, fmt='%f')
	#np.savetxt('vanillamodel.txt', b1, fmt='%f')
	#np.savetxt('vanillamodel.txt', W1, fmt='%f')
	#np.savetxt('vanillamodel.txt', b1, fmt='%f')


	#outputFile = open("nbmodel.txt","w",encoding="utf-8")
	#for key in priorProb1:
	#	outputFile.write(key+":"+str(priorProb1[key])+"\n")
	#for key in priorProb2:
	#	outputFile.write(key+":"+str(priorProb2[key])+"\n")

	#outputFile.write(":break:\n")
	#for key in likelihoodProb1:
	#	outputFile.write(key[0]+":"+key[1]+":"+str(likelihoodProb1[key])+"\n")

	#for key in likelihoodProb2:
	#	outputFile.write(key[0]+":"+key[1]+":"+str(likelihoodProb2[key])+"\n")
	#outputFile.close()

if __name__== "__main__":
	trainPercep(30,30)
