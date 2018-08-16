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
import argparse


#set of vocabulary words in training data
vocabList1 = set()
vocabList2 = set()

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


documentList = []

positionDic = defaultdict(int)
bPositionDic = defaultdict(int)


def trainPercep(trainFile="train.txt", itrV=25, itrA=41):

	inputFile = open(trainFile,"r",encoding="utf-8")
	lines = inputFile.readlines()
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
		
		prev_word = "_START_"
		bigrams = set()

		for word in wordList:
			tempDoc.append(word)
			vocabList1.add(word)
			vocabList2.add(prev_word+word)
			bigrams.add(prev_word+word)
			wordFreq[word] += 1
			prev_word = word

		for word in set(wordList):
			df[word] += 1

		for b in bigrams:
			dfb[b] += 1

		documentList.append(tempDoc)



	
	vocabList = list(vocabList1)
	vocabList.sort()
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


	assert(len(positionDic)==len(vocabList))

	df1 = np.zeros(len_vocabList,dtype=np.float64)
	dfb1 = np.zeros(len_bigramList,dtype=np.float64)

	for i in range(len_vocabList):
		df1[positionDic[vocabList[i]]] = 1+math.log((len(lines)+1)/(1+df[vocabList[i]]))

	for i in range(len_bigramList):
		dfb1[bPositionDic[bigramList[i]]] = 1+math.log((len(lines)+1)/(1+dfb[bigramList[i]]))

	df_final = np.concatenate([df1,dfb1])

	np.save("df1.npy",df1)
	np.save("df_final.npy",df_final)

	W1 = np.zeros(len_vocabList,dtype=np.float64)
	W2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	b1 = 0
	b2 = 0

	Wa1 = np.zeros(len_vocabList,dtype=np.float64)
	Wa2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
	ba1 = 0
	ba2 = 0
	c1 = 1
	c2 = 1
	beta1 = 0
	beta2 = 0
	u1 = np.zeros(len_vocabList,dtype=np.float64)
	u2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)	

	X1List = []
	Y1List = []
	X2List = []
	for doc in documentList:
		X1 = np.zeros(len_vocabList,dtype=np.float64)
		X2 = np.zeros(len_vocabList+len_bigramList,dtype=np.float64)
		prev_word = "_START_"
		for i in range(3,len(doc)):
			X1[positionDic[doc[i]]] += 1
			X2[positionDic[doc[i]]] += 1
			X2[bPositionDic[prev_word+doc[i]]+len_vocabList] += 1
			prev_word = doc[i]

		X1 = np.multiply(X1,df1)
		X1 = np.divide(X1,np.linalg.norm(X1))
		X2 = np.multiply(X2,df_final)
		X2 = np.divide(X2,np.linalg.norm(X2))

		X1List.append(X1)
		X2List.append(X2)
		Y1List.append([doc[1],doc[2]])


	maxIter = max(itrV,itrA)
	for it in range(maxIter):
		for i in range(len(X1List)):
			X1 = X1List[i]
			X2 = X2List[i]
				
			y1 = 1 if Y1List[i][0] == "True" else -1
			y2 = 1 if Y1List[i][1] == "Pos" else -1

			if it < itrV:
				van1 = np.dot(W1,X1) + b1
				van2 = np.dot(W2,X2) + b2
			
				if y1*van1 <= 0:
					W1 = W1 + y1*X1
					b1 = b1 + y1 

				if y2*van2 <= 0:
					W2 = W2 + y2*X2
					b2 = b2 + y2

			if it < itrA:
				avg1 = np.dot(Wa1,X1) + ba1
				avg2 = np.dot(Wa2,X2) + ba2

				if y1*avg1 <= 0:
					Wa1 = Wa1 + y1*X1
					ba1 = ba1 + y1 
					u1 = u1 + y1*c1*X1
					beta1 = beta1 + y1*c1

				if y2*avg2 <= 0:
					Wa2 = Wa2 + y2*X2
					ba2 = ba2 + y2
					u2 = u2 + y2*c2*X2
					beta2 = beta2 + y2*c2
				c1 = c1 + 1
				c2 = c2 + 1

	Wa1 = Wa1 - (u1/c1)
	ba1 = ba1 - (beta1/c1)
	Wa2 = Wa2 - (u2/c2)
	ba2 = ba2 - (beta2/c2)

	
	outputFile = open("vanilla.model","w",encoding="utf-8")
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


	outputFile = open("averaged.model","w",encoding="utf-8")
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


if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', help="path to review file for training", type=str, default="train.txt")
	args = parser.parse_args()
	trainPercep(args.data, 25, 41)
