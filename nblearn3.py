# hmmlearn3.py will learn Naive Bayes classifier from the training data to identify hotel reviews as either true or fake, and either positive or negative

#prior prob: p(c)


#likelihood prob: p(fi/c)


import math
import sys
from collections import defaultdict
import copy
import re
import argparse

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

#List of possible classes:
classList = [("True","Pos"),("True","Neg"),("Fake","Pos"),("Fake","Neg")]
classList1 = ["True","Fake"]
classList2 = ["Pos","Neg"]

#with stopwords: 0.9139, without stopwords: 0.9045
#FIXME: If I do something for unknown words in test dataset then I need to use the same list in nbclassify3.py also
stopWords = ['few', 'how', 'such', 'doing', 'through', 'should', 'your', 'yourselves', 'again', "let's", 'whom', 'because', "i'm", 'both', "shouldn't", 'are', 'is', "he'll", "we've", 'been', 'at', 'here', "they'll", "she'll", 'into', 'below', 'under', 'those', 'be', 'them', "you'll", 'about', 'am', 'her', 'him', "mustn't", "what's", 'ourselves', 'against', 'hers', "you've", "wasn't", 'no', "we'll", 'yours', '', 'themselves', 'than', 'being', 'above', 'in', 'these', 'then', "couldn't", 'between', 'not', 'on', "he'd", 'up', 'itself', "i've", 'to', 'very', "aren't", "you'd", 'only', 'any', 'for', 'from', 'ought', "can't", 'it', 'do', "they'd", 'would', 'further', "we're", "when's", 'some', 'have', 'why', 'does', 'too', "we'd", 'if', "there's", 'the', 'so', 'their', "don't", "weren't", "you're", 'did', 'after', 'myself', 'they', 'i', 'which', "they're", 'all', 'down', "i'd", 'with', 'own', "they've", 'our', 'he', 'or', 'you', 'an', "hadn't", 'cannot', 'its', 'who', "won't", "didn't", 'when', 'a', 'during', "hasn't", 'ours', 'had', 'where', 'but', 'himself', "where's", 'until', 'was', 'nor', 'his', 'and', "she's", 'this', "why's", "how's", "wouldn't", 'each', 'off', "isn't", 'herself', "it's", 'same', 'out', 'has', "that's", "she'd", 'what', 'theirs', 'by', "here's", 'of', 'as', 'over', 'having', 'yourself', 'we', 'most', 'once', 'could', 'she', 'that', 'while', "doesn't", "haven't", 'my', 'there', 'were', "i'll", 'me', "who's", "he's", 'before', "shan't", 'other', 'more']



def main(trainFile="train.txt"):

	inputFile = open(trainFile,"r",encoding="utf-8")
	lines = inputFile.readlines()
	inputFile.close()
	for line in lines:
		document = line.strip()
		[class1,class2] = document.split(' ')[1:3]

		document = document.lower()
		document = re.sub('[?.!/;:,"]', '', document)
		document = re.sub("[{}()']", '', document)
		document = document.split(' ')
		wordList = document[3:]
		
		#increase the count of documents for found class
		priorDic1[class1] += 1
		priorDic2[class2] += 1

		for word in wordList:
			if word in  stopWords:
				continue
			likelihoodDic1[(word,class1)] += 1
			likelihoodDic2[(word,class2)] += 1
			classLength1[class1] += 1
			classLength2[class2] += 1
			#FIXME: doesn't make sense to have vocabList1 and vocabList2, revist
			vocabList1.add(word)
			vocabList2.add(word)
			wordFreq[word] += 1


	#calculating prior probabilities:
	#2-binary classifiers
	N1 = 0.0
	for key in priorDic1:
		N1 += priorDic1[key]
	N2 = 0.0
	for key in priorDic2:
		N2 += priorDic2[key]

	#prior probability: 2-binary classifiers
	for key in priorDic1:
		priorProb1[key] = math.log(priorDic1[key]/N1)
	for key in priorDic2:
		priorProb2[key] = math.log(priorDic2[key]/N2)


	#calculating likelihood probabilities:
	#add k smoothing with k=0.5
	k = 0.5
	V1 = len(vocabList1)
	for fi in vocabList1:
		for c in classList1:
			likelihoodProb1[(fi,c)] = math.log((likelihoodDic1[(fi,c)]+k)/(classLength1[c]+k*V1))

	V2 = len(vocabList2)
	for fi in vocabList2:
		for c in classList2:
			likelihoodProb2[(fi,c)] = math.log((likelihoodDic2[(fi,c)]+k)/(classLength2[c]+k*V2))	


	outputFile = open("nb.model","w",encoding="utf-8")
	for key in priorProb1:
		outputFile.write(key+":"+str(priorProb1[key])+"\n")
	for key in priorProb2:
		outputFile.write(key+":"+str(priorProb2[key])+"\n")

	outputFile.write(":break:\n")
	for key in likelihoodProb1:
		outputFile.write(key[0]+":"+key[1]+":"+str(likelihoodProb1[key])+"\n")

	for key in likelihoodProb2:
		outputFile.write(key[0]+":"+key[1]+":"+str(likelihoodProb2[key])+"\n")
	outputFile.close()

if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', help="path to review file for training", type=str)
	args = parser.parse_args()
	main(args.data)
