# hmmdecode.py will use the model to tag new data

import math
import sys
import time
from collections import defaultdict
import cProfile
import re
import numpy as np

priorProb1 = defaultdict(float)
priorProb2 = defaultdict(float)
likelihoodProb1 = defaultdict(float)
likelihoodProb2 = defaultdict(float)


testFile = "coding-2-data-corpus/dev-text.txt"
modelFile = "vanillamodel.txt"


classList = [("True","Pos"),("True","Neg"),("Fake","Pos"),("Fake","Neg")]
classList1 = ["True","Fake"]
classList2 = ["Pos","Neg"]

positionDic = defaultdict(int)
bPositionDic = defaultdict(int)

#stopWords = ['few', 'how', 'such', 'doing', 'through', 'should', 'your', 'yourselves', 'again', "let's", 'whom', 'because', "i'm", 'both', "shouldn't", 'are', 'is', "he'll", "we've", 'been', 'at', 'here', "they'll", "she'll", 'into', 'below', 'under', 'those', 'be', 'them', "you'll", 'about', 'am', 'her', 'him', "mustn't", "what's", 'ourselves', 'against', 'hers', "you've", "wasn't", 'no', "we'll", 'yours', '', 'themselves', 'than', 'being', 'above', 'in', 'these', 'then', "couldn't", 'between', 'not', 'on', "he'd", 'up', 'itself', "i've", 'to', 'very', "aren't", "you'd", 'only', 'any', 'for', 'from', 'ought', "can't", 'it', 'do', "they'd", 'would', 'further', "we're", "when's", 'some', 'have', 'why', 'does', 'too', "we'd", 'if', "there's", 'the', 'so', 'their', "don't", "weren't", "you're", 'did', 'after', 'myself', 'they', 'i', 'which', "they're", 'all', 'down', "i'd", 'with', 'own', "they've", 'our', 'he', 'or', 'you', 'an', "hadn't", 'cannot', 'its', 'who', "won't", "didn't", 'when', 'a', 'during', "hasn't", 'ours', 'had', 'where', 'but', 'himself', "where's", 'until', 'was', 'nor', 'his', 'and', "she's", 'this', "why's", "how's", "wouldn't", 'each', 'off', "isn't", 'herself', "it's", 'same', 'out', 'has', "that's", "she'd", 'what', 'theirs', 'by', "here's", 'of', 'as', 'over', 'having', 'yourself', 'we', 'most', 'once', 'could', 'she', 'that', 'while', "doesn't", "haven't", 'my', 'there', 'were', "i'll", 'me', "who's", "he's", 'before', "shan't", 'other', 'more']
stopWords = []

def classifyPercep(testFile="coding-2-data-corpus/dev-text.txt",modelFile="vanillamodel.txt"):

	if len(sys.argv)>2:
		modelFile = sys.argv[1]
		testFile = sys.argv[2]

	flag = False
	inputFile = open(modelFile,"r",encoding="utf-8")
	lines = inputFile.readlines()
	
	lines[0] = lines[0].strip()[:-1].split(",")
	lines[1] = lines[1].strip()
	lines[2] = lines[2].strip()[:-1].split(",")
	lines[3] = lines[3].strip()

	W1 = np.array(lines[0],dtype=np.float64)
	b1 = float(lines[1])
	W2 = np.array(lines[2],dtype=np.float64)
	b2 = float(lines[3])
	for i in range(4,len(lines)):
		[word,pos] = lines[i].split(":")
		positionDic[word] = int(pos)

	inputFile.close()

	inputFile = open("bigrammodel.txt","r",encoding="utf-8")
	lines = inputFile.readlines()
	for line in lines:
		[word,pos] = line.strip().split(":")
		bPositionDic[word] = int(pos)
	inputFile.close()

	df1 = np.load("df1.npy")
	dfb1 = np.load("dfb1.npy")
	df_final = np.concatenate([df1,dfb1])

	inputFile = open(testFile,"r",encoding="utf-8")
	outputFile = open("percepoutput.txt","w",encoding="utf-8")
	lines = inputFile.readlines()

	for line in lines:
		document = line.strip()
		documentID = document.split(' ')[0]

		document = document.lower()
		document = re.sub('[?.!/;:,"]', '', document)
		document = re.sub("[{}()']", '', document)
		document = document.split(' ')
		document = document[1:]

		X = np.zeros(len(dfb1)+len(df1),dtype=np.float64)
		prev_word = "_START_"
		for i in range(len(document)):
			if document[i] in positionDic:
				#if document[i] not in stopWords:
				X[positionDic[document[i]]] += 1
				if prev_word+document[i] in bPositionDic:
					X[bPositionDic[prev_word+document[i]]+len(positionDic)] += 1
				prev_word = document[i]

		X = np.multiply(X,df_final)
		X = np.divide(X,np.linalg.norm(X))


		a1 = np.dot(W1,X)+b1
		a2 = np.dot(W2,X)+b2
		#print(a1)
		#print(a2)

		maxClass1 = classList1[0] if a1>0 else classList1[1]
		maxClass2 = classList2[0] if a2>0 else classList2[1]

		outputFile.write(documentID+" "+maxClass1+" "+maxClass2+"\n")

	inputFile.close()
	outputFile.close()



if __name__== "__main__":
	classifyPercep()