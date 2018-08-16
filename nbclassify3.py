# hmmdecode.py will use the model to tag new data

import math
import sys
import time
from collections import defaultdict
import cProfile
import re
import argparse

priorProb1 = defaultdict(float)
priorProb2 = defaultdict(float)
likelihoodProb1 = defaultdict(float)
likelihoodProb2 = defaultdict(float)



classList = [("True","Pos"),("True","Neg"),("Fake","Pos"),("Fake","Neg")]
classList1 = ["True","Fake"]
classList2 = ["Pos","Neg"]

def main(testFile="test.txt"):

	flag = False
	inputFile = open("nb.model","r",encoding="utf-8")
	lines = inputFile.readlines()

	for line in lines:
		line = line.strip()
		if line==":break:":
			flag = True
			continue

		if flag == False:
			#read prior probabilities
			[class1,prob] = line.split(':')
			priorProb1[class1] =  float(prob)
		else:
			#read likelihood probabilities
			[feature,class1,prob] = line.split(':')
			likelihoodProb1[(feature,class1)] = float(prob)

	inputFile.close()


	inputFile = open(testFile,"r",encoding="utf-8")
	outputFile = open("nboutput.txt","w",encoding="utf-8")
	lines = inputFile.readlines()

	for line in lines:
		document = line.strip()
		documentID = document.split(' ')[0]

		document = document.lower()
		document = re.sub('[?.!/;:,"]', '', document)
		document = re.sub("[{}()']", '', document)
		document = document.split(' ')
		document = document[1:]

		maxClass1 = ""
		maxProb1 = -1.0 * float(sys.float_info.max)
		#prob = posterior probability of a document given class c
		for c in classList1:
			prob = priorProb1[c]
			for word in document:
				prob += likelihoodProb1[(word,c)]

			if prob >= maxProb1:
				maxClass1 = c
				maxProb1 = prob

		maxClass2 = ""
		maxProb2 = -1.0 * float(sys.float_info.max)
		#prob = posterior probability of a document given class c
		for c in classList2:
			prob = priorProb1[c]
			for word in document:
				prob += likelihoodProb1[(word,c)]

			if prob >= maxProb2:
				maxClass2 = c
				maxProb2 = prob

		outputFile.write(documentID+" "+maxClass1+" "+maxClass2+"\n")

	inputFile.close()
	outputFile.close()



if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', help="path to review file for training", type=str)
	args = parser.parse_args()
	main(args.data)
