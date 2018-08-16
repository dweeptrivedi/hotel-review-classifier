# hotel-review-classifier
classify hotel reviews as either True or Fake, and either Positive or Negative


## Table of contents

- [Explanation](#explanation)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [input file format](#input-file-format)
- [Running the code](#running-the-code)
- [Authors](#authors)

## Explanation


## Prerequisites

You need to install:
- [Python3](https://www.python.org/downloads/)

## Input File Format

  **training data:**
  
  reviews should be present in following format in one file:
  
  `<review_id> <class1> <class2> <review text>`
  
  Where class1 = `Fake` or `True` and class2 = `Pos` or `Neg`
  
  **prediction data:**
  
  reviews should be present in following format in one file:
  
  `<review_id> <review text>`

## Quick-start
To start using the hotel-review-classifier you need to clone the repo:

```
git clone https://github.com/dweeptrivedi/hotel-review-classifier.git
```

## Running the code

**Naive Bayes classifier:**

  Training:
  
  `python3 nblearn3.py --data <train.txt>`
  
  Prediction:
  
  `python3 nbclassify3.py --data <test.txt>`
  
  output:
  `nboutput.txt`

**Perceptron classifier:**

  Training:
  
  `python3 perceplearn3.py --data <train.txt>`
  
  Prediction:
  
  `python3 percepclassify3.py --model <model.txt> --data <test.txt>`


  output:
  `percepoutput.txt`
  
## Authors:
* **Dweep Trivedi** - Please give me your feedback: dweeptrivedi1994@gmail.com

    Feel free to contribute

