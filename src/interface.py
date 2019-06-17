# -*- coding: utf-8 -*-
from src.logistic_regression import *
from src.neural_network import *
import numpy as np
import os, re, operator
import nltk
from nltk.corpus import stopwords

REMOVE_BELOW_THRESHOLD = 3
MIN_GRAM = 1
MAX_GRAM = 2

ZERO = 0
ENGLISH = 'english'
STOPWORDS = 'stopwords'

# Download do dicionário de stop wordsgreat
nltk.download(STOPWORDS)

def stopwords_removal(tokens_list):
	'''
		Dado uma lista de tokens.
		Então é removido as stopwords.

		Entrada: tokens_list - Uma lista de tokens.
		Saida: new_tokens_list - Uma lista de tokens.
	'''
	if len(tokens_list) == ZERO:
		raise Exception('Error on stopwords_removal.')

	new_tokens_list = []

	stop_words = stopwords.words(ENGLISH)
	new_tokens_list = [token for token in tokens_list if token not in stop_words]

	if len(new_tokens_list) == ZERO:
		raise Exception('Error on stopwords_removal.')

	return new_tokens_list

def generate_ngram(tokens_list):
    '''
		Dado uma lista de tokens.
		Então é gerado N-Gram com base no MAX_GRAM e MIN_GRAM.

		Entrada: tokens_list - Uma lista de tokens.
		Saida: allNGrams - Uma lista de N-Gram.
    '''
    if len(tokens_list) == ZERO:
        raise Exception('Error on generate_ngram.')
        
    allNGrams = []
    
    #N-Gram
    for idx in range(MIN_GRAM, MAX_GRAM + 1):
        ngrams = zip(*[tokens_list[i:] for i in range(idx)])
        allNGrams += ([" ".join(ngram) for ngram in ngrams])
    
    if len(allNGrams) == ZERO:
        raise Exception('Error on generate_ngram.')
        
    return allNGrams

def generateFeatures(text, listFeatures, stopwords=False):
	'''
		Dado um texto.
		Então o texto é filtrado pelo numero do threshold
			e é gerado uma matriz que faz a contagem das features.

		Entrada: text - Texto a ser analizado para a geração das features.
				 listFeatures - Uma lista de features.
				 stopwords - Um boolean indicando se é para utilizar este método.
		Saida: X - Matriz com as features calculadas.
	'''
	X = []
	if len(text) == ZERO:
		raise Exception('Error on generateFeatures.')

	row = np.zeros(len(listFeatures))
	tokens = []
	tokens = [word for word in (re.sub(r'[^\w\s]+','', text.replace('\n','')).lower().split(' ')) if word != '']

	if stopwords:
		tokens = stopwords_removal(tokens)

	tokens = generate_ngram(tokens)

	for token in tokens:
		for idx, feature in enumerate(listFeatures):
			if token == feature:
				row[idx] += 1

	X.append(row)
	return np.array(X)

def predPositive(theta1Positive, theta2Positive, X):
	'''
        Dado uma matriz de features e os thetas.
        Então o texto é classificado.
        
        Entrada: theta1Positive - Theta 1.
                 theta2Positive - Thetha 2.
                 X - Matriz de features para o texto de entrada.
        Saida: Predição - resultado da predição.
    '''
	return rna_predicao(theta1Positive, theta2Positive, X)

def predNegative(thetaNegative, X):
	'''
        Dado uma matriz de features e o theta.
        Então o texto é classificado.
        
        Entrada: thetaNegative - Theta.
                 X - Matriz de features para o texto de entrada.
        Saida: Predição - resultado da predição.
    '''
	return predicao(thetaNegative, X)

def predBoth(theta1Both, theta2Both, X):
	'''
        Dado uma matriz de features e os thetas.
        Então o texto é classificado.
        
        Entrada: theta1Both - Theta 1.
                 theta2Both - Thetha 2.
                 X - Matriz de features para o texto de entrada.
        Saida: Predição - resultado da predição.
    '''
	return rna_predicao(theta1Both, theta2Both, X)

def preProcessing(text, features, removeStopWord):
	'''
        Dado um texto.
        Então o texto é pre processado.
        
        Entrada: text - Texto a ser analisado.
				 features - Lista de features.
                 removeStopWord - Boolean indicando se tem que haver remoção de stopwords.
        Saida: X - Uma matriz com as features.
    '''
	return generateFeatures(text, features, removeStopWord)