#!/usr/bin/env python
# coding: utf-8
from flask import Flask, render_template, jsonify, request
from src.interface import *
from random import *
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/example', methods=['POST'])
def example():
	result = {}
	
	posNeg = 'Positive' if randint(1,2) == 1 else 'Negative'
	truDec = 'Truthful' if randint(1,2) == 1 else 'Deceptive'
	file = randint(1,10)
	
	path = './Examples/{}/{}/{}.txt'.format(posNeg, truDec, file)
	with open(path, 'r+', encoding='utf8') as stream:
		exampleText = stream.read()
		category = '{}, {}'.format(posNeg, truDec)
		result['category'] = category
		result['example'] = exampleText
		return jsonify(result)

@app.route('/compute', methods=['POST'])
def compute():
	req = request.form.to_dict()
	result = {}
	result['status'] = 'OK'
	
	if req['text'] == '':
		result['status'] = 'INVALID_REQUEST'
		return jsonify(result)
	
	if req['type'] == 'positive':
		features = np.load('./Features/Fetures_POS.npy')
		X = preProcessing(req['text'], features.tolist(), False)
		theta1 = np.load('./Thetas/POS_Theta1.npy')
		theta2 = np.load('./Thetas/POS_Theta2.npy')
		result['predicao'] = str(predPositive(theta1, theta2, X)[0])

	elif req['type'] == 'negative':
		features = np.load('./Features/Fetures_NEG.npy')
		X = preProcessing(req['text'], features.tolist(), False)
		theta = np.load('./Thetas/NEG_Theta.npy')
		X = np.column_stack((np.ones(X.shape[0]), X))
		result['predicao'] = str(predNegative(theta, X)[0])

	elif req['type'] == 'both':
		features = np.load('./Features/Fetures_POSNEG.npy')
		X = preProcessing(req['text'], features.tolist(), True)
		theta1 = np.load('./Thetas/POSNEG_Theta1.npy')
		theta2 = np.load('./Thetas/POSNEG_Theta2.npy')
		result['predicao'] = str(predBoth(theta1, theta2, X)[0])
	else:
		result['status'] = 'INVALID_REQUEST'

	return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)