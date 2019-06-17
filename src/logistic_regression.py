# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(z):
    """
    Calcula a funcao sigmoidal  
    """
    
    if isinstance(z, int):
        g = 0
    
    else:
        g = np.zeros( z.shape );

    g = 1 / (1 + np.exp(-z))
    
    return g

def funcaoCusto(theta, X, Y):
    """
    Calcula o custo da regressao logística
    
       J = COMPUTARCUSTO(X, y, theta) calcula o custo de usar theta como 
       parametro da regressao logistica para ajustar os dados de X e y    
    """
    
    m = len(Y) 
    J = 0;
    grad = np.zeros( len(theta) );
    eps = 1e-15
    
    J = (- Y * np.log(sigmoid((X * theta).sum(axis=1)) + eps) - (1 - Y) * np.log(1 - sigmoid((X * theta).sum(axis=1)) + eps)).sum() / m
    grad = ((sigmoid((X * theta).sum(axis=1)) - Y) * X.T).sum(axis=1) / m
    
    return J, grad


def predicao(theta, X):
    """
    Prediz se a entrada pertence a classe 0 ou 1 usando o parametro
    theta obtido pela regressao logistica
    
    p = PREDICAO(theta, X) calcula a predicao de X usando um 
    limiar igual a 0.5 (ex. se sigmoid(theta'*x) >= 0.5, classe = 1)
    """   
    
    m = X.shape[0]
    p = np.zeros(m, dtype=int) 

    pred = lambda x: 1 if x >= 0.5 else 0
    p = [pred(i) for i in sigmoid((X * theta).sum(axis=1))]
    
    return p