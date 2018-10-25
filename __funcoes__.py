#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:34:21 2018

@author: anarocha
"""

def processa_texto(texto):
        textoProcessado = BeautifulSoup(texto, 'html.parser').string
        #TODO: ainda é preciso remover as tags XML e word....
        textoProcessado = normalize('NFKD', textoProcessado).encode('ASCII','ignore').decode('ASCII')
        textoProcessado = re.sub('[^a-zA-Z]',' ',textoProcessado)
        textoProcessado = textoProcessado.lower()
        textoProcessado = textoProcessado.split()
        textoProcessado = [palavra for palavra in textoProcessado if not palavra in stopwords]
        textoProcessado = [palavra for palavra in textoProcessado if len(palavra)>3]
        textoProcessado =  [stemmer.stem(palavra) for palavra in textoProcessado]
        return textoProcessado