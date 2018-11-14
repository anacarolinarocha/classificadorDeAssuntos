#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:08:24 2018

@author: anarocha
"""

class Modelo:
    def __init__(self, nome, best_params_, best_estimator_, grid_scores_, macro_precision, macro_recall, macro_fscore, micro_precision, micro_recall, micro_fscore):  
        self.nome = nome
        self.best_params_ = best_params_
        self.best_estimator_ = best_estimator_
        self.grid_scores_ = grid_scores_
        self.macro_precision = macro_precision
        self.macro_recall = macro_recall
        self.macro_fscore = macro_fscore
        self.micro_precision = micro_precision
        self.micro_recall = micro_recall
        self.micro_fscore = micro_fscore
     
    def setNome(self, nome):
        self.nome = nome
     
    def setBestParams(self, best_params_):
        self.best_params_ = best_params_
    
    def setBestEstimator(self, best_estimator_):
        self.best_estimator_ = best_estimator_
        
    def setGridScores(self, grid_scores_):
        self.grid_scores_ = grid_scores_
        
    def setMacroPrecision(self, macro_precision):
        self.macro_precision = macro_precision
     
    def setMacroRecall(self, macro_recall):
        self.macro_recall = macro_recall
        
    def setMacroFscore(self, macro_fscore):
        self.macro_fscore = macro_fscore
        
    def setMicroPrecision(self, micro_precision):
        self.micro_precision = micro_precision
     
    def setMicroRecall(self, micro_recall):
        self.micro_recall = micro_recall
        
    def setMicroFscore(self, micro_fscore):
        self.micro_fscore = micro_fscore
     
    def getNome(self):
        return self.nome
         
    def getBestParams(self):
        return self.best_params_
    
    def getBestEstimator(self):
        return self.best_estimator_ 
    
    def getGridScores(self, grid_scores_):
        return self.grid_scores_ 
    
    def getMacroPrecision(self):
        return self.macro_precision 
     
    def getMacroRecall(self):
        return self.macro_recall 
        
    def getMacroFscore(self):
        return self.macro_fscore 
        
    def getMicroPrecision(self):
        return self.micro_precision 
     
    def getMicroRecall(self):
        return self.micro_recall 
        
    def getMicroFscore(self):
        return self.micro_fscore 
        
 