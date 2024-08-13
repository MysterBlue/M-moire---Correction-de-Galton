""" Programme qui fait une comparaison du portefeuille à la variance minimal, le portefeuille optimale de Markowitz et le portefeuille de talmud
estime a partir de la methode classique et de la correction de galton."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import Markowitz as mk
import LoadData as ld
import statistics as st
from sklearn.linear_model import LinearRegression
import time
import statsmodels.api as sm
import itertools

#Charger les fichier
fichier = ld.choisir_fichiers()
#Cree un dictionnaire avec le nom de l'actif comme KEY et une dataframe de ce qu'on a besoin comme ITEM
dico = ld.lire_fichiers(fichier, ["Date", "Close"])

#Actif = ["AMZN", "ASML", "BRK-B", "CAT", "CVX", "DIS", "FDX", "GE", "GOOGL", "JPM", "KO", "LLY", "MSFT", "NVDA", "NVO", "ORCL", "PFE", "REGN", "XOM"] #AJOUTER ACTIF
Actif = ["AMZN", "ASML", "BRK-B", "CAT", "DIS", "GE", "GOOGL", "JPM", "KO", "MSFT", "NVO", "ORCL", "PFE", "REGN", "XOM"] #AJOUTER ACTIF
#Actif = ["AAPL", "ADBE", "AKAM", "AMD", "AMZN", "BRK-B", "CAG", "CAT", "CPB", "DIS", "ESS", "GE", "HD", "IBM", "INTC", "IPG", "JPM", 
#          "KMB", "KO", "LLY", "LNT", "LOW", "MRK", "MSFT", "NFLX", "NVDA", "NVO", "PFE","ROST", "SLB","SO", "SYY", "T", "TAP", "TXN", "VTR", "WDC", "WMB", "XEL", "XOM"]
#Actif = ["AAPL", "ADBE", "AMZN", "DIS", "ESS", "GE", "HD", "IBM", "INTC", "IPG","KO", "LLY", "LNT", "LOW", "MRK", "MSFT", "NFLX", "NVDA", "NVO", "SLB","SO", "SYY", "T", "TAP", "TXN",
#                   "VTR", "WDC", "WMB", "XEL", "XOM"]

#Actif = ["AAPL", "ADBE", "AKAM", "AMD", "AMZN", "BRK-B", "CAG", "CAT", "CPB", "DIS", "ESS", "GE", "HD", "IBM", "INTC", "IPG", "JPM", 
#          "KMB", "KO", "LLY", "LNT", "LOW", "MRK", "MSFT", "MTB", "MU", "NEM",  "NFLX", "NVDA", "NVO", "PFE","ROST","SBAC", "SJM",
#            "SLB","SO", "SYY", "T", "TAP", "TXN", "TXT", "USB", "VTR", "WDC", "WMB", "XEL", "XOM", "WFC", "WYNN", "ZION"]
print(len(Actif))



#Calcul le rendement des actifs
for key, item in dico.items():

    item["Close"] = item["Close"].pct_change()
    item = item.drop(index=0)
    item = item.reset_index(drop=True)
    item.loc[:,'Date'] = pd.to_datetime(item["Date"])
    item.set_index('Date', inplace = True)
    dico[key] = item


#Nom des titres
Name_tot = list(dico.keys())

#FENETRE HISTORIQUE ET EX POST
Hist = 60 #int(input("Indiquez la fenetre historique H: "))
Fut =  12 #int(input("Indiquez la fenetre future E: "))
L = 100 #int(input("Indiquez la periode d'apprendissage L: "))
annee = 10 #int(input("Indiquez la période à comparer : "))



df_rendement = pd.DataFrame()
#Cree une dataframe general
for df in Name_tot:
    df_rendement = pd.concat([df_rendement, dico[df]], axis = 1)

df_rendement.columns = Name_tot

count_tot = list(df_rendement.count(axis = 1))
k = 0
i = 0
#supprime les lignes de la dataframe jusq'au moment nous avons pour la premiere fois 30 donnees sur la ligne.
while i <= len(count_tot):
    if count_tot[i] < 30:
        k = k+1
        i = i+1
    else:
        break
nb_row = len(df_rendement.index)
#Ajuster la periode d'apprentissage L. METTRE .loc[nb_row-7 - annee * fut - Hist - L : nb_row - 7]
df_rendement = df_rendement.reset_index(drop = True).loc[nb_row-7 - (annee) * Fut - L- Hist - Fut: nb_row - 7].reset_index(drop = True)

#supprime les actifs qui possèdent moins de Hist + Fut données
nb_isna = df_rendement.isna().sum()
nb_row = len(df_rendement.index)
for i in Name_tot:
    if nb_row - nb_isna[i] < Hist + Fut:
        df_rendement.drop(i, axis = 1, inplace = True)

print(df_rendement)
Name = list(df_rendement.columns)
#Donne la liste des actifs utilises dans notre portefeuille qui ne possede pas toutes les donnees necessaires sur la periode de comparaison 
for i in Actif:
    if df_rendement[i].tail(Fut * (annee +1) - Hist-12).isna().sum() > 0:
        print(i, df_rendement[i].tail(Fut * (annee +1) - Hist).isna().sum())


g0_m = 0
g1_m = 0
g0_c = 0
g1_c = 0
j = 1
#combinai = list(itertools.combinations(Name, 2))

mu_marko_est = {}
mu_true = {}
Cov_marko_est = {}
Cov_true = {}

mu_galton_est = {}
Cov_galton_est = {}
periode = []


t = 1
#Calcule les coefficients de Galton pour les covariance et les rendements et calcule les estimations classiques et corrigees sur la periode voulue
for i in range(Hist + Fut -1, nb_row- Fut):
    mean_hist  = []
    mean_fut = []
    for k in Name:
        if df_rendement[k].iloc[i-Hist - Fut + 1:i -Fut+1].isna().sum() == 0 and df_rendement[k].loc[i- Fut+ 1: i+1].isna().sum() == 0:
             mean1 = df_rendement[k].iloc[i-Hist - Fut + 1:i -Fut+1].mean()
             mean2 = df_rendement[k].iloc[i- Fut+ 1: i+1].mean()
             mean_hist.append(mean1)
             mean_fut.append(mean2)
                     
    model = LinearRegression(fit_intercept=True)
    model.fit(np.array(mean_hist).reshape(-1, 1), np.array(mean_fut))
    g0_m = ((j-1) * g0_m + model.intercept_)/j
    g1_m = ((j-1) * g1_m + model.coef_[0])/j
    
    
    cor_hist  = []
    cor_fut = []
    Matrix_hist = df_rendement.iloc[i-Hist - Fut + 1:i -Fut+1].cov(min_periods= Hist)
    Matrix_fut = df_rendement.iloc[i- Fut+ 1: i+1].cov(min_periods=Fut)
    for col1 in Name:
        for col2 in Name:
            if pd.notna(Matrix_hist.loc[col1, col2]) and pd.notna(Matrix_fut.loc[col1, col2]):
                cor_hist.append(Matrix_hist.loc[col1, col2])
                cor_fut.append(Matrix_fut.loc[col1, col2])
    model2 = LinearRegression(fit_intercept=True)
    model2.fit(np.array(cor_hist).reshape(-1, 1), np.array(cor_fut))
    g0_c = ((j-1) * g0_c + model2.intercept_)/j
    g1_c = ((j-1) * g1_c + model2.coef_[0])/j
    j = j+1
    #Calcul des estimateurs sur la periode choisie
    if i >= nb_row-Fut*(annee +1):
        mu_marko_est_temp = []
        mu_true_temp = []
        for k in Actif:
            mu_marko_est_temp.append(df_rendement[k].iloc[i-Hist+1:i+1].mean())
            mu_true_temp.append(df_rendement[k][i+1])
        mu_marko_est[t] = np.array(mu_marko_est_temp).reshape(1, len(Actif))
        mu_true[t] = np.array(mu_true_temp).reshape(1, len(Actif))
        Cov_marko_est[t] = df_rendement[Actif].iloc[i-Hist+1:i+1].cov(min_periods= Hist)
        mu_galton_est[t] = np.array([g0_m + g1_m * i for i in mu_marko_est_temp]).reshape(1, len(Actif))
        Cov_galton_est_temp = df_rendement[Actif].iloc[i-Hist+1:i+1].cov(min_periods= Hist)
        for m in range(Cov_galton_est_temp.shape[0]):
            for n in range(Cov_galton_est_temp.shape[1]):
                Cov_galton_est_temp.iloc[m, n] = g0_c + g1_c * Cov_galton_est_temp.iloc[m, n]
        Cov_galton_est[t] = Cov_galton_est_temp
        t = t+1



#possible d'optimiser ce qui suit via ce dico. Mais je ne suis pas parvenu.
variables = {
    'mu_min_maro', 'std_min_maro', 'mu_min_galton', 'std_min_galton',
    'mu_min__marko_true', 'std_min_marko_true', 'mu_min_galton_true', 'std_min_galton_true',
    'mu_m_maro', 'std_m_maro', 'mu_m_galton', 'std_m_galton',
    'mu_m_marko_true', 'std_m_marko_true', 'mu_m_galton_true', 'std_m_galton_true',
    'mu_tal_maro', 'std_tal_maro', 'mu_tal_galton', 'std_tal_galton',
    'mu_tal_marko_true', 'std_tal_marko_true', 'mu_tal_galton_true', 'std_tal_galton_true'
}



mu_min_maro = []
mu_min_galton = []
mu_min_true = []
std_min_maro = []
std_min_galton = []
std_min_marko_true = []
std_min_galton_true = []

mu_m_maro = []
mu_m_galton = []
mu_min__marko_true = []
mu_min_galton_true = []
std_m_maro = []
std_m_galton = []
std_m_marko_true = []
mu_m_marko_true = []
mu_m_galton_true = []
std_m_galton_true = []

mu_tal_maro = []
std_tal_maro = []
mu_tal_galton = []
std_tal_galton = []
mu_tal_marko_true = []
mu_tal_galton_true = []
std_tal_marko_true = []
std_tal_galton_true = []



Res = {}
#Classe les données par années
for i in list(Cov_galton_est.keys()):
    #Portefeuille minimal
    w_min = mk.MinimalVariance(Cov_marko_est[i], len(Actif))
    w_min_corrige = mk.MinimalVariance(Cov_galton_est[i], len(Actif))
    #print(w_min, w_min_corrige)
    mu_min_maro.append(mk.esperance(mu_marko_est[i], w_min))
    std_min_maro.append(mk.risque(w_min, Cov_marko_est[i]))
    
    mu_min_galton.append(mk.esperance(mu_galton_est[i], w_min_corrige))
    std_min_galton.append(mk.risque(w_min_corrige, Cov_galton_est[i]))
    mu_min__marko_true.append(mk.esperance(mu_true[i], w_min))
    mu_min_galton_true.append(mk.esperance(mu_true[i], w_min_corrige))
    #std_min_marko_true.append(mk.risque(w_min, Cov_true[i]))
    #std_min_galton_true.append(mk.risque(w_min_corrige, Cov_true[i]))

    #portefueille optimal
    w_m = mk.MarketPortfolio(mu_marko_est[i], Cov_marko_est[i], len(Actif))
    w_m_corrige = mk.MarketPortfolio(mu_galton_est[i], Cov_galton_est[i], len(Actif))
    mu_m_maro.append(mk.esperance(mu_marko_est[i], w_m))
    std_m_maro.append(mk.risque(w_m, Cov_marko_est[i]))
    
    mu_m_galton.append(mk.esperance(mu_galton_est[i], w_m_corrige))
    std_m_galton.append(mk.risque(w_m_corrige, Cov_galton_est[i]))
    mu_m_marko_true.append(mk.esperance(mu_true[i], w_m))
    mu_m_galton_true.append(mk.esperance(mu_true[i], w_m_corrige))
    #std_m_marko_true.append(mk.risque(w_m, Cov_true[i]))
    #std_m_galton_true.append(mk.risque(w_m_corrige, Cov_true[i]))

    #portefeuille talmud
    w_tal = np.ones(len(Actif)).reshape(1, len(Actif))/len(Actif)
    mu_tal_maro.append(mk.esperance(mu_marko_est[i], w_tal))
    std_tal_maro.append(mk.risque(w_tal, Cov_marko_est[i]))
    mu_tal_galton.append(mk.esperance(mu_galton_est[i], w_tal))
    std_tal_galton.append(mk.risque(w_tal, Cov_galton_est[i]))

    mu_tal_marko_true.append(mk.esperance(mu_true[i], w_tal))
    mu_tal_galton_true.append(mk.esperance(mu_true[i], w_tal))
    #std_tal_marko_true.append(mk.risque(w_tal, Cov_true[i]))
    #std_tal_galton_true.append(mk.risque(w_tal, Cov_true[i]))

#classe les rendements mensuels par annee.
mu_min_miro = [mu_min_maro[i:i + 12] for i in range(0, len(mu_min_maro), 12)]
std_min_maro = [std_min_maro[i:i + 12] for i in range(0, len(std_min_maro), 12)]

mu_min_galton = [mu_min_galton[i:i + 12] for i in range(0, len(mu_min_galton), 12)]
std_min_galton = [std_min_galton[i:i + 12] for i in range(0, len(std_min_galton), 12)]

temp = mu_min__marko_true
mu_min__marko_true = [mu_min__marko_true[i:i + 12] for i in range(0, len(mu_min__marko_true), 12)]
std_min_marko_true = [temp[i:i + 12] for i in range(0, len(temp), 12)]

temp = mu_min_galton_true
mu_min_galton_true = [mu_min_galton_true[i:i + 12] for i in range(0, len(mu_min_galton_true), 12)]
std_min_galton_true = [temp[i:i + 12] for i in range(0, len(temp), 12)]

mu_m_maro = [mu_m_maro[i:i + 12] for i in range(0, len(mu_m_maro), 12)]
std_m_maro = [std_m_maro[i:i + 12] for i in range(0, len(std_m_maro), 12)]

mu_m_galton = [mu_m_galton[i:i + 12] for i in range(0, len(mu_m_galton), 12)]

std_m_galton = [std_m_galton[i:i + 12] for i in range(0, len(std_m_galton), 12)]

temp = mu_m_marko_true
mu_m_marko_true = [mu_m_marko_true[i:i + 12] for i in range(0, len(mu_m_marko_true), 12)]
std_m_marko_true = [temp[i:i + 12] for i in range(0, len(temp), 12)]

temp = mu_m_galton_true
mu_m_galton_true = [mu_m_galton_true[i:i + 12] for i in range(0, len(mu_m_galton_true), 12)]
std_m_galton_true = [temp[i:i + 12] for i in range(0, len(temp), 12)]

mu_tal_maro = [mu_tal_maro[i:i + 12] for i in range(0, len(mu_tal_maro), 12)]
std_tal_maro = [std_tal_maro[i:i + 12] for i in range(0, len(std_tal_maro), 12)]

mu_tal_galton = [mu_tal_galton[i:i + 12] for i in range(0, len(mu_tal_galton), 12)]
std_tal_galton = [std_tal_galton[i:i + 12] for i in range(0, len(std_tal_galton), 12)]

temp = mu_tal_marko_true
mu_tal_marko_true = [mu_tal_marko_true[i:i + 12] for i in range(0, len(mu_tal_marko_true), 12)]
std_tal_marko_true = [temp[i:i + 12] for i in range(0, len(temp), 12)]

temp = mu_tal_galton_true
mu_tal_galton_true = [mu_tal_galton_true[i:i + 12] for i in range(0, len(mu_tal_galton_true), 12)]
std_tal_galton_true = [temp[i:i + 12] for i in range(0, len(temp), 12)]
#print(std_tal_galton_true)

temp = {}
temp[1] = mu_min_miro 
temp[2] = mu_min_galton
temp[3] =mu_min__marko_true
temp[4] =mu_min_galton_true
temp[5] =mu_m_maro
temp[6] =mu_m_galton
temp[7] =mu_m_marko_true
temp[8] =mu_m_galton_true
temp[9] =mu_tal_maro
temp[10] =mu_tal_galton
temp[11] =mu_tal_marko_true
temp[12] =mu_tal_galton_true
#Calcule le rendement accumule sur un an
for key, item in temp.items():
    mu_temp = []
    for j in item:
        rend= 1
        for p in list(j):
            rend = rend * (1 + p)
        rend = rend -1
        mu_temp.append(rend)
    item = mu_temp
    temp[key] = item

temp_std = {}
temp_std[1] = std_min_maro
temp_std[2] = std_min_galton
temp_std[3] = std_min_marko_true
temp_std[4] =std_min_galton_true
temp_std[5] = std_m_maro
temp_std[6] =std_m_galton
temp_std[7] =std_m_marko_true
temp_std[8] =std_m_galton_true
temp_std[9] =std_tal_maro
temp_std[10] =std_tal_galton
temp_std[11] =std_tal_marko_true
temp_std[12] =std_tal_galton_true
#Calcule le risque total sur un an
for key, item in temp_std.items():
    std_temp = []
    if key in [1, 2, 5, 6, 9, 10]:
        for j in item:    
            std= 0
            for p in j:
                std = std + p**2
            std = np.sqrt(std)
            std_temp.append(std)
    else:
        for j in item:
            std = np.std(j)*np.sqrt(12)
            std_temp.append(std)
    item = std_temp
    temp_std[key] = item

#print(temp_std[6], temp_std[8])


#Mets les donnees dans un dictionnaire pour creer une dataframe
Res["rendement  minimal"] = temp[1] 
Res["std  minimal"] = temp_std[1]
Res["Rendement reel minimal"] = temp[3]
Res['Std reel minimal '] = temp_std[3]
Res['Difference rendement minimal reel'] = np.array(temp[3]) - np.array(temp[1])

Res["rendement corrige  minimal "] = temp[2] 
Res['std corrige minimal'] = temp_std[2]
Res["Rendement reel galton minimal"] = temp[4]
Res["Std reel galton minimal"] = temp_std[4]
Res['Difference rendement corrige minimal reel'] = np.array(temp[4]) - np.array(temp[2])

Res["rendement  optimal"] = temp[5] 
Res["std  optimal"] = temp_std[5]
Res["Rendement reel optimal"] = temp[7]
Res['Std reel optimal'] = temp_std[7]
Res['Difference rendement optimal reel'] = np.array(temp[5]) - np.array(temp[7])

Res["rendement corrige  optimal "] = temp[6] 
Res['std corrige optimal'] = temp_std[6]
Res["Rendement reel galton optimal"] = temp[8]
Res["Std reel galton optimal"] = temp_std[8]
Res['Difference rendement corrige optimal reel'] = np.array(temp[6]) - np.array(temp[8])


Res["rendement talmud"] = temp[9] 
Res["std  talmud"] = temp_std[9]
Res["Rendement reel talmud"] = temp[11]
Res['Std reel talmud'] = temp_std[11]
Res['Difference rendement talmud reel'] = np.array(temp[9])- np.array(temp[11])
Res["rendement corrige  talmud "] = temp[10]
Res['std corrige talmud'] = temp_std[10]
Res["Rendement reel galton talmud"] = temp[12]
Res["Std reel galton talmud"] = temp_std[12]
Res['Difference rendement corrige talmud reel'] = np.array(temp[10])- np.array(temp[12])
#print(Res)
resultat = pd.DataFrame(Res)

print(resultat)

resultat.to_csv("comparaison.csv", header= True, index= False)


