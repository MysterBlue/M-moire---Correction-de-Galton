""" Programme qui calcul le portefeuille à la variance minimal, le portefeuille du marche / optimal
la frontiere efficiente, la CML avec les estimations de galton."""

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
import itertools




fichier = ld.choisir_fichiers()

dico = ld.lire_fichiers(fichier, ["Date", "Close"])

Actif = ["AMZN", "BRK-B", "CAT", "GE", "JPM", "MSFT", "NVDA", "NVO", "PFE", "XOM"]

for key, item in dico.items():
    #item.loc[:,'Date'] = pd.to_datetime(item["Date"])
    item["Close"] = item["Close"].pct_change()
    item = item.drop(index=0)
    item = item.reset_index(drop=True)
    item.loc[:,'Date'] = pd.to_datetime(item["Date"])
    item.set_index('Date', inplace = True)
    dico[key] = item

#print(dico)
#Nom des titres
Name_tot = list(dico.keys())
#print(Name_tot)


Hist = 60 #int(input("Indiquez la fenetre historique : "))
Fut =  12 #int(input("Indiquez la fenetre future : "))

df_rendement = pd.DataFrame()
#print(Name)
for df in Name_tot:
    df_rendement = pd.concat([df_rendement, dico[df]], axis = 1)

df_rendement.columns = Name_tot
#print(df_rendement)
count_tot = list(df_rendement.count(axis = 1))
k = 0
i = 0
#Avoir minimum 30 actifs qui possèdent des données au temps k
while i <= len(count_tot):
    if count_tot[i] < 30:
        k = k+1
        i = i+1
    else:
        break
nb_row = len(df_rendement.index)
df_rendement = df_rendement.reset_index(drop = True).loc[k:nb_row-7]
#print(df_rendement)
#supprime les actifs qui possèdent moins de Hist + Fut données
nb_isna = df_rendement.isna().sum()
nb_row = len(df_rendement.index)
for i in Name_tot:
    if nb_row - nb_isna[i] < Hist + Fut:
        df_rendement.drop(i, axis = 1, inplace = True)



nb_row = len(df_rendement)
print(df_rendement)
Name = list(df_rendement.columns)


#Correction de galton moyenne rendement et std
start = time.time()
g0_m = []
g1_m = []
g0_v = []
g1_v = []
#i = Hist-1
for i in range(Hist + Fut -1, nb_row-1):
    mean_hist  = []
    mean_fut = []
    var_hist  = []
    var_fut = []
    for k in Name:
        if df_rendement[k].iloc[i-Hist - Fut + 1:i -Fut+1].isna().sum() == 0 and df_rendement[k].iloc[i- Fut+ 1: i+1].isna().sum() == 0:
             mean_hist.append(df_rendement[k].iloc[i-Hist - Fut + 1:i -Fut+1].mean())
             mean_fut.append(df_rendement[k].iloc[i- Fut+ 1: i+1].mean())
             var_hist.append(df_rendement[k].iloc[i-Hist - Fut + 1:i -Fut+1].std())
             var_fut.append(df_rendement[k].iloc[i- Fut+ 1: i+1].std())
    #mean_data = pd.DataFrame({"Hist" : mean_hist, "Fut" : mean_fut})
    #var_data = pd.DataFrame({"Hist" : var_hist, "Fut" : var_fut})
    model = LinearRegression(fit_intercept=True)
    model.fit(np.array(mean_hist).reshape(-1, 1), np.array(mean_fut))
    g0_m.append(model.intercept_)
    g1_m.append(model.coef_[0])
    modelvar = LinearRegression(fit_intercept=True)
    modelvar.fit(np.array(var_hist).reshape(-1, 1), np.array(var_fut))
    g0_v.append(modelvar.intercept_)
    g1_v.append(modelvar.coef_[0])
 
end = time.time()
print(f'temps exécution {end-start}')

g0_v_mean = np.mean(g0_v)#ATTENTION C'EST STD et NON VAR DANS CALCUL
g1_v_mean = np.mean(g1_v)
g0_m_mean = np.mean(g0_m)
g1_m_mean = np.mean(g1_m)
#print(g0_m, g1_m)
print(f'les coefficients des rendements attendus sont g0_m {g0_m_mean} et g1_m {g1_m_mean}\n')
print(f'les coefficients des variances sont g0_v {g0_v_mean} et g1_v {g1_v_mean}\n')



#covariance Correction Galton
start = time.time()
g0_p = []
g1_p = []
#i = Hist-1
combinai = list(itertools.combinations(Name, 2))
for i in range(Hist + Fut -1, nb_row-1):
    cor_hist  = []
    cor_fut = []
    Matrix_hist = df_rendement.iloc[i-Hist - Fut + 1:i -Fut+1].corr(min_periods= Hist-1)
    Matrix_fut = df_rendement.iloc[i- Fut+ 1: i+1].corr(min_periods=Fut-1)
    for col1, col2 in combinai:
        if pd.notna(Matrix_hist.loc[col1, col2]) and pd.notna(Matrix_fut.loc[col1, col2]):
            cor_hist.append(Matrix_hist.loc[col1, col2])
            cor_fut.append(Matrix_fut.loc[col1, col2])
    #cor_data = pd.DataFrame({"Hist" : cor_hist, "Fut" : cor_fut})
    model = LinearRegression(fit_intercept=True)
    model.fit(np.array(cor_hist).reshape(-1, 1), np.array(cor_fut))
    g0_p.append(model.intercept_)
    g1_p.append(model.coef_[0])
    #print(i)

end = time.time()
print(f'temps exécution {end-start}')

g0_p_mean = np.mean(g0_p)
g1_p_mean = np.mean(g1_p)



print(f'les coefficients des rendements attendus sont g0_m {g0_m_mean} et g1_m {g1_m_mean}\n')

print(f'les coefficients des variances sont g0_v {g0_v_mean} et g1_v {g1_v_mean}\n')

#print(f'les coefficients des coefficient de correlations sont g0_p {g0_p} et g1_p {g1_p}\n')
print(f'les coefficients des covariances sont g0_p {g0_p_mean} et g1_p {g1_p_mean}\n')


mu = []
mu_non_corrige = []
std = []
std_non_corrige = []
for i in Actif:
    std_i = df_rendement[i].iloc[:nb_row-1].tail(Hist).std()
    mean_i = df_rendement[i].iloc[:nb_row-1].tail(Hist).mean()
    mu_non_corrige.append(mean_i)
    mu.append(g0_m_mean + g1_m_mean * mean_i)
    std_non_corrige.append(std_i)
    std.append(g0_v_mean + g1_v_mean * std_i)

mu = np.array(mu)
std = np.array(std)
mu = mu.reshape(1, len(Actif))

coef_p = df_rendement[Actif].iloc[nb_row-Hist:].corr()
coef_p_corrige = df_rendement[Actif].iloc[nb_row-Hist:].corr() #probleme lors des calculs si je fais coef_p_corrige = coef_p
#Cov = df_rendement[Actif].iloc[:nb_row-1].tail(Hist).cov()
#Cov_corrige = df_rendement[Actif].iloc[:nb_row-1].tail(Hist).cov()
#for i in range(Cov.shape[0]):
#    for j in range(Cov.shape[1]):
#        Cov_corrige.iloc[i, j] = g0_p_mean + g1_p_mean * Cov_corrige.iloc[i, j]

for i in range(coef_p_corrige.shape[0]):
    for j in range(coef_p_corrige.shape[1]):
        if i != j:
            coef_p_corrige.iloc[i, j] = g0_p_mean + g1_p_mean * coef_p_corrige.iloc[i, j]

Cov = np.dot(np.diag(std_non_corrige), np.dot(coef_p, np.diag(std)))
Cov_corrige = np.dot(np.diag(std), np.dot(coef_p_corrige, np.diag(std)))


print(f"Vecteur rendement {mu_non_corrige}")
print("Le vecteur rendement corrigé est ", mu)
print(f'La matrice de covariance est\n {Cov}')
print(f'La matrice de covariance corrigée est\n {Cov_corrige}')
print(f'Le vecteur risque est {std_non_corrige}')
print(f'Le vecteur risqué corrigé est {std}')

#Portefeuille minimal
w_min = mk.MinimalVariance(Cov_corrige, len(Actif))
mu_min = mk.esperance(mu, w_min)
std_min = mk.risque(w_min, Cov_corrige)


print(f"Portefeuille Minimal:\nPoids: {w_min[0]}\nEspérance: {mu_min}\nRisque: {std_min}")

#Portefeuille marche
w_m2 = mk.MarketPortfolio(mu, Cov_corrige, len(Actif))
mu_m2 = mk.esperance(mu, w_m2)
std_m2 = mk.risque(w_m2, Cov_corrige)
print(f"Portefeuille du Marché sans taux sans risque:\nPoids: {w_m2[0]}\nEspérance: {mu_m2}\nRisque: {std_m2}")

mu_reel = []
for k in Actif:
    mu_reel.append(df_rendement[k].tail(1))
mu_reel = np.array(mu_reel).reshape(1, len(Actif))
print(mu_reel)

print(f'Le rendement reel minimal est {mk.esperance(mu_reel, w_min)}')
print(f' rendement minimal reel marko : {mk.esperance(mu_reel, mk.MinimalVariance(Cov, len(Actif)))}')
print(f"'Rendement optimal reel {mk.esperance(mu_reel, w_m2)}")
w_tal = np.ones(len(Actif)).reshape(1, len(Actif))/len(Actif)
print(f'portefeuille talmud reel : {mk.esperance(mu_reel, w_tal)}')
print(f'portefeuille talmud estime galton : {mk.esperance(mu, w_tal)}')
print(f'portefeuille talmud estime marko : {mk.esperance(np.array(mu_non_corrige).reshape(1, len(Actif)), w_tal)}')
print(f'risque talmud estime galton : {mk.risque(w_tal, Cov_corrige)}')
print(f'risque talmud estime marko : {mk.risque(w_tal,Cov)}')