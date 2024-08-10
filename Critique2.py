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
df_rendement = df_rendement.reset_index(drop = True)
df_rendement.columns = Name_tot
nb_row = len(df_rendement.index)


mean_data = {}
var_data = {}
#i = Hist -1
for i in range(Hist-1, nb_row- Fut -1):
    for k in Name_tot:
        if df_rendement[k].iloc[i-Hist+1:i+1].isna().sum() == 0 and df_rendement[k].iloc[i +1:i + Fut+1].isna().sum() == 0:
            mean_data[df_rendement[k].iloc[i-Hist+1:i+1].mean()] = df_rendement[k].iloc[i+1:i+Fut+1].mean()
            var_data[df_rendement[k].iloc[i-Hist+1:i+1].var()] = df_rendement[k].iloc[i+1:i+Fut+1].var()
print(len(list(mean_data.keys())))


decile_rend_hist = np.array_split(np.sort(list(mean_data.keys())), 10)
decile_rend_fut = []
for i in decile_rend_hist:
    temp = []
    for j in i:
        temp = temp + [mean_data[j]]
    decile_rend_fut = decile_rend_fut + [np.mean(temp)]
decile_rend_hist = [np.mean(decile) for decile in decile_rend_hist]


decile_var_hist = np.array_split(np.sort(list(var_data.keys())), 10)
decile_var_fut = []
for i in decile_var_hist:
    temp = []
    for j in i:
        temp = temp + [var_data[j]]
    decile_var_fut = decile_var_fut + [np.mean(temp)]
decile_var_hist = [np.mean(decile) for decile in decile_var_hist]


combinai = list(itertools.combinations(Name_tot, 2))
cov_data = {}
for i in range(Hist-1, nb_row- Fut -1):
    Matrix_hist = df_rendement.iloc[i-Hist+1:i+1].cov(min_periods= Hist)
    Matrix_fut = df_rendement.iloc[i+1:i+Fut+1].cov(min_periods=Fut)
    for col1, col2 in combinai:
        if pd.notna(Matrix_hist.loc[col1, col2]) and pd.notna(Matrix_fut.loc[col1, col2]):
            cov_data[Matrix_hist.loc[col1, col2]] = Matrix_fut.loc[col1, col2]
decile_cov_hist = np.array_split(np.sort(list(cov_data.keys())), 10)
decile_cov_fut = []
for i in decile_cov_hist:
    temp = []
    for j in i:
        temp.append(cov_data[j])
    decile_cov_fut.append(np.mean(temp))
decile_cov_hist = [np.mean(decile) for decile in decile_cov_hist]


corr_data = {}
for i in range(Hist-1, nb_row- Fut -1):
    Matrix_hist = df_rendement.iloc[i-Hist+1:i+1].corr(min_periods= Hist)
    Matrix_fut = df_rendement.iloc[i+1:i+Fut+1].corr(min_periods=Fut)
    for col1, col2 in combinai:
        if pd.notna(Matrix_hist.loc[col1, col2]) and pd.notna(Matrix_fut.loc[col1, col2]):
            corr_data[Matrix_hist.loc[col1, col2]] = Matrix_fut.loc[col1, col2]
decile_cor_hist = np.array_split(np.sort(list(corr_data.keys())), 10)
decile_cor_fut = []
for i in decile_cor_hist:
    temp = []
    for j in i:
        temp.append(corr_data[j])
    decile_cor_fut.append(np.mean(temp))
decile_cor_hist = [np.mean(decile) for decile in decile_cor_hist]





fig, axs= plt.subplots(2, 2)
axs[1, 1].scatter(decile_rend_hist, decile_rend_fut)
x = np.linspace(np.min(decile_rend_hist)-0.01, np.max(decile_rend_hist)+0.01, 1000)
y = x
axs[1, 1].plot(x, y)
axs[1, 1].set_xlabel("moyenne historique")
axs[1, 1].set_ylabel("moyenne reelle")

axs[1, 0].scatter(decile_var_hist, decile_var_fut)
x = np.linspace(np.min(decile_var_hist)-0.01, np.max(decile_var_hist)+0.01, 1000)
y = x
axs[1, 0].plot(x, y)
axs[1, 0].set_xlabel("variance historique")
axs[1, 0].set_ylabel("variance reelle")

axs[0, 0].scatter(decile_cov_hist, decile_cov_fut)
x = np.linspace(np.min(decile_cov_hist)-0.01, np.max(decile_cov_hist)+0.01, 1000)
y = x
axs[0, 0].plot(x, y)
axs[0, 0].set_xlabel("covariance historique")
axs[0, 0].set_ylabel("covariance reelle")

axs[0, 1].scatter(decile_cor_hist, decile_cor_fut)
x = np.linspace(np.min(decile_cor_hist)-0.01, np.max(decile_cor_hist)+0.01, 1000)
y = x
axs[0, 1].plot(x, y)
axs[0, 1].set_xlabel("correlation historique")
axs[0, 1].set_ylabel("correlation reelle")

plt.tight_layout()
plt.show()