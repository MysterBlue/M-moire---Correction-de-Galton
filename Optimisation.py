""" Programme qui calcul le portefeuille à la variance minimal, le portefeuille du marche / optimal
la frontiere efficiente, la CML avec les estimations classiques.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import Markowitz as mk
import LoadData as ld
import statistics as st

fichier = ld.choisir_fichiers()

dico = ld.lire_fichiers(fichier, "Close")

Name_tot = list(dico.keys())
print(Name_tot)
rf_name = input("indiquez nom taux sans risque : ")

dico[rf_name] = dico[rf_name].apply(lambda x : (1+x/100)**(1/12) - 1)
nb_row = len(dico[rf_name].index)
rf = dico[rf_name].iloc[nb_row  -66:nb_row-6].mean()
Name = [i for i in Name_tot if i != rf_name]

#print(dico)
for key, item in dico.items():
    item = item.pct_change()
    item = item.drop(index=0)
    item = item.reset_index(drop=True)
    dico[key] = item

#Calcul rendement
df_rendement = pd.DataFrame()
#rf = dico[rf_name].mean()

for df in Name:
    df_rendement = pd.concat([df_rendement, dico[df]], axis = 1)

df_rendement.columns = Name
nb_row = len(df_rendement.index)
df_rendement = df_rendement.iloc[:nb_row -6].tail(61)
df_rendement.to_csv("Donnes.csv", header= True, index= False)
df_rendement.reset_index(drop=True)
print(df_rendement)
nb_row = len(df_rendement.index)

mu = df_rendement.iloc[0:60].mean().to_numpy()
mu = mu.reshape(1, len(mu))
mu_reel = []
for k in Name:
    mu_reel.append(df_rendement[k][60])
mu_reel = np.array(mu_reel).reshape(1, len(mu_reel))

Cov = df_rendement.iloc[0:60].cov()
print(f'Rendement reel {mu_reel}')
print("L'esperance est ", mu, "\n", "La matrice de covariance est ", Cov)

#Portefeuille minimal
w_min = mk.MinimalVariance(Cov, len(Name))
mu_min = mk.esperance(mu, w_min)
std_min = mk.risque(w_min, Cov)

print("le taux sans risque est de " , rf)
print(Name)
print(f"Portefeuille Minimal estim:\nPoids: {w_min[0]}\nEspérance: {mu_min}\nRisque: {std_min}")
print(f'Rendement réel du portefeuille minimal : {mk.esperance(mu_reel,w_min)}')

#Portefeuille marche
w_m = mk.MarketPortfolio(mu, Cov, len(Name), rf)
w_m2 = mk.MarketPortfolio(mu, Cov, len(Name))
mu_m = mk.esperance(mu, w_m)
std_m = mk.risque(w_m, Cov)
mu_m2 = mk.esperance(mu, w_m2)
std_m2 = mk.risque(w_m2, Cov)
print(f"Portefeuille du Marché:\nPoids: {w_m[0]}\nEspérance: {mu_m}\nRisque: {std_m}")
print(f"rendement portefeuille du marche reel : {mk.esperance(mu_reel, w_m)}")

print(f"Portefeuille du Marché sans taux sans risque:\nPoids: {w_m2[0]}\nEspérance: {mu_m2}\nRisque: {std_m2}")
print(f"rendement reel optimal : {mk.esperance(mu_reel, w_m2)}")




#frontiere efficiente
rp = np.linspace(mu_min, mu_m +0.01, 1000)
w_efficient = []
for i in rp:
    temp = mk.efficiente(mu, Cov, i, len(Name))
    w_efficient = w_efficient + [temp]
esp_efficient = []
risq_efficient = []
for i in w_efficient:
    temp = [mk.esperance(mu, i), mk.risque(i, Cov)]
    esp_efficient.append(temp[0])
    risq_efficient.append(temp[1])

#Creation de portefeuille aléatoire
combinaison = []
for j in range(1500):
        combi = np.random.randint(low = -1.9, high = 1.9, size= (1, len(Name)))
        total = np.sum(combi)
        if total != 0 :
            scale_combi = combi / total
            combinaison.append(scale_combi)
        else:
            continue
#Verifie que la somme des combinaison vaille 1.
for combi in combinaison:
    assert round(sum(combi[0]), 2) == 1

tab_esp = []
tab_risq = []
for i in combinaison:
    tab_esp.append(mk.esperance(mu, i))
    tab_risq.append(mk.risque(i, Cov))




#CML
std_CML = np.linspace(0, std_m + 0.01, 2)
mu_CML = []
for i in std_CML:
    mu_CML = mu_CML + [mk.CML(rf, mu_m, std_m, i)]



#plot
#Plot frontiere efficiente
plt.plot(risq_efficient, esp_efficient, c = 'red', label = "Frontiere efficiente")

#plot portefeuille minimal
plt.scatter(std_min, mu_min, c = "black", label = "Portefeuille minimale")

#Portefeuille du marché
plt.scatter(std_m, mu_m, c = "green", label = "Portefeuille du marche")

#portefeuiller random
plt.scatter(tab_risq, tab_esp, color = "yellow", label ="Portefeuille")

#CML
plt.plot(std_CML, mu_CML, label = "Droite CML")
plt.xlabel("Risque")
plt.ylabel("Rendement attendu")
plt.legend()
plt.show()