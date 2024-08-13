"""Générée par CHATGPT"""

import pandas as pd
import os
from tkinter import filedialog, Tk

# Permet de choisir les fichiers à lire via une boîte de dialogue
def choisir_fichiers():
    root = Tk()
    root.withdraw()
    fichiers = filedialog.askopenfilenames(title='Choisir les fichiers des titres', filetypes=[('CSV files', '*.csv')])
    return fichiers

# Lecture des fichiers CSV et conversion en DataFrames
def lire_fichiers(fichiers, val):
    dico = {}
    for fichier in fichiers:
        df = pd.read_csv(fichier)
        nom_fichier = os.path.basename(fichier)
        dico[nom_fichier.split('.')[0]] = df

    # Extraire les colonne de val de chaque DataFrame
    for key, item in dico.items():
        item = item[val]
        dico[key] = item

    return dico