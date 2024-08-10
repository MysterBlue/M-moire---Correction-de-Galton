""" GENERER PAR CHATGPT"""

import yfinance as yf
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

def telecharger_donnees_financieres(actifs, duree, frequence, dossier_sortie):
    """
    Télécharge les données financières pour une liste d'actifs avec la durée et la fréquence spécifiées et les enregistre dans des fichiers CSV.
    
    :param actifs: Liste des symboles d'actifs séparés par des virgules
    :param duree: Durée des données à télécharger (par exemple, '1y' pour un an, '1mo' pour un mois)
    :param frequence: Fréquence des données (par exemple, '1d' pour quotidien, '1wk' pour hebdomadaire)
    :param dossier_sortie: Dossier de sortie pour enregistrer les fichiers CSV
    """
    actifs_list = [actif.strip() for actif in actifs.split(',')]
    for actif in actifs_list:
        try:
            data = yf.download(tickers=actif, period=duree, interval=frequence)
            if data is not None and not data.empty:
                fichier_sortie = f"{dossier_sortie}/{actif}.csv"
                data.to_csv(fichier_sortie)
                #messagebox.showinfo("Succès", f"Les données pour {actif} ont été téléchargées et enregistrées dans {fichier_sortie}")
            else:
                messagebox.showerror("Erreur", f"Aucune donnée disponible pour {actif}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du téléchargement des données pour {actif} : {e}")
    print("it s done")

def selectionner_dossier():
    dossier = filedialog.askdirectory()
    if dossier:
        dossier_sortie.set(dossier)

def lancer_telechargement():
    actifs = entree_actifs.get()
    duree = duree_selection.get()
    frequence = frequence_selection.get()
    dossier = dossier_sortie.get()
    
    if not actifs or not duree or not frequence or not dossier:
        messagebox.showwarning("Attention", "Tous les champs doivent être remplis")
        return
    
    telecharger_donnees_financieres(actifs, duree, frequence, dossier)

# Création de l'interface graphique
root = tk.Tk()
root.title("Téléchargement de données financières")

# Symboles des actifs
ttk.Label(root, text="Symboles des actifs (séparés par des virgules) :").grid(row=0, column=0, padx=10, pady=10)
entree_actifs = ttk.Entry(root)
entree_actifs.grid(row=0, column=1, padx=10, pady=10)

# Durée
ttk.Label(root, text="Durée :").grid(row=1, column=0, padx=10, pady=10)
duree_selection = ttk.Combobox(root, values=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
duree_selection.grid(row=1, column=1, padx=10, pady=10)
duree_selection.current(5)  # Par défaut, sélectionne "1y"

# Fréquence
ttk.Label(root, text="Fréquence :").grid(row=2, column=0, padx=10, pady=10)
frequence_selection = ttk.Combobox(root, values=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])
frequence_selection.grid(row=2, column=1, padx=10, pady=10)
frequence_selection.current(8)  # Par défaut, sélectionne "1d"

# Dossier de sortie
ttk.Label(root, text="Dossier de sortie :").grid(row=3, column=0, padx=10, pady=10)
dossier_sortie = tk.StringVar()
ttk.Entry(root, textvariable=dossier_sortie, state='readonly').grid(row=3, column=1, padx=10, pady=10)
ttk.Button(root, text="Sélectionner", command=selectionner_dossier).grid(row=3, column=2, padx=10, pady=10)

# Bouton pour lancer le téléchargement
ttk.Button(root, text="Télécharger", command=lancer_telechargement).grid(row=4, columnspan=3, padx=10, pady=10)

root.mainloop()
