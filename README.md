# Mémoire---Correction-de-Galton
Programmes créés pour la réalisation de mon mémoire : Optimisation de portefeuille - La théorie de Markowitz et la correction de Galton

DataDownload : permet de télécharger les données des actifs via yfinance et par une interface

LoadData : permet de charger les fichiers dans un programme

Markowitz : librairie qui permet de calculer le rendement, le risque d'un portefeuille mais également les vecteurs poids des portefeuilles clés

Optimisation : programme qui calcule les portefeuilles clés de la théorie de markowitz en utilsant la méthode classique des estimations. Il crée également un graphe avec les résultats.

Galton : calcule le portefeuille de Talmud, variance minimale et optimal de markowitz avec la correction de Galton

critique2.py = crée les graphes de la critique basée sur les erreurs d'estimations

comparaison_rebal_V2 2 fait la comparaison des portefeuilles de talmud, minimal, optimal dont une partie est estimée via la méthode classique, l'autre via la correction de Galton. Les portefeuilles sont rebalancés tous les mois. Nous comparons le rendement et le risque accumulé sur un an.
