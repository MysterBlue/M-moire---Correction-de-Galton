"""Librarie pour calculer les variables d'entrée des optimisations
        de portefeuille"""

import numpy as np

"""Calcul le rendement d'un portefeuille
entree : mu : vecteur rendement de shape (1, n)
         w  : vecteur poids de shape (1, n)
         alg : boolean
sortie : rendement (float) """
def esperance(mu, w, alg = False):
    if alg == False:
        return np.dot(mu, w.T)[0][0]
    else:
        som = 0
        n = len(mu[0])
        j = 0
        while j <n:
            som = som + mu[0][j] * w[0][j]
            j = j+1
        return som

"""Calcul le risque du portefeuille
entrée : w : vecteur poids de shape (1, n)
         C : Matrice de covariance shape (n, n)
         alg : boolean
sortie : float """
def risque(w, C, alg = False):
    if alg == False:
        return np.sqrt(np.dot(w, np.dot(C, w.T))[0][0])
    else:
        som = 0
        n = len(w[0])
        for i in range(n):
            for j in range(n):
                som = som + w[0, i]*w[0, j] * C[i, j]
        return np.sqrt(som)

"""Calcul les poids du portefeuille au risque minimal
entree : n : nombre d'actif (int)
         C : matrice de Covariance de shape(n,n)
sortie : vecteur poids de shape (1, n)"""
def MinimalVariance(C, n):
    u = np.ones(n).reshape(1, n)
    Cinv = np.linalg.inv(C)
    N = np.dot(u, Cinv)
    D = np.dot(u, np.dot(Cinv, u.T))[0][0]
    return N / D

"""Calcul le poids du portefeuille du Marche
entre : rf : taux sans risque float
        mu : rendement de vecteur de shape (1, n)
        C  : matrice de covariance de shape (n x n)
        n  : nombre d'actif int
sortie : vecteur poids de shape (1, n)"""
def MarketPortfolio(mu, C, n, rf = 0):
    u = np.ones(n).reshape(1, n)
    Cinv = np.linalg.inv(C)  
    urfu = mu - rf * u
    N = np.dot(urfu, Cinv)
    D = np.dot(urfu, np.dot(Cinv, u.T))[0][0]
    return N / D

"""Calcul portefeuille efficient
entree : mu : vecteur rendement de shape (1, n)
         C : Matrice de covariance (n, n)
         mu_p : rendement souhaité float
         n : nombre d'actif int
sortie : vecteur poids de shape (1, n)"""
def efficiente(mu, C, mu_p, n):
    u = np.ones(n).reshape(1, n)
    Cinv = np.linalg.inv(C)
    a1 = np.dot(u, np.dot(Cinv,mu.T))[0][0]
    a2 = np.dot(mu, np.dot(Cinv,mu.T))[0][0]
    a3 = np.dot(u, np.dot(Cinv,u.T))[0][0]
    t1 = a2 / (a2 * a3 - a1**2) * np.dot(u, Cinv)
    t2 = a1 / (a2 * a3 - a1**2) * np.dot(mu, Cinv)
    t3 = a1 / (a2 * a3 - a1**2) * np.dot(u, Cinv)
    t4 = a3 / (a2 * a3 - a1**2) * np.dot(mu, Cinv)
    return t1 - t2 + mu_p * (t4 - t3)
    
"""Droite CML
entree : rf : taux sans risque 
         Erm : rendement (attendu) du portefeuille du marché
         sigma_m : risque du portefeuille du marché
         sigma_p : risque du portefeuille P
sortie : rendement attendu du portefeuille P
"""
def CML(rf, Erm, sigma_m, sigma_p):
    return rf + (Erm - rf)/sigma_m * sigma_p


""" Coefficient Beta
entree : sigma_im : covariance entre i et le portefeuille du marche
         sigma_m  : risque du portefeuille du marche
sortie : coefficient beta"""
def coefb(sigma_im, sigma_m):
    return sigma_im / (sigma_m^2)


"""Droite CAPM
entree : rf : taux sans risque
         beta : coefficient beta
         Er_m : rendement attendu portefeuille du marche
sortie : Rendement attendu portefeuille P """
def CAPM(rf, beta, Er_m):
    return rf + beta * (Er_m - rf)