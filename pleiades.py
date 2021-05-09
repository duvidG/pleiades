#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:40:10 2021

@author: dgiron
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.optimize import curve_fit as cf
import math
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from uncertainties.unumpy import nominal_values as nv
from uncertainties.unumpy import std_devs as st
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy as aspy

a_v = 0.13
sp_type = ['O5', 'B0', 'B5', 'A0', 'A5', 'F0', 'F5', 'G0', 'G5', 'K0', 'K5', 'M0', 'M5', 'M8']

def pinta_grafica(B, V, VB_t, V_t):
    x = B - V
    
    plt.clf()
    plt.plot(VB_t, V_t, '.', color='orange', label='Main sequence')
    plt.errorbar(nv(x), nv(V), xerr=st(x), yerr=st(V), fmt='b.', label='Pleiades')
    plt.xlabel('V-B')
    plt.ylabel('V')
    axes = plt.gca()
    axes.set_ylim([-8, 20])
    plt.gca().invert_yaxis()
    plt.legend()
   
    
    w = 0.05 *2
    h = 0.8*2
    
    axes.annotate("Out of MS", xy=(1.26, 7.37), xytext=(1.48, 4.7), arrowprops=dict(arrowstyle="->"))
    c1 = matplotlib.patches.Ellipse(xy=(1.234, 7.74), width=w, height=h, fill=None)
    axes.add_patch(c1)
    
    axes.annotate("         ", xy=(1.72, 8.30), xytext=(1.48, 4.7), arrowprops=dict(arrowstyle="->"))
    c2 = matplotlib.patches.Ellipse(xy=(1.75, 8.88), width=w, height=h, fill=None)
    axes.add_patch(c2)
    
    axes.annotate("         ", xytext=(1.45, 4.45), xy=(-0.0038, 2.8), arrowprops=dict(arrowstyle="->"))
    c2 = matplotlib.patches.Ellipse(xy=(-0.09, 2.87), width=w, height=h, fill=None)
    axes.add_patch(c2)
    
    axes.annotate("Temperature", xytext=(1.25, 18.29), xy=(0.75, 18), arrowprops=dict(arrowstyle="->"))
    axes.annotate("Brightness", xytext=(-0.439, 18), xy=(-0.265, 10), arrowprops=dict(arrowstyle="->"))
    
    h1 = -8
    h2 = h1 + 0.5
   
    a = 0
    for j, i in enumerate(VB_t):
        plt.plot([i, i], [h1-0.5, h2], 'k-')
        if j == 0:
            a = 0.11
        elif j == 1:
            a = 0.02
        else:
            a = 0.05
            
        plt.text(i-a, h1-1, sp_type[j])    

def remove_out_ms(B, V, VB_t, V_t):
    x = B - V
    x_ms = []
    y_ms = []
    
    out = False
    for j, i in enumerate(x):
        out = False
        if V[j] > 3.92:
            if i > 0.7 and V[j] < 9.00:
                out = True
            if not out:
                x_ms.append(i)
                y_ms.append(V[j])
    
    return np.array(x_ms), np.array(y_ms)

def distance_modulus(x, y, x_m, y_m):
    def aj(x1, y1, err):
        if err == 'no':
            ppot, pcov = np.polyfit(x1, y1, deg=1, cov=True)
        else:
            ppot, pcov = np.polyfit(x1, y1, deg=1, cov=True, w=1/err)
        perr = np.sqrt(np.diag(pcov))
        return ppot, perr
    
    x_aj = []
    y_aj = []
    y_m_a = []
    x_m_a = []
    
    for j, i in enumerate(x):
        if i < 1.5 and i > 0:
            x_aj.append(i)
            y_aj.append(y[j])
    for j, i in enumerate(x_m):
        if i < 1.5 and i > 0:
            x_m_a.append(i)
            y_m_a.append(y_m[j])
    
    pol, err = aj(x_aj, y_aj, 'no')


    pol_t, err_t = aj(nv(x_m_a), nv(y_m_a), st(y_m_a))
    
    mean_slope = (pol_t[0] + pol[0])/2
    
    
    def f(x, b):
        return mean_slope * x + b
    
    def ajuste(x2, y2, err):
        if err == 'no':
            ppot, pcov = cf(f, x2, y2)
        else:
            ppot, pcov = cf(f, x2, y2)
        perr = np.sqrt(np.diag(pcov))
        b = ufloat(ppot[0], perr[0])
        return b
    
    b = ajuste(nv(x_m_a), nv(y_m_a), st(y_m_a))
    b_t = ajuste(x_aj, y_aj, 'no')
    
    
    xx = np.linspace(0, 1.4, 1000)
       
    plt.plot(x, y, '.', label='Main sequence', color='orange')
    plt.errorbar(nv(x_m), nv(y_m), xerr=st(x_m), yerr=st(y_m), fmt= 'b.', label='Pleiades')
    
    plt.plot(xx, f(xx, nv(b)), 'b-')
    plt.plot(xx, f(xx, nv(b_t)), '-', color='orange')
    
    plt.annotate(text='', xy=(0.02,1.0), xytext=(0.02,6.90), arrowprops=dict(arrowstyle='<->'))
    plt.annotate('m - M', xy =(0.05, 5.))
    
    h1 = -8
    h2 = h1 + 0.5
   
    a = 0
    for j, i in enumerate(x):
        plt.plot([i, i], [h1-0.5, h2], 'k-')
        if j == 0:
            a = 0.11
        elif j == 1:
            a = 0.02
        else:
            a = 0.05
            
        plt.text(i-a, h1-1, sp_type[j])   
    
    plt.xlabel('V-B')
    plt.ylabel('V')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([-8, 20])
    plt.gca().invert_yaxis()
    
    return b - b_t

def tabla_latex(tabla, ind, col, r):
    """
    Prints an array in latex format
    Args:
        tabla: array to print in latex format (formed as an array of the variables (arrays) that want to be displayed)
        ind: list with index names
        col: list with columns names
        r: number of decimals to be rounded
        
    Returns:
        ptabla: table with the data in a pandas.Dataframe format
    """
    tabla = tabla.T
    # tabla = tabla.round(r)
    ptabla = pd.DataFrame(tabla, index=ind, columns=col)
    print("Tabla en latex:\n")
    print(ptabla.to_latex(index=False, escape=False))
    return ptabla

def main():
    datos = np.genfromtxt('OBSERVATIONS.CSV', delimiter=',')
    datos = datos[1:]
    
    B = datos[:, 9]
    V = datos[:, 10]
    
    datos_t = np.genfromtxt('teoricos.txt', delimiter=',')
    datos_t = datos_t[1:]
    V_t = datos_t[:, 0]
    VB_t = datos_t[:, 1]
    
    datos1 = np.genfromtxt('datos.txt', delimiter=',')
    datos1 = datos1[1:]
    
    b_b = datos1[:, 0]
    snb = datos1[:, 1]
    v_b = datos1[:, 2]
    snv = datos1[:, 3]
    
    delta_b = B/snb
    delta_v = V/snv
    
    B_err = unumpy.uarray(B, delta_b) 
    V_err = unumpy.uarray(V, delta_v)
    
    # pinta_grafica(B_err, V_err, VB_t, V_t)
    # plt.savefig('out.png', dpi=720)
    
    x_ms, y_ms = remove_out_ms(B_err, V_err, VB_t, V_t)
    
    plt.clf()
    dm = distance_modulus(VB_t, V_t, x_ms, y_ms)
    print(dm)
    plt.plot([0, 0], [20, -8], 'k-')
    # plt.savefig('dist_mod.png', dpi=720)
    
    d = 10**((dm + 5)/5)
    print('Wo extinction', d)
    
    d2 = 10**((dm + 5-a_v)/5)
    print('W extinction', d2)
    
    
    ra_exp = datos[:, 2] + datos[:, 3] / 60 + datos[:, 4]/3600
    star_id = np.array(['N2230-01442','N2230-00478','N2230-00526','N2230-00306',
                        'N2230-01585','N2230-00156','N2230-01554','N2230-00319',
                        'N2230-00990','N2230-02089','N2230-01908','N2230-01621',
                        'N2230-01627','N2230-02081','N2230-01820','N2230-02202',
                        'N2230-02192','N2230-02091','N2230-02449','N2230-02207',
                        'N2230-01601','N2230-01175','N2230-01127','N2230-00985']) 
    dec_exp = datos[:, 5] + datos[:, 6] / 60 + datos[:, 7]/3600
    c = SkyCoord(ra= ra_exp*u.degree, dec=dec_exp*u.degree)
      
    
    # tabla = np.array([star_id, c.ra.to_string(u.hour), c.dec.to_string(u.hour), B_err, V_err, (B_err-V_err)])
    # tabla_latex(tabla, ['' for i in B], ['ID', '$\alpha$', '$\delta$', '$B$', '$V$', '$B-V$'], 3)

    # tabla = np.array([star_id, c.ra.to_string(u.hour), c.dec.to_string(u.hour), b_b, v_b, snb, snv])
    # tabla_latex(tabla, ['' for i in B], ['ID', '$\alpha$', '$\delta$', '$B_b$', '$V_b$', '$SNB$', '$SNV$'], 3)



main()