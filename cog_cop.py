# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:12:11 2022

@author: gbiomech
"""

import pandas as pd  # use Pandas to read data from a website
from numpy import loadtxt, array, matrix
import pandas as pd
# from cogve import cogve
from cogveap import cogveap
from cogveml import cogveml
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
from scipy.signal import detrend
from hyperellipsoid import hyperellipsoid
from psd import psd

URL='https://raw.githubusercontent.com/gbiomech/HasteVibratoria/main/data/ANGELA_HRH_02.txt'
name=URL[69:-4]

print("URL: ",URL)
print("Nome: ",name)

fx1,fy1,fz1,mx1,my1,mz1,fx2,fy2,fz2,mx2,my2,mz2 = np.loadtxt(URL, delimiter=',', unpack=True)

fx=fx1+fx2
fy=fy1+fy2
fz=fz1+fz2
mx=mx1+mx2
my=my1+my2
mz=mz1+mz2

CoPml=(-(my+fx)/fz)*100
CoPap=((mx-fy)/fz)*100

freq = 100
b, a = butter(4, (5/(freq/2)), btype = 'low')
CoPap = filtfilt(b, a, CoPap)
CoPml = filtfilt(b, a, CoPml)

CoPap = detrend(CoPap, axis=0, type='constant')
CoPml = detrend(CoPml, axis=0, type='constant')

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
CoGap = cogveap(CoPap, freq=100, mass=70, height=175, ax=ax1, show=True)  # guess mass, height
fig1.savefig(name+'_CoGap_CoPap.png',format='png',dpi=300,orientation='portrait',bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
CoGml = cogveml(CoPml, freq=100, mass=70, height=175, ax=ax2, show=True)  # guess mass, height
fig2.savefig(name+'_CoGml_CoPml.png',format='png',dpi=300,orientation='portrait',bbox_inches='tight')
plt.close(fig2)

deslocamento_total_CoP = sum(np.sqrt((CoPap**2 + CoPml**2)));
amplitude_CoPap=np.max(CoPap)-np.min(CoPap)
amplitude_CoPml=np.max(CoPml)-np.min(CoPml)
perimetro_CoPap = sum(abs(np.diff(CoPap)));
perimetro_CoPml = sum(abs(np.diff(CoPml)));
vel_media_CoPap = sum(abs(np.diff(CoPap)))*(freq/CoPap.shape[0]);
vel_media_CoPml = sum(abs(np.diff(CoPml)))*(freq/CoPml.shape[0]);
vel_media_total_CoP = sum(np.sqrt(np.diff(CoPap)**2+np.diff(CoPml)**2))*(freq/CoPap.shape[0]);

deslocamento_total_CoG = sum(np.sqrt((CoGap**2 + CoGml**2)));
amplitude_CoGap=np.max(CoGap)-np.min(CoGap)
amplitude_CoGml=np.max(CoGml)-np.min(CoGml)
perimetro_CoGap = sum(abs(np.diff(CoGap)));
perimetro_CoGml = sum(abs(np.diff(CoGml)));
vel_media_CoGap = sum(abs(np.diff(CoGap)))*(freq/CoGap.shape[0]);
vel_media_CoGml = sum(abs(np.diff(CoGml)))*(freq/CoGml.shape[0]);
vel_media_total_CoG = sum(np.sqrt(np.diff(CoGap)**2+np.diff(CoGml)**2))*(freq/CoGap.shape[0]);

print("CoP Deslocamnto Total: ",deslocamento_total_CoP)
print("CoPap amplitude: ",amplitude_CoPap)
print("CoPml amplitude: ",amplitude_CoPml)
print("CoPap perímetro: ",perimetro_CoPap)
print("CoPml perímetro: ",perimetro_CoPml)
print("CoPap Velocidade Média: ",vel_media_CoPap)
print("CoPml Velocidade Média: ",vel_media_CoPml)
print("CoP Velocidade Média Total: ",vel_media_total_CoP)

print("CoG Deslocamnto Total: ",deslocamento_total_CoG)
print("CoGap amplitude: ",amplitude_CoGap)
print("CoGml amplitude: ",amplitude_CoGml)
print("CoGap perímetro: ",perimetro_CoGap)
print("CoGml perímetro: ",perimetro_CoGml)
print("CoGap Velocidade Média: ",vel_media_CoGap)
print("CoGml Velocidade Média: ",vel_media_CoGml)
print("CoG Velocidade Média Total: ",vel_media_total_CoG)

fig3, ax3 = plt.subplots(1, 1)
CoParea, CoPaxes, CoPangles, CoPcenter, CoPR = hyperellipsoid(CoPml, CoPap, units='cm', show=True, ax=ax3)
fig3.savefig(name+'_CoP_area.png',format='png',dpi=300,orientation='portrait',bbox_inches='tight')
CoPAxey=CoPaxes[0]
CoPAxex=CoPaxes[1]
print('CoP Area =', CoParea)
print('CoP Eixo anteroposterior =', CoPAxey)
print('CoP Eixo mediolateral =', CoPAxex)

fig4, ax4 = plt.subplots(1, 1)
CoGarea, CoGaxes, CoGangles, CoGcenter, CoGR = hyperellipsoid(CoGml, CoGap, units='cm', show=True, ax=ax4)
fig4.savefig(name+'_CoG_area.png',format='png',dpi=300,orientation='portrait',bbox_inches='tight')
CoGAxey=CoGaxes[0]
CoGAxex=CoGaxes[1]
print('CoG Area =', CoGarea)
print('CoG Eixo anteroposterior =', CoGAxey)
print('CoG Eixo mediolateral =', CoGAxex)

fig5, ax5 = plt.subplots(1, 1)
CoP_fp_ap, CoP_mf_ap, CoP_fmax_ap, CoP_Ptot_ap, CoP_F, CoP_P_ap = psd(
    CoPap, fs=freq, scales='linear', xlim=[0, 6], units='cm', ax=ax5)
fig5.savefig(name+'_CoPap_spectral.png',format='png',dpi=600,orientation='portrait',bbox_inches='tight')

fig6, ax6 = plt.subplots(1, 1)
CoP_fp_ml, CoP_mf_ml, CoP_fmax_ml, CoP_Ptot_ml, CoP_F, CoP_P_ml = psd(
    CoPml, fs=freq, scales='linear', xlim=[0, 6], units='cm', ax=ax6)
fig6.savefig(name+'_CoPml_spectral.png',format='png',dpi=600,orientation='portrait',bbox_inches='tight')

fig7, ax7 = plt.subplots(1, 1)
CoG_fp_ap, CoG_mf_ap, CoG_fmax_ap, CoG_Ptot_ap, CoG_F, CoG_P_ap = psd(
    CoGap, fs=freq, scales='linear', xlim=[0, 6], units='cm', ax=ax7)
fig7.savefig(name+'_CoGap_spectral.png',format='png',dpi=600,orientation='portrait',bbox_inches='tight')

fig8, ax8 = plt.subplots(1, 1)
CoG_fp_ml, CoG_mf_ml, CoG_fmax_ml, CoG_Ptot_ml, CoG_F, CoG_P_ml = psd(
    CoGml, fs=freq, scales='linear', xlim=[0, 6], units='cm', ax=ax8)
fig8.savefig(name+'_CoGml_spectral.png',format='png',dpi=600,orientation='portrait',bbox_inches='tight')


rms_CoPap_CoGap=np.sqrt(np.mean((CoPap-CoGap)**2))
rms_CoPml_CoGml=np.sqrt(np.mean((CoPml-CoGml)**2))

print("RMS CoPap - CoGap: ",rms_CoPap_CoGap)
print("RMS CoPml - CoGml: ",rms_CoPml_CoGml)

cabecalho=['CoP Deslocamnto Total','CoPap amplitude','CoPml amplitude','CoPap perímetro','CoPml perímetro',
            'CoPap Velocidade Média','CoGml Velocidade Média','CoP Velocidade Média Total','CoP Area',
            'CoP Eixo anteroposterior','CoP Eixo mediolateral','CoPap Fpeak','CoPap F50%','CoPap Fmean','CoPap F95%',
            'CoPml Fpeak','CoPml F50%','CoPml Fmean','CoPml F95%',
            'CoG Deslocamnto Total','CoGap amplitude','CoGml amplitude','CoGap perímetro','CoGml perímetro',
            'CoGap Velocidade Média','CoGml Velocidade Média','CoG Velocidade Média Total','CoG Area',
            'CoG Eixo anteroposterior','CoG Eixo mediolateral','CoGap Fpeak','CoGap F50%','CoG Fmean','CoGap F95%',
            'CoGml Fpeak','CoGml F50%','CoGml Fmean','CoGml F95%','RMS CoPap - CoGap','RMS CoPml - CoGml']

data=matrix([deslocamento_total_CoP,amplitude_CoPap,amplitude_CoPml,perimetro_CoPap,perimetro_CoPml,vel_media_CoPap,
             vel_media_CoPml,vel_media_total_CoP,CoParea,CoPAxey,CoPAxex,CoP_fmax_ap,CoP_fp_ap[50],CoP_mf_ap,
             CoP_fp_ap[95],CoP_fmax_ml,CoP_fp_ml[50],CoP_mf_ml,CoP_fp_ml[95],
             deslocamento_total_CoG,amplitude_CoGap,amplitude_CoGml,perimetro_CoGap,perimetro_CoGml,vel_media_CoGap,
             vel_media_CoGml,vel_media_total_CoG,CoGarea,CoGAxey,CoGAxex,CoG_fmax_ap,CoG_fp_ap[50],CoG_mf_ap,
             CoG_fp_ap[95],CoG_fmax_ml,CoG_fp_ml[50],CoG_mf_ml,CoG_fp_ml[95],rms_CoPap_CoGap,rms_CoPml_CoGml])

dados=pd.DataFrame(data,columns=cabecalho)

dados.to_csv(name+"_results.csv", index=False, header=True, sep=',')

# dt=data.transpose()


# np.savetxt(name+"_results.csv",data,delimiter=",",header=['CoP Deslocamnto Total','CoPap amplitude','CoPml amplitude','CoPap perímetro','CoPml perímetro',
#            'CoPap Velocidade Média','CoGml Velocidade Média','CoP Velocidade Média Total','CoP Area',
#            'CoP Eixo anteroposterior','CoP Eixo mediolateral','CoPap Fpeak','CoPap F50%','CoPap Fmean','CoPap F95%',
#            'CoPml Fpeak','CoPml F50%','CoPml Fmean','CoPml F95%',
#            'CoG Deslocamnto Total','CoGap amplitude','CoGml amplitude','CoGap perímetro','CoGml perímetro',
#            'CoGap Velocidade Média','CoGml Velocidade Média','CoG Velocidade Média Total','CoG Area',
#            'CoG Eixo anteroposterior','CoG Eixo mediolateral','CoGap Fpeak','CoGap F50%','CoG Fmean','CoGap F95%',
#            'CoGml Fpeak','CoGml F50%','CoGml Fmean','CoGml F95%'],encoding="utf16")