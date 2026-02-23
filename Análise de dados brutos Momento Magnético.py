#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise de Dados - Experimento 6: Momento Magnético
Laboratório de Física 2 - UFABC

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})

# =============================================================================
# CONSTANTES E PARAMETROS
# =============================================================================
mu_0 = 4 * np.pi * 1e-7
n_H = 154
R_H = 0.200
L_braco = 0.240  # m

fator_HH = (4.0/5.0)**(3.0/2.0) * mu_0 * n_H / R_H
print(f"Fator Helmholtz (B/Ip): {fator_HH:.6e} T/A")
print(f"B(Ip=3A) = {fator_HH*3:.4e} T = {fator_HH*3*1e3:.4f} mT")
print(f"Braco de alavanca: L = {L_braco*1e3:.0f} mm")

# =============================================================================
# INCERTEZAS
# =============================================================================
sigma_F = 0.05e-3   # 0.05 mN em N (sensibilidade do dinamometro)
sigma_L = 0.001      # 1 mm (incerteza mecanica do braco)
sigma_paquimetro = 0.05e-3  # m

def erro_multimetro_DC(I_medido, faixa='20A'):
    if faixa == '20A':
        resolucao = 0.001
    elif faixa == '2A':
        resolucao = 0.0001
    else:
        resolucao = 0.001
    return 0.008 * abs(I_medido) + 10 * resolucao

def sigma_torque(F_mN):
    """Incerteza no torque em mN.m, dado F em mN."""
    termo_F = (L_braco * 0.05)**2
    termo_L = (F_mN * sigma_L)**2
    return np.sqrt(termo_F + termo_L)

print(f"sigma_tau base (L*sigma_F): {L_braco * 0.05:.4f} mN.m")

# =============================================================================
# DADOS - FORCA (leitura direta do dinamometro, em mN)
# =============================================================================

# TAREFA 1: F vs Ip
t1_n = 3; t1_d = 118.20e-3; t1_I = 3.004
t1_F = np.array([0.33, 0.44, 0.61, 0.83, 1.01, 1.19, 1.37, 1.51, 1.72, 1.92])
t1_Ip = np.array([0.372, 0.607, 0.909, 1.204, 1.501, 1.802, 2.102, 2.410, 2.716, 3.019])

# TAREFA 2: F vs n
t2_Ip = 3.016; t2_I = 3.004
t2_F = np.array([0.65, 1.11, 1.92])
t2_n = np.array([1, 2, 3])
t2_d = np.array([119.70, 119.00, 118.20]) * 1e-3

# TAREFA 3: F vs alfa
t3_n = 3; t3_d = 118.20e-3
t3_alpha_deg = np.array([0, 15, 30, 45, 60, 75, 90])
t3_F = np.array([0, 0.38, 0.75, 1.12, 1.68, 1.71, 1.91])
t3_Ip = np.array([3.034, 3.033, 3.034, 3.034, 3.034, 3.035, 3.034])
t3_I = np.array([3.019, 3.019, 3.019, 3.019, 3.019, 3.019, 3.019])
t3_sin_alpha = np.sin(np.radians(t3_alpha_deg))

# TAREFA 4: F vs d
t4_n = 1; t4_Ip = 3.032; t4_I = 3.015
t4_d = np.array([59.25, 84.20, 119.70]) * 1e-3
t4_F = np.array([0.1, 0.2, 0.58])

# TAREFA 5: F vs I
t5_n = 3; t5_d = 118.20e-3; t5_Ip = 3.035
t5_F = np.array([0.21, 0.4, 0.61, 0.74, 0.9, 1.12, 1.3, 1.48, 1.63, 1.82])
t5_I = np.array([0.309, 0.606, 0.905, 1.207, 1.505, 1.807, 2.105, 2.42, 2.709, 3.008])

# =============================================================================
# CONVERSAO F -> tau
# =============================================================================
# tau(mN.m) = F(mN) * L(m)
t1_tau = t1_F * L_braco
t2_tau = t2_F * L_braco
t3_tau = t3_F * L_braco
t4_tau = t4_F * L_braco
t5_tau = t5_F * L_braco

print(f"\nExemplo: F=1.92 mN -> tau = 1.92 * 0.240 = {1.92*0.240:.4f} mN.m")

# Incertezas no torque
sigma_t1_tau = sigma_torque(t1_F)
sigma_t2_tau = sigma_torque(t2_F)
sigma_t3_tau = sigma_torque(t3_F)
sigma_t4_tau = sigma_torque(t4_F)
sigma_t5_tau = sigma_torque(t5_F)

# Incertezas nas correntes
sigma_t1_Ip = np.array([erro_multimetro_DC(I) for I in t1_Ip])
sigma_t5_I = np.array([erro_multimetro_DC(I) for I in t5_I])

# =============================================================================
# FUNCOES DE AJUSTE
# =============================================================================
def linear(x, a, b):
    return a * x + b

def power_law(x, A, B):
    return A * np.power(x, B)

# =============================================================================
# TAREFA 1
# =============================================================================
print("\n" + "="*60)
print("TAREFA 1: Torque vs Ip")
print("="*60)

slope1, intercept1, r1, p1, se1 = linregress(t1_Ip, t1_tau)
print(f"Ajuste: tau = ({slope1:.4f} +/- {se1:.4f}) * Ip + ({intercept1:.4f})")
print(f"R2 = {r1**2:.6f}")

popt1, pcov1 = curve_fit(power_law, t1_Ip, t1_tau, p0=[0.15, 1.0])
perr1 = np.sqrt(np.diag(pcov1))
print(f"Potencia: expoente = {popt1[1]:.3f} +/- {perr1[1]:.3f}")

coef_teo1 = t1_n * t1_I * np.pi * t1_d**2 / 4 * fator_HH * 1e3
print(f"Coef. angular teorico: {coef_teo1:.4f} mN.m/A")
print(f"Coef. angular experimental: {slope1:.4f} mN.m/A")
print(f"Discrepancia: {abs(slope1-coef_teo1)/coef_teo1*100:.1f}%")

fig1, ax1 = plt.subplots()
ax1.errorbar(t1_Ip, t1_tau, yerr=sigma_t1_tau, xerr=sigma_t1_Ip,
             fmt='ko', markersize=5, capsize=3, label='Dados experimentais')
x_fit = np.linspace(0, 3.2, 100)
ax1.plot(x_fit, linear(x_fit, slope1, intercept1), 'r-',
         label=f'Ajuste linear ($R^2 = {r1**2:.4f}$)')
ax1.set_xlabel("Corrente nas bobinas de Helmholtz, $I'$ (A)")
ax1.set_ylabel("Torque, $\\tau$ (mN$\\cdot$m)")
ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 3.3); ax1.set_ylim(0, 0.55)
plt.tight_layout(); plt.savefig('tarefa1_tau_vs_Ip.png'); plt.close()

# =============================================================================
# TAREFA 2
# =============================================================================
print("\n" + "="*60)
print("TAREFA 2: Torque vs n")
print("="*60)

slope2, intercept2, r2, p2, se2 = linregress(t2_n, t2_tau)
print(f"Ajuste: tau = ({slope2:.4f} +/- {se2:.4f}) * n + ({intercept2:.4f})")
print(f"R2 = {r2**2:.6f}")

d_med2 = np.mean(t2_d)
coef_teo2 = t2_I * np.pi * d_med2**2 / 4 * fator_HH * t2_Ip * 1e3
print(f"Coef. teorico: {coef_teo2:.4f}, Exp: {slope2:.4f}, Desvio: {abs(slope2-coef_teo2)/coef_teo2*100:.1f}%")

fig2, ax2 = plt.subplots()
ax2.errorbar(t2_n, t2_tau, yerr=sigma_t2_tau, fmt='ko', markersize=6, capsize=3,
             label='Dados experimentais')
x_fit = np.linspace(0.5, 3.5, 100)
ax2.plot(x_fit, linear(x_fit, slope2, intercept2), 'r-',
         label=f'Ajuste linear ($R^2 = {r2**2:.4f}$)')
ax2.set_xlabel("Numero de espiras, $n$"); ax2.set_ylabel("Torque, $\\tau$ (mN$\\cdot$m)")
ax2.set_xticks([1, 2, 3]); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('tarefa2_tau_vs_n.png'); plt.close()

# =============================================================================
# TAREFA 3
# =============================================================================
print("\n" + "="*60)
print("TAREFA 3: Torque vs sin(alfa)")
print("="*60)

slope3, intercept3, r3, p3, se3 = linregress(t3_sin_alpha, t3_tau)
print(f"Ajuste: tau = ({slope3:.4f} +/- {se3:.4f}) * sin(a) + ({intercept3:.4f})")
print(f"R2 = {r3**2:.6f}")

B_t3 = fator_HH * np.mean(t3_Ip)
tau_max_teo = t3_n * np.mean(t3_I) * np.pi * t3_d**2 / 4 * B_t3 * 1e3
print(f"Coef. teorico (tau_max): {tau_max_teo:.4f}, Exp: {slope3:.4f}, Desvio: {abs(slope3-tau_max_teo)/tau_max_teo*100:.1f}%")

fig3, ax3 = plt.subplots()
ax3.errorbar(t3_sin_alpha, t3_tau, yerr=sigma_t3_tau, fmt='ko', markersize=5, capsize=3,
             label='Dados experimentais')
x_fit = np.linspace(-0.05, 1.05, 100)
ax3.plot(x_fit, linear(x_fit, slope3, intercept3), 'r-',
         label=f'Ajuste linear ($R^2 = {r3**2:.4f}$)')
for i, ang in enumerate(t3_alpha_deg):
    if 0 < ang < 90:
        ax3.annotate(f'{ang}'+u'\u00b0', (t3_sin_alpha[i], t3_tau[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9, color='gray')
ax3.set_xlabel("$\\sin(\\alpha)$"); ax3.set_ylabel("Torque, $\\tau$ (mN$\\cdot$m)")
ax3.legend(); ax3.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('tarefa3_tau_vs_sinalpha.png'); plt.close()

# =============================================================================
# TAREFA 4
# =============================================================================
print("\n" + "="*60)
print("TAREFA 4: Torque vs d")
print("="*60)

t4_d_cm = t4_d * 100
t4_d2_cm2 = t4_d_cm**2

slope4, intercept4, r4, p4, se4 = linregress(t4_d2_cm2, t4_tau)
print(f"Ajuste (tau vs d2): tau = ({slope4:.6f} +/- {se4:.6f}) * d2 + ({intercept4:.4f})")
print(f"R2 = {r4**2:.6f}")

popt4, pcov4 = curve_fit(power_law, t4_d * 1e3, t4_tau, p0=[0.000001, 2.0])
perr4 = np.sqrt(np.diag(pcov4))
print(f"Potencia: expoente = {popt4[1]:.3f} +/- {perr4[1]:.3f} (esperado: 2)")

fig4, ax4 = plt.subplots()
ax4.errorbar(t4_d2_cm2, t4_tau, yerr=sigma_t4_tau, fmt='ko', markersize=6, capsize=3,
             label='Dados experimentais')
x_fit = np.linspace(0, 160, 100)
ax4.plot(x_fit, linear(x_fit, slope4, intercept4), 'r-',
         label=f'Ajuste linear ($R^2 = {r4**2:.4f}$)')
ax4.set_xlabel("$d^2$ (cm$^2$)"); ax4.set_ylabel("Torque, $\\tau$ (mN$\\cdot$m)")
ax4.legend(); ax4.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('tarefa4_tau_vs_d2.png'); plt.close()

fig4b, ax4b = plt.subplots()
ax4b.loglog(t4_d * 1e3, t4_tau, 'ko', markersize=6, label='Dados experimentais')
d_fit = np.linspace(50, 130, 100)
ax4b.loglog(d_fit, power_law(d_fit, *popt4), 'r-',
            label=f'$\\tau \\propto d^{{{popt4[1]:.2f}}}$')
ax4b.set_xlabel("$d$ (mm)"); ax4b.set_ylabel("Torque, $\\tau$ (mN$\\cdot$m)")
ax4b.legend(); ax4b.grid(True, alpha=0.3, which='both')
plt.tight_layout(); plt.savefig('tarefa4_loglog.png'); plt.close()

# =============================================================================
# TAREFA 5
# =============================================================================
print("\n" + "="*60)
print("TAREFA 5: Torque vs I")
print("="*60)

slope5, intercept5, r5, p5, se5 = linregress(t5_I, t5_tau)
print(f"Ajuste: tau = ({slope5:.4f} +/- {se5:.4f}) * I + ({intercept5:.4f})")
print(f"R2 = {r5**2:.6f}")

coef_teo5 = t5_n * np.pi * t5_d**2 / 4 * fator_HH * t5_Ip * 1e3
print(f"Coef. teorico: {coef_teo5:.4f}, Exp: {slope5:.4f}, Desvio: {abs(slope5-coef_teo5)/coef_teo5*100:.1f}%")

fig5, ax5 = plt.subplots()
ax5.errorbar(t5_I, t5_tau, yerr=sigma_t5_tau, xerr=sigma_t5_I,
             fmt='ko', markersize=5, capsize=3, label='Dados experimentais')
x_fit = np.linspace(0, 3.2, 100)
ax5.plot(x_fit, linear(x_fit, slope5, intercept5), 'r-',
         label=f'Ajuste linear ($R^2 = {r5**2:.4f}$)')
ax5.set_xlabel("Corrente na espira, $I$ (A)"); ax5.set_ylabel("Torque, $\\tau$ (mN$\\cdot$m)")
ax5.legend(); ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 3.3); ax5.set_ylim(0, 0.52)
plt.tight_layout(); plt.savefig('tarefa5_tau_vs_I.png'); plt.close()

# =============================================================================
# PROPAGACAO DE INCERTEZAS - Exemplo
# =============================================================================
print("\n" + "="*60)
print("PROPAGACAO DE INCERTEZAS - Tarefa 1, ponto 5")
print("="*60)

i_ex = 4
Ip_ex = t1_Ip[i_ex]; F_ex = t1_F[i_ex]; tau_ex = t1_tau[i_ex]
sigma_Ip_ex = erro_multimetro_DC(Ip_ex)
sigma_I_ex = erro_multimetro_DC(t1_I)

print(f"F = {F_ex} mN, tau = {tau_ex:.4f} mN.m")
print(f"sigma(F) = 0.05 mN, sigma(L) = 1 mm")
print(f"sigma(tau_med) = {sigma_torque(F_ex):.4f} mN.m")

B_ex = fator_HH * Ip_ex
tau_teo = t1_n * t1_I * np.pi * t1_d**2 / 4 * B_ex * 1e3
print(f"tau_teo = {tau_teo:.4f} mN.m, tau_exp = {tau_ex:.4f} mN.m")

sr_I = sigma_I_ex / t1_I
sr_d = 2 * sigma_paquimetro / t1_d
sr_Ip = sigma_Ip_ex / Ip_ex
sr_tot = np.sqrt(sr_I**2 + sr_d**2 + sr_Ip**2)
print(f"sigma_rel: I={sr_I*100:.2f}%, d={sr_d*100:.2f}%, Ip={sr_Ip*100:.2f}%, total={sr_tot*100:.2f}%")
print(f"sigma(tau_teo) = {tau_teo*sr_tot:.4f} mN.m")

# Compatibilidade
diff = abs(tau_ex - tau_teo)
sig_comb = np.sqrt(sigma_torque(F_ex)**2 + (tau_teo*sr_tot)**2)
print(f"|tau_exp - tau_teo| = {diff:.4f} mN.m")
print(f"sigma_combinado = {sig_comb:.4f} mN.m")
print(f"Diferenca: {diff/sig_comb:.1f} sigma")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*60)
print("RESUMO")
print("="*60)
print(f"T1: slope={slope1:.4f}, teo={coef_teo1:.4f}, R2={r1**2:.4f}, exp_pot={popt1[1]:.3f}")
print(f"T2: slope={slope2:.4f}, teo={coef_teo2:.4f}, R2={r2**2:.4f}")
print(f"T3: slope={slope3:.4f}, teo={tau_max_teo:.4f}, R2={r3**2:.4f}")
print(f"T4: exp_pot={popt4[1]:.3f}+/-{perr4[1]:.3f}, R2(d2)={r4**2:.4f}")
print(f"T5: slope={slope5:.4f}, teo={coef_teo5:.4f}, R2={r5**2:.4f}")

print("\nAnalise concluida! Graficos salvos.")
