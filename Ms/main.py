# -*- coding: utf-8 -*-
# ЛР-1: Стратегическое и тактическое планирование (Вейбулл, дисперсия)
# Автор: (впиши ФИО)
# Вариант 24: b∈(1,3), c∈(5,7); d=0.05; alpha=0.06

import numpy as np
from scipy.stats import norm
from scipy.special import gamma
import matplotlib.pyplot as plt

# -----------------------------
# Модель системы: Weibull(b, c)
# Генерация по обратному преобразованию: X = b * (-ln U)^(1/c)
# -----------------------------
rng = np.random.default_rng()

def system_eqv_weibull(b, c):
    U = rng.uniform(0.0, 1.0)
    return b * (-np.log(U))**(1.0 / c)

# -----------------------------
# СТРАТЕГИЧЕСКОЕ ПЛАНИРОВАНИЕ
# Два фактора: x1=b ∈ (1,3), x2=c ∈ (5,7)
# Дробный двухуровневый план 2^(2-1) = 4 опыта.
# Кодированные уровни: -1 ↔ min, +1 ↔ max
# Столбец взаимодействия = произведение столбцов факторов.
# -----------------------------
nf = 2
minf = np.array([1.0, 5.0])  # [b_min, c_min]
maxf = np.array([3.0, 7.0])  # [b_max, c_max]

# Полуреплика ПФЭ 2^2: возьмём стандартную последовательность
x1 = np.array([-1, +1, -1, +1])  # кодированный b
x2 = np.array([-1, -1, +1, +1])  # кодированный c
x12 = x1 * x2                     # взаимодействие

fracplan = np.column_stack([x1, x2, x12])  # (N x 3) без фиктивного фактора
N = fracplan.shape[0]  # 4 опыта

# Матрица планирования с фиктивным фактором x0 ≡ 1
X = np.column_stack([np.ones(N), fracplan]).T  # форма (p x N), p=4

print("Матрица планирования (кодированные):")
print("a=x1  b=x2  ab=x1*x2\n", fracplan)

# Перевод в физические значения (b, c) для каждого опыта
fraceks = np.zeros((N, nf))
for i in range(nf):  # 0: b, 1: c
    # линейная шкала: real = min + (code+1)/2 * (max-min)
    fraceks[:, i] = minf[i] + (fracplan[:, i] + 1.0) * 0.5 * (maxf[i] - minf[i])

print("\nФизические значения факторов [b, c] для опытов:")
print(fraceks)

# -----------------------------
# ТАКТИЧЕСКОЕ ПЛАНИРОВАНИЕ
# Показатель эффективности: дисперсия D
# Требуемая точность: доверительный интервал d=0.05 при α=0.06
# Из методички (задача определения дисперсии):
#   d = t_kp * sqrt(2/(n-1))  =>  n = 1 + 2 * (t_kp / d)^2
# где t_kp — квантиль стандартной нормали для (1 - α/2)
# -----------------------------
alpha = 0.06
d = 0.05
t_kp = norm.ppf(1 - alpha/2.0)   # нормальная аппроксимация t при n>=30
NE = int(np.ceil(1 + 2.0 * (t_kp / d)**2))

print(f"\nТребуемое число испытаний на одну точку плана: NE = {NE} (d={d}, alpha={alpha}, t={t_kp:.4f})")

# -----------------------------
# ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТОВ
# Для каждого опыта выполнить NE прогонов и оценить выборочную дисперсию
# -----------------------------
Y = np.zeros(N)  # сюда кладём оценку дисперсии для каждой точки плана

for j in range(N):
    b_val, c_val = fraceks[j, 0], fraceks[j, 1]
    u = np.array([system_eqv_weibull(b_val, c_val) for _ in range(NE)])
    # Выборочная дисперсия со смещением Бесселя (ddof=1), как в методичке
    D_hat = np.var(u, ddof=1)
    Y[j] = D_hat

print("\nОценки дисперсии по опытам (Y):")
print(Y)

# -----------------------------
# РЕГРЕССИОННЫЙ АНАЛИЗ
# Модель: Y ≈ b0 + b1*x1 + b2*x2 + b3*(x1*x2)
# Оценки МНК: b = (X X^T)^{-1} X Y
# -----------------------------
C = X @ X.T
b_hat = np.linalg.solve(C, X @ Y)

print("\nКоэффициенты линейной регрессии (в кодированных переменных):")
print("b0, b1(b), b2(c), b3(b×c) = ", b_hat)

# -----------------------------
# ПОСТРОЕНИЕ ПОВЕРХНОСТЕЙ
# Экспериментальная (по регрессии) и теоретическая (аналитически)
# Теоретическая дисперсия Weibull:
#   D = b^2 * ( Γ((c+2)/c) - Γ^2((c+1)/c) )
# -----------------------------
# сетки по физическим шкалам
b_vals = np.linspace(minf[0], maxf[0], 81)
c_vals = np.linspace(minf[1], maxf[1], 81)
B_grid, C_grid = np.meshgrid(b_vals, c_vals)

# перевод в кодированные для регрессионной модели
x1_grid = 2*(B_grid - minf[0])/(maxf[0]-minf[0]) - 1
x2_grid = 2*(C_grid - minf[1])/(maxf[1]-minf[1]) - 1

Y_exp = (b_hat[0]
         + b_hat[1]*x1_grid
         + b_hat[2]*x2_grid
         + b_hat[3]*x1_grid*x2_grid)

# Теоретическая дисперсия
G1 = gamma((C_grid + 1.0)/C_grid)
G2 = gamma((C_grid + 2.0)/C_grid)
Y_theor = (B_grid**2) * (G2 - G1**2)

# -----------------------------
# ВИЗУАЛИЗАЦИЯ
# -----------------------------
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(B_grid, C_grid, Y_exp, edgecolor='k', alpha=0.6)
ax1.set_xlabel('b')
ax1.set_ylabel('c')
ax1.set_zlabel('D̂ (регрессия)')
ax1.set_title('Экспериментальная поверхность дисперсии')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(B_grid, C_grid, Y_theor, edgecolor='k', alpha=0.6)
ax2.set_xlabel('b')
ax2.set_ylabel('c')
ax2.set_zlabel('D (теория)')
ax2.set_title('Теоретическая поверхность дисперсии (Weibull)')

plt.tight_layout()
plt.show()
