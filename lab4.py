# -*- coding: utf-8 -*-
# Лабораторна робота №4. Кореляційний аналіз
# Усі обчислення виконані "вручну": без sum, len, sqrt, abs, range, round, min, max,
# без модулів math / statistics / numpy / scipy. Використано лише арифметичні
# оператори, цикли while, списки та print. Бібліотека matplotlib застосована
# ВИКЛЮЧНО для малювання графіків (блок 8).

# ===== БЛОК 1. ВХІДНІ ДАНІ =====
l = 4
m = 5
n = 20
# Центри інтервалів основних фондів X (млн. грн.)
X_values = [12.5, 17.5, 22.5, 27.5]
# Центри інтервалів річного обсягу виробництва Y (млн. грн.)
Y_values = [20.5, 21.5, 22.5, 23.5, 24.5]
# Кореляційна таблиця частот (таблиця 4.2): рядок - X, стовпець - Y
freq = [
    [1, 0, 0, 0, 0],
    [0, 2, 1, 0, 0],
    [0, 0, 2, 3, 0],
    [0, 0, 0, 3, 8],
]
alpha = 0.05
PI = 3.141592653589793

# ===== БЛОК 2. ВЛАСНІ ДОПОМІЖНІ ФУНКЦІЇ =====
def count(a):
    c = 0
    i = 0
    while True:
        try:
            a[i]
        except IndexError:
            break
        c = c + 1
        i = i + 1
    return c

def total(a):
    s = 0.0
    i = 0
    k = count(a)
    while i < k:
        s = s + a[i]
        i = i + 1
    return s

def my_abs(v):
    if v < 0:
        return -v
    return v

def my_sqrt(v):
    # метод Ньютона (ітерації Герона)
    if v <= 0:
        return 0.0
    x = v
    if x < 1:
        x = 1.0
    i = 0
    while i < 100:
        x = 0.5 * (x + v / x)
        i = i + 1
    return x

def mean(a):
    return total(a) / count(a)

def variance(a):
    # дисперсія вибірки: D = (Σ (x - x̄)²) / n
    mu = mean(a)
    s = 0.0
    i = 0
    k = count(a)
    while i < k:
        s = s + (a[i] - mu) * (a[i] - mu)
        i = i + 1
    return s / k

def variation_level(v):
    # оцінка варіації за таблицею 4.1
    if v <= 5:
        return "варіація слабка"
    if v <= 10:
        return "варіація помірна"
    if v <= 20:
        return "варіація значна"
    if v <= 50:
        return "варіація велика"
    return "варіація дуже велика"

def strength_level(r):
    # оцінка щільності зв'язку за шкалою з теорії
    a = my_abs(r)
    if a < 0.3:
        return "слабкий"
    if a < 0.5:
        return "помірний"
    if a < 0.7:
        return "помітний"
    if a < 0.9:
        return "досить тісний"
    if a < 0.99:
        return "надто тісний"
    return "функціональний (майже)"

def gamma_half(two_z):
    # Γ(two_z / 2) для натурального two_z: Γ(z+1) = z·Γ(z), Γ(1) = 1, Γ(1/2) = √π
    if two_z % 2 == 0:
        g = 1.0
        z = 1.0
    else:
        g = my_sqrt(PI)
        z = 0.5
    target = two_z / 2
    while z < target:
        g = g * z
        z = z + 1
    return g

def t_density(x, k):
    # щільність розподілу Стьюдента з k ступенями свободи
    c = gamma_half(k + 1) / (my_sqrt(k * PI) * gamma_half(k))
    return c * (1 + x * x / k) ** (-(k + 1) / 2)

def integral_0_to(t, k, steps):
    # інтеграл від 0 до t щільності Стьюдента (формула Сімпсона), steps - парне
    h = t / steps
    s = t_density(0.0, k) + t_density(t, k)
    i = 1
    while i < steps:
        if i % 2 == 1:
            s = s + 4 * t_density(i * h, k)
        else:
            s = s + 2 * t_density(i * h, k)
        i = i + 1
    return s * h / 3

def p_two_sided(t, k):
    # P(|T| > t) = 1 - 2·∫₀ᵗ f(x)dx
    return 1 - 2 * integral_0_to(my_abs(t), k, 20000)

def t_critical(a, k):
    # двостороннє критичне значення t_кр(α; k) методом бісекції
    lo = 0.0
    hi = 50.0
    i = 0
    while i < 80:
        mid = (lo + hi) / 2
        if p_two_sided(mid, k) > a:
            lo = mid
        else:
            hi = mid
        i = i + 1
    return (lo + hi) / 2

# ===== БЛОК 3. РОЗГОРТАННЯ ТАБЛИЦІ ЧАСТОТ У ВИБІРКУ (n пар) =====
xs = []
ys = []
i = 0
while i < 4:
    j = 0
    while j < 5:
        c = 0
        while c < freq[i][j]:
            xs.append(X_values[i])
            ys.append(Y_values[j])
            c = c + 1
        j = j + 1
    i = i + 1
N = count(xs)

# ===== БЛОК 4. ЧИСЛОВІ ХАРАКТЕРИСТИКИ =====
sum_x = total(xs)
sum_y = total(ys)
mean_x = sum_x / N
mean_y = sum_y / N
sum_xx = 0.0
sum_yy = 0.0
sum_xy = 0.0
i = 0
while i < N:
    sum_xx = sum_xx + xs[i] * xs[i]
    sum_yy = sum_yy + ys[i] * ys[i]
    sum_xy = sum_xy + xs[i] * ys[i]
    i = i + 1
var_x = sum_xx / N - mean_x * mean_x
var_y = sum_yy / N - mean_y * mean_y
sigma_x = my_sqrt(var_x)
sigma_y = my_sqrt(var_y)
cv_x = sigma_x / mean_x * 100
cv_y = sigma_y / mean_y * 100

# ===== БЛОК 5. КОЕФІЦІЄНТ КОРЕЛЯЦІЇ =====
# Формула 4.2: r = (mean(xy) - mean(x)·mean(y)) / (σx·σy)
cov_xy = sum_xy / N - mean_x * mean_y
r = cov_xy / (sigma_x * sigma_y)

# Перевірка іншою формулою: r = Σ(x-x̄)(y-ȳ) / √(Σ(x-x̄)²·Σ(y-ȳ)²)
sa = 0.0
sb = 0.0
sc = 0.0
i = 0
while i < N:
    dx = xs[i] - mean_x
    dy = ys[i] - mean_y
    sa = sa + dx * dy
    sb = sb + dx * dx
    sc = sc + dy * dy
    i = i + 1
r_check = sa / my_sqrt(sb * sc)

# Рівняння регресії y = a0 + a1·x та коефіцієнт детермінації
a1 = cov_xy / var_x
a0 = mean_y - a1 * mean_x
r2 = r * r

# ===== БЛОК 6. ПЕРЕВІРКА ЗНАЧУЩОСТІ (t-критерій Стьюдента) =====
k_deg = N - 2
t_obs = r * my_sqrt(k_deg) / my_sqrt(1 - r * r)
t_cr = t_critical(alpha, k_deg)
p_val = p_two_sided(t_obs, k_deg)
significant = my_abs(t_obs) > t_cr

# ===== БЛОК 7. ВИВЕДЕННЯ РЕЗУЛЬТАТІВ =====
print("ЛАБОРАТОРНА РОБОТА №4. КОРЕЛЯЦІЙНИЙ АНАЛІЗ")
print("Варіант: l = %d, m = %d, n = %d" % (l, m, n))
print("-" * 62)
print("Кореляційна таблиця (рядки - X, стовпці - Y):")
i = 0
while i < 4:
    print("X = %5.1f :  %s   n_i = %d" % (X_values[i], freq[i], total(freq[i])))
    i = i + 1
print("Обсяг вибірки n = %d" % N)
print("-" * 62)
print("Σx = %.2f   Σy = %.2f" % (sum_x, sum_y))
print("Σx² = %.2f   Σy² = %.2f   Σxy = %.2f" % (sum_xx, sum_yy, sum_xy))
print("x̄ = %.4f   ȳ = %.4f" % (mean_x, mean_y))
print("Dx = %.4f   Dy = %.4f" % (var_x, var_y))
print("σx = %.4f   σy = %.4f" % (sigma_x, sigma_y))
print("Vx = %.2f%% (%s)" % (cv_x, variation_level(cv_x)))
print("Vy = %.2f%% (%s)" % (cv_y, variation_level(cv_y)))
print("cov(x,y) = %.4f" % cov_xy)
print("-" * 62)
print("Коефіцієнт кореляції r (формула 4.2) = %.4f" % r)
print("Перевірка за відхиленнями          r = %.4f" % r_check)
print("Напрям зв'язку: %s" % ("прямий (позитивний)" if r > 0 else "зворотний (негативний)"))
print("Щільність зв'язку: %s" % strength_level(r))
print("r² = %.4f (%.2f%% варіації Y пояснюється X)" % (r2, r2 * 100))
print("Рівняння регресії: y = %.4f + %.4f·x" % (a0, a1))
print("-" * 62)
print("Перевірка гіпотези H0: r_ген = 0, H1: r_ген ≠ 0, α = %.2f" % alpha)
print("Ступені свободи k = n - 2 = %d" % k_deg)
print("t_спост = %.4f" % t_obs)
print("t_кр(%.2f; %d) = %.4f" % (alpha, k_deg, t_cr))
print("p-значення = %.3e" % p_val)
if significant:
    print("|t_спост| > t_кр  =>  H0 відхиляємо, r значущий")
else:
    print("|t_спост| <= t_кр =>  H0 не відхиляємо, r незначущий")

# ===== БЛОК 8. ВІЗУАЛІЗАЦІЯ (matplotlib - лише для малювання) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
OUT = sys.argv[1] + "/"

# 8.1 Дані X та Y за номерами підприємств
idx = []
i = 0
while i < N:
    idx.append(i + 1)
    i = i + 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(idx, xs, "o-", color="#1f5fa8")
ax[0].axhline(mean_x, color="gray", ls="--", label="x̄ = %.1f" % mean_x)
ax[0].set_title("Основні фонди X")
ax[0].set_xlabel("Номер підприємства (за зростанням X)")
ax[0].set_ylabel("млн. грн.")
ax[0].grid(alpha=0.3)
ax[0].legend()
ax[1].plot(idx, ys, "s-", color="#c0392b")
ax[1].axhline(mean_y, color="gray", ls="--", label="ȳ = %.1f" % mean_y)
ax[1].set_title("Річний обсяг виробництва Y")
ax[1].set_xlabel("Номер підприємства (за зростанням X)")
ax[1].set_ylabel("млн. грн.")
ax[1].grid(alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT + "fig_xy.png", dpi=200)
plt.close(fig)

# 8.2 Кореляційне поле з лінією регресії (розмір точки ~ частота)
fig, ax = plt.subplots(figsize=(7, 5))
i = 0
while i < 4:
    j = 0
    while j < 5:
        if freq[i][j] > 0:
            ax.scatter(X_values[i], Y_values[j], s=120 * freq[i][j], color="#1f5fa8", alpha=0.6, edgecolor="black")
            ax.annotate(str(freq[i][j]), (X_values[i], Y_values[j]), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        j = j + 1
    i = i + 1
lx = [10.0, 30.0]
ly = [a0 + a1 * 10.0, a0 + a1 * 30.0]
ax.plot(lx, ly, color="#c0392b", label="y = %.3f + %.3f·x" % (a0, a1))
ax.axvline(mean_x, color="gray", ls=":")
ax.axhline(mean_y, color="gray", ls=":")
ax.set_xlabel("Основні фонди X, млн. грн.")
ax.set_ylabel("Річний обсяг виробництва Y, млн. грн.")
ax.set_title("Кореляційне поле (r = %.4f)" % r)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT + "fig_field.png", dpi=200)
plt.close(fig)

# 8.3 Розподіл Стьюдента: критичні області та t_спост (щільність - власна функція)
px = []
py = []
v = -12.0
while v <= 12.0001:
    px.append(v)
    py.append(t_density(v, k_deg))
    v = v + 0.05
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(px, py, color="black")
fx = []
fy = []
i = 0
while i < count(px):
    if px[i] >= t_cr:
        fx.append(px[i])
        fy.append(py[i])
    i = i + 1
ax.fill_between(fx, fy, color="#c0392b", alpha=0.5, label="критична область (α = 0,05)")
gx = []
gy = []
i = 0
while i < count(px):
    if px[i] <= -t_cr:
        gx.append(px[i])
        gy.append(py[i])
    i = i + 1
ax.fill_between(gx, gy, color="#c0392b", alpha=0.5)
ax.axvline(t_obs, color="#1f5fa8", lw=2, label="t_спост = %.2f" % t_obs)
ax.axvline(t_cr, color="gray", ls="--", label="±t_кр = %.3f" % t_cr)
ax.axvline(-t_cr, color="gray", ls="--")
ax.set_title("Розподіл Стьюдента, k = %d" % k_deg)
ax.set_xlabel("t")
ax.set_ylabel("f(t)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "fig_t.png", dpi=200)
plt.close(fig)
