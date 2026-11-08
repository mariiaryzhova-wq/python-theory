# -*- coding: utf-8 -*-
# Лабораторна робота №5. Регресійний аналіз (парна лінійна регресія)
# Усі обчислення виконані "вручну": без sum, len, sqrt, abs, range, round, min, max,
# sorted, без модулів math / random / statistics / numpy / scipy. Використано лише
# арифметичні оператори, цикли while, списки та print. Бібліотека matplotlib
# застосована ВИКЛЮЧНО для малювання графіків (блок 9).

# ===== БЛОК 1. ПАРАМЕТРИ ТА ГЕНЕРАЦІЯ ВИПАДКОВИХ ДАНИХ =====
n = 30                 # кількість спостережень
SEED = 20261108        # початкове значення генератора (для відтворюваності)
TRUE_B0 = 20.0         # "істинний" вільний член моделі (тис. $)
TRUE_B1 = 12.0         # "істинний" коефіцієнт (тис. $ на 1 °C)
SIGMA = 55.0           # стандартне відхилення випадкової складової (тис. $)
alpha = 0.05
X0_FORECAST = 25.0     # значення X для прогнозу
PI = 3.141592653589793

state = [SEED]
def rnd():
    # лінійний конгруентний генератор: випадкове число з [0; 1)
    state[0] = (1103515245 * state[0] + 12345) % 2147483648
    return state[0] / 2147483648

def rnd_normal():
    # N(0; 1) наближено: сума 12 рівномірних чисел мінус 6 (теорема Ліндеберга-Леві)
    s = 0.0
    i = 0
    while i < 12:
        s = s + rnd()
        i = i + 1
    return s - 6.0

# X - середньомісячна температура повітря, °C (рівномірно від 2 до 36)
# Y - обсяг продажів морозива за місяць, тис. $ (лінійна залежність + випадкова похибка)
xs = []
ys = []
i = 0
while i < n:
    x = 2.0 + 34.0 * rnd()
    x = ((x * 10 + 0.5) // 1) / 10          # округлення до 0,1
    y = TRUE_B0 + TRUE_B1 * x + SIGMA * rnd_normal()
    y = ((y * 10 + 0.5) // 1) / 10
    xs.append(x)
    ys.append(y)
    i = i + 1

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

def gamma_half(two_z):
    # Γ(two_z / 2): Γ(z+1) = z·Γ(z), Γ(1) = 1, Γ(1/2) = √π
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
    # ∫₀ᵗ f(x)dx за формулою Сімпсона (steps - парне)
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
    while i < 60:
        mid = (lo + hi) / 2
        if p_two_sided(mid, k) > a:
            lo = mid
        else:
            hi = mid
        i = i + 1
    return (lo + hi) / 2

def determination_level(r2):
    # словесна оцінка коефіцієнта детермінації
    if r2 < 0.3:
        return "слабка пояснювальна здатність моделі"
    if r2 < 0.5:
        return "помірна пояснювальна здатність моделі"
    if r2 < 0.7:
        return "задовільна пояснювальна здатність моделі"
    if r2 < 0.9:
        return "висока пояснювальна здатність моделі"
    return "дуже висока пояснювальна здатність моделі"

# ===== БЛОК 3. ОЦІНЮВАННЯ ПАРАМЕТРІВ РЕГРЕСІЇ (МНК) =====
sum_x = total(xs)
sum_y = total(ys)
mean_x = sum_x / n
mean_y = sum_y / n
Sxx = 0.0     # Σ(x - x̄)²
Syy = 0.0     # Σ(y - ȳ)²  (повна сума квадратів SST)
Sxy = 0.0     # Σ(x - x̄)(y - ȳ)
i = 0
while i < n:
    dx = xs[i] - mean_x
    dy = ys[i] - mean_y
    Sxx = Sxx + dx * dx
    Syy = Syy + dy * dy
    Sxy = Sxy + dx * dy
    i = i + 1
b1 = Sxy / Sxx                    # коефіцієнт нахилу
b0 = mean_y - b1 * mean_x         # вільний член

# Перевірка через нормальні рівняння: n·b0 + b1·Σx = Σy;  b0·Σx + b1·Σx² = Σxy
sum_xx = 0.0
sum_xy = 0.0
i = 0
while i < n:
    sum_xx = sum_xx + xs[i] * xs[i]
    sum_xy = sum_xy + xs[i] * ys[i]
    i = i + 1
det = n * sum_xx - sum_x * sum_x
b1_check = (n * sum_xy - sum_x * sum_y) / det
b0_check = (sum_y * sum_xx - sum_x * sum_xy) / det

# ===== БЛОК 4. ПРОГНОЗНІ ЗНАЧЕННЯ ТА ЗАЛИШКИ =====
fit = []       # ŷ_i
res = []       # e_i = y_i - ŷ_i
SSE = 0.0      # сума квадратів залишків
SSR = 0.0      # пояснена сума квадратів
sum_e = 0.0
sum_ex = 0.0
sum_efit = 0.0
i = 0
while i < n:
    f = b0 + b1 * xs[i]
    e = ys[i] - f
    fit.append(f)
    res.append(e)
    SSE = SSE + e * e
    SSR = SSR + (f - mean_y) * (f - mean_y)
    sum_e = sum_e + e
    sum_ex = sum_ex + e * xs[i]
    sum_efit = sum_efit + e * f
    i = i + 1
SST = Syy

# ===== БЛОК 5. КОЕФІЦІЄНТ ДЕТЕРМІНАЦІЇ ТА ЗВ'ЯЗОК З КОЕФІЦІЄНТОМ КОРЕЛЯЦІЇ =====
R2 = SSR / SST
R2_alt = 1 - SSE / SST
r = Sxy / my_sqrt(Sxx * Syy)           # коефіцієнт кореляції
r_squared = r * r
sigma_x = my_sqrt(Sxx / n)
sigma_y = my_sqrt(Syy / n)
r_from_b1 = b1 * sigma_x / sigma_y     # r = b1·σx/σy
df = n - 2
R2_adj = 1 - (1 - R2) * (n - 1) / df

# ===== БЛОК 6. СТАНДАРТНІ ПОМИЛКИ, t- ТА F-КРИТЕРІЇ =====
s2 = SSE / df                          # незміщена оцінка дисперсії залишків
s_err = my_sqrt(s2)                    # стандартна похибка оцінки Y
se_b1 = s_err / my_sqrt(Sxx)
se_b0 = s_err * my_sqrt(1.0 / n + mean_x * mean_x / Sxx)
t_b1 = b1 / se_b1
t_b0 = b0 / se_b0
t_cr = t_critical(alpha, df)
p_b1 = p_two_sided(t_b1, df)
p_b0 = p_two_sided(t_b0, df)
F_stat = (SSR / 1) / (SSE / df)        # для парної регресії F = t²
b1_low = b1 - t_cr * se_b1
b1_high = b1 + t_cr * se_b1
b0_low = b0 - t_cr * se_b0
b0_high = b0 + t_cr * se_b0

# ===== БЛОК 7. АНАЛІЗ ЗАЛИШКІВ =====
mean_e = sum_e / n
std_res = []        # стьюдентизовані залишки e_i / (s·√(1 - h_i))
leverage = []
outliers = 0
within1 = 0
within2 = 0
e_abs_sum = 0.0
i = 0
while i < n:
    h = 1.0 / n + (xs[i] - mean_x) * (xs[i] - mean_x) / Sxx
    leverage.append(h)
    z = res[i] / (s_err * my_sqrt(1 - h))
    std_res.append(z)
    if my_abs(z) > 2:
        outliers = outliers + 1
    if my_abs(res[i]) <= s_err:
        within1 = within1 + 1
    if my_abs(res[i]) <= 2 * s_err:
        within2 = within2 + 1
    i = i + 1
# Статистика Дарбіна-Уотсона (автокореляція залишків у часі)
dw_num = 0.0
i = 1
while i < n:
    dd = res[i] - res[i - 1]
    dw_num = dw_num + dd * dd
    i = i + 1
DW = dw_num / SSE
# Асиметрія та ексцес залишків
m3 = 0.0
m4 = 0.0
i = 0
while i < n:
    m3 = m3 + res[i] * res[i] * res[i]
    m4 = m4 + res[i] * res[i] * res[i] * res[i]
    i = i + 1
m2 = SSE / n
skew = (m3 / n) / (m2 * my_sqrt(m2))
kurt = (m4 / n) / (m2 * m2) - 3
# Кореляція |e| і x (ознака гетероскедастичності)
mean_abs_e = 0.0
i = 0
while i < n:
    mean_abs_e = mean_abs_e + my_abs(res[i])
    i = i + 1
mean_abs_e = mean_abs_e / n
sax = 0.0
saa = 0.0
i = 0
while i < n:
    da = my_abs(res[i]) - mean_abs_e
    sax = sax + da * (xs[i] - mean_x)
    saa = saa + da * da
    i = i + 1
r_abs_e_x = sax / my_sqrt(saa * Sxx)

# ===== БЛОК 8. ПРОГНОЗ ТА ДОВІРЧІ ІНТЕРВАЛИ =====
y_forecast = b0 + b1 * X0_FORECAST
se_mean = s_err * my_sqrt(1.0 / n + (X0_FORECAST - mean_x) * (X0_FORECAST - mean_x) / Sxx)
se_pred = s_err * my_sqrt(1.0 + 1.0 / n + (X0_FORECAST - mean_x) * (X0_FORECAST - mean_x) / Sxx)
ci_low = y_forecast - t_cr * se_mean
ci_high = y_forecast + t_cr * se_mean
pi_low = y_forecast - t_cr * se_pred
pi_high = y_forecast + t_cr * se_pred

# ===== БЛОК 9. ВИВЕДЕННЯ РЕЗУЛЬТАТІВ =====
print("ЛАБОРАТОРНА РОБОТА №5. РЕГРЕСІЙНИЙ АНАЛІЗ")
print("Парна лінійна регресія: Y = b0 + b1*X + e,  n = %d" % n)
print("X - температура, °C;  Y - продажі морозива, тис. $")
print("-" * 70)
print(" №      X      Y       Y_розр      залишок e     станд. залишок")
i = 0
while i < n:
    print("%2d  %5.1f  %7.1f  %9.2f  %12.2f  %12.3f" % (i + 1, xs[i], ys[i], fit[i], res[i], std_res[i]))
    i = i + 1
print("-" * 70)
print("Σx = %.1f   Σy = %.1f   x_ср = %.4f   y_ср = %.4f" % (sum_x, sum_y, mean_x, mean_y))
print("Sxx = %.4f   Syy = %.4f   Sxy = %.4f" % (Sxx, Syy, Sxy))
print("-" * 70)
print("ПАРАМЕТРИ РЕГРЕСІЇ (МНК)")
print("b1 = Sxy/Sxx = %.4f   (перевірка нормальними рівняннями: %.4f)" % (b1, b1_check))
print("b0 = y_ср - b1*x_ср = %.4f   (перевірка: %.4f)" % (b0, b0_check))
print("Рівняння регресії: Y = %.4f + %.4f * X" % (b0, b1))
print("(\"істинна\" модель: Y = %.1f + %.1f * X)" % (TRUE_B0, TRUE_B1))
print("-" * 70)
print("СУМИ КВАДРАТІВ ТА КОЕФІЦІЄНТ ДЕТЕРМІНАЦІЇ")
print("SST = %.2f   SSR = %.2f   SSE = %.2f" % (SST, SSR, SSE))
print("SSR + SSE = %.2f (має дорівнювати SST)" % (SSR + SSE))
print("R² = SSR/SST = %.6f   (1 - SSE/SST = %.6f)" % (R2, R2_alt))
print("Скоригований R² = %.6f" % R2_adj)
print("Оцінка: %s" % determination_level(R2))
print("Коефіцієнт кореляції r = %.6f   r² = %.6f" % (r, r_squared))
print("r = b1*σx/σy = %.6f" % r_from_b1)
print("|R² - r²| = %.2e  => R² = r²" % my_abs(R2 - r_squared))
print("-" * 70)
print("СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ МОДЕЛІ")
print("Стандартна похибка оцінки Y: s = %.4f" % s_err)
print("se(b1) = %.4f   se(b0) = %.4f" % (se_b1, se_b0))
print("t(b1) = %.4f   p = %.3e" % (t_b1, p_b1))
print("t(b0) = %.4f   p = %.4f" % (t_b0, p_b0))
print("t_кр(%.2f; %d) = %.4f" % (alpha, df, t_cr))
print("F = %.4f   (t(b1)² = %.4f)" % (F_stat, t_b1 * t_b1))
print("95%% ДІ для b1: [%.4f; %.4f]" % (b1_low, b1_high))
print("95%% ДІ для b0: [%.4f; %.4f]" % (b0_low, b0_high))
print("Значущість b1: %s" % ("значущий (|t| > t_кр)" if my_abs(t_b1) > t_cr else "незначущий"))
print("Значущість b0: %s" % ("значущий (|t| > t_кр)" if my_abs(t_b0) > t_cr else "незначущий"))
print("-" * 70)
print("АНАЛІЗ ЗАЛИШКІВ")
print("Σe = %.3e   середнє e = %.3e" % (sum_e, mean_e))
print("Σe·x = %.3e   Σe·Y_розр = %.3e" % (sum_ex, sum_efit))
print("Залишків у межах ±s: %d з %d (%.1f%%)" % (within1, n, within1 * 100.0 / n))
print("Залишків у межах ±2s: %d з %d (%.1f%%)" % (within2, n, within2 * 100.0 / n))
print("Викидів (|станд. залишок| > 2): %d" % outliers)
print("Асиметрія = %.4f   ексцес = %.4f" % (skew, kurt))
print("Дарбін-Уотсон DW = %.4f" % DW)
print("r(|e|, x) = %.4f" % r_abs_e_x)
print("-" * 70)
print("ПРОГНОЗ при X = %.1f °C" % X0_FORECAST)
print("Y_прогноз = %.2f тис. $" % y_forecast)
print("95%% ДІ середнього значення: [%.2f; %.2f]" % (ci_low, ci_high))
print("95%% інтервал прогнозу окремого значення: [%.2f; %.2f]" % (pi_low, pi_high))

# ===== БЛОК 10. ВІЗУАЛІЗАЦІЯ (matplotlib - лише для малювання) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
OUT = sys.argv[1] + "/"
idx = []
i = 0
while i < n:
    idx.append(i + 1)
    i = i + 1

# 10.1 Дані X та Y за номерами спостережень
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(idx, xs, "o-", color="#1f5fa8")
ax[0].axhline(mean_x, color="gray", ls="--", label="x_ср = %.1f" % mean_x)
ax[0].set_title("Температура X")
ax[0].set_xlabel("Номер спостереження")
ax[0].set_ylabel("°C")
ax[0].grid(alpha=0.3)
ax[0].legend()
ax[1].plot(idx, ys, "s-", color="#c0392b")
ax[1].axhline(mean_y, color="gray", ls="--", label="y_ср = %.1f" % mean_y)
ax[1].set_title("Обсяг продажів Y")
ax[1].set_xlabel("Номер спостереження")
ax[1].set_ylabel("тис. $")
ax[1].grid(alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT + "fig_xy.png", dpi=200)
plt.close(fig)

# 10.2 Діаграма розсіювання, лінія регресії та довірчі смуги
gx = []
g_fit = []
g_cl = []
g_ch = []
g_pl = []
g_ph = []
x = 0.0
while x <= 38.0001:
    f = b0 + b1 * x
    q = (x - mean_x) * (x - mean_x) / Sxx
    sm = s_err * my_sqrt(1.0 / n + q)
    sp = s_err * my_sqrt(1.0 + 1.0 / n + q)
    gx.append(x)
    g_fit.append(f)
    g_cl.append(f - t_cr * sm)
    g_ch.append(f + t_cr * sm)
    g_pl.append(f - t_cr * sp)
    g_ph.append(f + t_cr * sp)
    x = x + 0.5
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.fill_between(gx, g_pl, g_ph, color="#f5b7b1", alpha=0.35, label="95% інтервал прогнозу")
ax.fill_between(gx, g_cl, g_ch, color="#85c1e9", alpha=0.55, label="95% ДІ середнього")
ax.scatter(xs, ys, color="#1f5fa8", edgecolor="black", zorder=3, label="спостереження")
ax.plot(gx, g_fit, color="#c0392b", lw=2, label="Y = %.2f + %.3f·X" % (b0, b1))
ax.scatter([X0_FORECAST], [y_forecast], marker="*", s=220, color="gold", edgecolor="black", zorder=4, label="прогноз при X = %.0f" % X0_FORECAST)
ax.set_xlabel("Температура X, °C")
ax.set_ylabel("Обсяг продажів Y, тис. $")
ax.set_title("Лінія регресії (R² = %.4f)" % R2)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT + "fig_reg.png", dpi=200)
plt.close(fig)

# 10.3 Залишки: від прогнозу, за номером, гістограма
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].scatter(fit, res, color="#1f5fa8", edgecolor="black")
ax[0].axhline(0, color="#c0392b")
ax[0].axhline(2 * s_err, color="gray", ls="--")
ax[0].axhline(-2 * s_err, color="gray", ls="--")
ax[0].set_title("Залишки та прогноз")
ax[0].set_xlabel("Розрахункове Y")
ax[0].set_ylabel("e")
ax[0].grid(alpha=0.3)
ax[1].plot(idx, res, "o-", color="#1f5fa8")
ax[1].axhline(0, color="#c0392b")
ax[1].set_title("Залишки за порядком спостережень")
ax[1].set_xlabel("Номер спостереження")
ax[1].set_ylabel("e")
ax[1].grid(alpha=0.3)
# гістограма залишків (підрахунок вручну)
bins = 12
lo = -3.0 * s_err
width = s_err / 2
counts = []
j = 0
while j < bins:
    counts.append(0)
    j = j + 1
i = 0
while i < n:
    pos = (res[i] - lo) / width
    k = 0
    while k < bins - 1 and pos >= k + 1:
        k = k + 1
    counts[k] = counts[k] + 1
    i = i + 1
centers = []
j = 0
while j < bins:
    centers.append(lo + (j + 0.5) * width)
    j = j + 1
ax[2].bar(centers, counts, width=width * 0.95, color="#85c1e9", edgecolor="black")
ax[2].set_title("Гістограма залишків")
ax[2].set_xlabel("e")
ax[2].set_ylabel("кількість")
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "fig_res.png", dpi=200)
plt.close(fig)

# 10.4 Зв'язок R² і r: поле та поле в стандартизованих величинах
zx = []
zy = []
i = 0
while i < n:
    zx.append((xs[i] - mean_x) / sigma_x)
    zy.append((ys[i] - mean_y) / sigma_y)
    i = i + 1
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.scatter(zx, zy, color="#1f5fa8", edgecolor="black", zorder=3)
ax.plot([-2.2, 2.2], [r * -2.2, r * 2.2], color="#c0392b", lw=2, label="z_y = r·z_x,  r = %.4f" % r)
ax.plot([-2.2, 2.2], [-2.2, 2.2], color="gray", ls="--", label="z_y = z_x (r = 1)")
ax.set_xlabel("z_x")
ax.set_ylabel("z_y")
ax.set_title("Стандартизовані величини: нахил = r,  R² = r² = %.4f" % r_squared, fontsize=10)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT + "fig_rr.png", dpi=200)
plt.close(fig)
