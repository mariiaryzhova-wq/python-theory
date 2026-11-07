# -*- coding: utf-8 -*-
"""
Лабораторна робота №3. Закони розподілу.

Обмеження: НЕ використовуються вбудовані/бібліотечні функції для
випадкових чисел та математики (random, math, numpy, statistics, pow, abs,
sum, min, max, round, int тощо). Усе реалізовано вручну:
  - генератор псевдовипадкових чисел (xorshift64*),
  - sqrt, ln, exp, sin, cos (ряди Тейлора, метод Ньютона),
  - рівномірний, експоненційний, нормальний розподіли,
  - метод Бокса-Мюллера (класичний та полярний) і алгоритм зиккурат,
  - статистики (середнє, дисперсія, асиметрія, ексцес), гістограми.
Бібліотека matplotlib застосовується ЛИШЕ для малювання готових масивів,
time.perf_counter - ЛИШЕ для вимірювання часу.
"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PI = 3.141592653589793
TWO_PI = 6.283185307179586
MASK64 = 0xFFFFFFFFFFFFFFFF
INV_2_53 = 1.0 / 9007199254740992.0          # 1 / 2^53


# =====================================================================
# 1. Елементарні математичні функції (власна реалізація)
# =====================================================================
def my_abs(x):
    """Модуль числа."""
    return x if x >= 0 else -x


def my_sqrt(x):
    """Квадратний корінь методом Ньютона: g <- (g + x/g) / 2."""
    if x <= 0:
        return 0.0
    g = x if x >= 1.0 else 1.0
    for _ in range(200):
        g_new = 0.5 * (g + x / g)
        if my_abs(g_new - g) <= 1e-15 * g_new:
            return g_new
        g = g_new
    return g


def _atanh_series(z):
    """ln((1+z)/(1-z)) / 2 = z + z^3/3 + z^5/5 + ... (|z| < 1)."""
    z2 = z * z
    term = z
    s = 0.0
    n = 1
    for _ in range(40):
        s += term / n
        term *= z2
        n += 2
    return s


LN2 = 2.0 * _atanh_series(1.0 / 3.0)         # ln 2 = 2*atanh(1/3)


def my_ln(x):
    """Натуральний логарифм: зведення x до [0.7071; 1.4142] та ряд для atanh."""
    if x <= 0:
        return -1e300
    e = 0
    while x > 1.4142135623730951:
        x *= 0.5
        e += 1
    while x < 0.7071067811865476:
        x *= 2.0
        e -= 1
    z = (x - 1.0) / (x + 1.0)
    return 2.0 * _atanh_series(z) + e * LN2


def my_exp(x):
    """Експонента: e^x = (e^(x/2^k))^(2^k), ряд Тейлора для малого аргументу."""
    if x < -745.0:
        return 0.0
    neg = x < 0
    if neg:
        x = -x
    k = 0
    while x > 0.5:
        x *= 0.5
        k += 1
    term = 1.0
    s = 1.0
    for n in range(1, 25):
        term *= x / n
        s += term
    for _ in range(k):
        s *= s
    return 1.0 / s if neg else s


def my_pow(base, p):
    """base^p для base > 0 через exp(p * ln(base))."""
    return my_exp(p * my_ln(base))


def my_sin(x):
    """Синус: зведення до [-pi; pi] та ряд Тейлора."""
    while x > PI:
        x -= TWO_PI
    while x < -PI:
        x += TWO_PI
    term = x
    s = x
    x2 = x * x
    for n in range(1, 16):
        term *= -x2 / ((2 * n) * (2 * n + 1))
        s += term
    return s


def my_cos(x):
    """Косинус: зведення до [-pi; pi] та ряд Тейлора."""
    while x > PI:
        x -= TWO_PI
    while x < -PI:
        x += TWO_PI
    term = 1.0
    s = 1.0
    x2 = x * x
    for n in range(1, 16):
        term *= -x2 / ((2 * n - 1) * (2 * n))
        s += term
    return s


def gamma_half(n):
    """Гамма-функція Г(n/2) для натурального n (через Г(1)=1, Г(1/2)=sqrt(pi))."""
    if n % 2 == 0:
        g, x = 1.0, 1.0
    else:
        g, x = my_sqrt(PI), 0.5
    while x < n / 2.0 - 1e-9:
        g *= x
        x += 1.0
    return g


def simpson(f, a, b, n=2000):
    """Чисельне інтегрування за формулою Сімпсона (n - парне)."""
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4.0 if i % 2 == 1 else 2.0) * f(a + i * h)
    return s * h / 3.0


# =====================================================================
# 2. Генератор рівномірних псевдовипадкових чисел (xorshift64*)
# =====================================================================
class Xorshift64Star:
    """Генератор xorshift64* (Марсалья/Вінья): 64-бітний стан, період 2^64-1."""

    def __init__(self, seed):
        self.state = (seed & MASK64) or 1

    def next_u64(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self.state = x
        return (x * 2685821657736338717) & MASK64

    def next_float(self):
        """Рівномірне число у відкритому інтервалі (0; 1)."""
        return ((self.next_u64() >> 11) + 0.5) * INV_2_53


rng = Xorshift64Star(88172645463325252)


# =====================================================================
# 3. Рівномірний розподіл на [a; b]
# =====================================================================
def uniform(a, b):
    """Рівномірно розподілена величина на [a; b]: x = a + (b - a) * u."""
    return a + (b - a) * rng.next_float()


# =====================================================================
# 4. Експоненційний розподіл
# =====================================================================
def exponential(lam=1.0):
    """Метод оберненої функції: y = -ln(u) / lambda, u ~ U(0;1)."""
    return -my_ln(rng.next_float()) / lam


def exponential_01(lam=3.0):
    """Експоненційна величина у діапазоні [0; 1] (відкидання значень y > 1)."""
    while True:
        y = exponential(lam)
        if y <= 1.0:
            return y


# =====================================================================
# 5. Нормальний розподіл (три алгоритми)
# =====================================================================
_spare_bm = [None]
_spare_polar = [None]


def normal_std_boxmuller():
    """Класичний метод Бокса-Мюллера (з тригонометрією): дає пару N(0;1)."""
    if _spare_bm[0] is not None:
        z = _spare_bm[0]
        _spare_bm[0] = None
        return z
    u1 = rng.next_float()
    u2 = rng.next_float()
    r = my_sqrt(-2.0 * my_ln(u1))
    phi = TWO_PI * u2
    _spare_bm[0] = r * my_sin(phi)
    return r * my_cos(phi)


def normal_std_polar():
    """Полярний метод Бокса-Мюллера: точка (v1; v2) всередині одиничного кола,
    cos та sin замінюються на v1/sqrt(R2) та v2/sqrt(R2) (без тригонометрії)."""
    if _spare_polar[0] is not None:
        z = _spare_polar[0]
        _spare_polar[0] = None
        return z
    while True:
        v1 = 2.0 * rng.next_float() - 1.0
        v2 = 2.0 * rng.next_float() - 1.0
        r2 = v1 * v1 + v2 * v2
        if 0.0 < r2 < 1.0:
            break
    f = my_sqrt(-2.0 * my_ln(r2) / r2)
    _spare_polar[0] = v2 * f
    return v1 * f


# ---------- Зиккурат-алгоритм (Марсалья-Цанг, 128 шарів) ----------
ZIG_R = 3.442619855899                 # правий край основи (хвіст)
ZIG_V = 9.91256303526217e-3            # площа кожного шару
KN = [0.0] * 128                       # пороги швидкого прийняття
WN = [0.0] * 128                       # масштаби
FN = [0.0] * 128                       # значення густини f(x) = exp(-x^2/2)


def build_ziggurat_tables():
    """Побудова таблиць шарів зиккурату (власні exp, ln, sqrt)."""
    m1 = 2147483648.0                  # 2^31
    dn = ZIG_R
    tn = dn
    q = ZIG_V / my_exp(-0.5 * dn * dn)
    KN[0] = (dn / q) * m1
    KN[1] = 0.0
    WN[0] = q / m1
    WN[127] = dn / m1
    FN[0] = 1.0
    FN[127] = my_exp(-0.5 * dn * dn)
    for i in range(126, 0, -1):
        dn = my_sqrt(-2.0 * my_ln(ZIG_V / dn + my_exp(-0.5 * dn * dn)))
        KN[i + 1] = (dn / tn) * m1
        tn = dn
        FN[i] = my_exp(-0.5 * dn * dn)
        WN[i] = dn / m1


build_ziggurat_tables()


def normal_std_ziggurat():
    """Нормально розподілена N(0;1) величина алгоритмом зиккурат."""
    while True:
        u = rng.next_u64()
        j = (u >> 32) - 2147483648     # знакове 32-бітне ціле
        i = (u >> 20) & 127            # номер шару (7 бітів)
        aj = j if j >= 0 else -j
        if aj < KN[i]:                 # ~98.8 % випадків: точка під кривою
            return j * WN[i]
        if i == 0:                     # хвіст розподілу (|x| > r)
            while True:
                x = -my_ln(rng.next_float()) / ZIG_R
                y = -my_ln(rng.next_float())
                if y + y >= x * x:
                    break
            return ZIG_R + x if j > 0 else -ZIG_R - x
        x = j * WN[i]                  # клин між шарами: перевірка по f(x)
        if FN[i] + rng.next_float() * (FN[i - 1] - FN[i]) < my_exp(-0.5 * x * x):
            return x


def normal(mu=0.0, sigma=1.0):
    """N(mu; sigma): x = mu + sigma * z."""
    return mu + sigma * normal_std_polar()


def normal_01(mu=0.5, sigma=0.15):
    """Нормально розподілена величина у діапазоні [0; 1] (відкидання виходу за межі)."""
    while True:
        x = normal(mu, sigma)
        if 0.0 <= x <= 1.0:
            return x


# =====================================================================
# 6. Розподіли, пов'язані з нормальним (для контрольних питань)
# =====================================================================
def chi_square(k):
    """Хі-квадрат з k ступенями свободи: сума k квадратів N(0;1)."""
    s = 0.0
    for _ in range(k):
        z = normal_std_polar()
        s += z * z
    return s


def student_t(k):
    """Стьюдента: t = Z / sqrt(chi2_k / k)."""
    return normal_std_polar() / my_sqrt(chi_square(k) / k)


def fisher_f(d1, d2):
    """Фішера: F = (chi2_d1 / d1) / (chi2_d2 / d2)."""
    return (chi_square(d1) / d1) / (chi_square(d2) / d2)


# Теоретичні щільності
def pdf_uniform(x, a, b):
    return 1.0 / (b - a) if a <= x <= b else 0.0


def pdf_exp(x, lam):
    return lam * my_exp(-lam * x) if x >= 0 else 0.0


def pdf_normal(x, mu, sigma):
    d = x - mu
    return my_exp(-d * d / (2.0 * sigma * sigma)) / (sigma * my_sqrt(TWO_PI))


def pdf_chi2(x, k):
    if x <= 0:
        return 0.0
    return my_pow(x, k / 2.0 - 1.0) * my_exp(-x / 2.0) / (my_pow(2.0, k / 2.0) * gamma_half(k))


def pdf_t(x, k):
    c = gamma_half(k + 1) / (my_sqrt(k * PI) * gamma_half(k))
    return c * my_pow(1.0 + x * x / k, -(k + 1) / 2.0)


def pdf_f(x, d1, d2):
    if x <= 0:
        return 0.0
    c = gamma_half(d1 + d2) / (gamma_half(d1) * gamma_half(d2))
    return (c * my_pow(d1 / d2, d1 / 2.0) * my_pow(x, d1 / 2.0 - 1.0)
            * my_pow(1.0 + d1 * x / d2, -(d1 + d2) / 2.0))


# =====================================================================
# 7. Статистика та гістограми (власна реалізація)
# =====================================================================
def my_sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def my_min(xs):
    m = xs[0]
    for x in xs:
        if x < m:
            m = x
    return m


def my_mean(xs):
    return my_sum(xs) / len(xs)


def central_moment(xs, m, p):
    s = 0.0
    for x in xs:
        d = x - m
        t = 1.0
        for _ in range(p):
            t *= d
        s += t
    return s / len(xs)


def describe(xs):
    """Повертає (середнє, дисперсія, асиметрія, надлишковий ексцес)."""
    m = my_mean(xs)
    m2 = central_moment(xs, m, 2)
    m3 = central_moment(xs, m, 3)
    m4 = central_moment(xs, m, 4)
    return m, m2, m3 / (m2 * my_sqrt(m2)), m4 / (m2 * m2) - 3.0


def theory_moments(pdf, a, b):
    """Теоретичні середнє та дисперсія щільності на [a; b] (інтеграл Сімпсона)."""
    z = simpson(pdf, a, b)
    m = simpson(lambda x: x * pdf(x), a, b) / z
    v = simpson(lambda x: (x - m) * (x - m) * pdf(x), a, b) / z
    return m, v


def histogram(xs, lo, hi, bins):
    """Щільність гістограми: рахуємо влучання в стовпчики бінарним пошуком по межах."""
    w = (hi - lo) / bins
    edges = [lo + i * w for i in range(bins + 1)]
    counts = [0] * bins
    for x in xs:
        if x < lo or x >= hi:
            continue
        l, r = 0, bins
        while r - l > 1:
            mid = (l + r) // 2
            if x >= edges[mid]:
                l = mid
            else:
                r = mid
        counts[l] += 1
    n = len(xs)
    dens = [c / (n * w) for c in counts]
    centers = [lo + (i + 0.5) * w for i in range(bins)]
    return centers, dens, w


def linspace(a, b, n):
    h = (b - a) / (n - 1)
    return [a + i * h for i in range(n)]


# =====================================================================
# 8. Побудова графіків (matplotlib лише малює готові списки чисел)
# =====================================================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def draw_hist(ax, xs, lo, hi, bins, curve=None, xr=None, label="Вибірка (гістограма)", color="#4C78A8", loc="upper left"):
    c, d, w = histogram(xs, lo, hi, bins)
    ax.bar(c, d, width=w * 0.95, color=color, alpha=0.85, label=label)
    if curve is not None:
        grid = linspace(xr[0], xr[1], 400)
        ax.plot(grid, [curve(x) for x in grid], color="#E45756", lw=2.2, label="Теоретична щільність")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc=loc)


def main():
    N = 100000
    out = []

    def say(s=""):
        print(s)

    say("=" * 78)
    say("ЛАБОРАТОРНА РОБОТА №3. ЗАКОНИ РОЗПОДІЛУ (без вбудованих функцій)")
    say("=" * 78)
    say(f"Обсяг кожної вибірки N = {N}")
    say(f"Перевірка власних функцій: ln(2)={my_ln(2.0):.12f}, exp(1)={my_exp(1.0):.12f}, "
        f"sqrt(2)={my_sqrt(2.0):.12f}")
    say(f"                           sin(1)={my_sin(1.0):.12f}, cos(1)={my_cos(1.0):.12f}")
    say()

    rows = []   # (назва, середнє_вибірки, середнє_теор, дисп_вибірки, дисп_теор, асиметрія, ексцес)

    # ---------- 1. Рівномірний [a; b] ----------
    a, b = 2.0, 5.0
    data_u = [uniform(a, b) for _ in range(N)]
    m, v, sk, ku = describe(data_u)
    rows.append((f"Рівномірний [{a:.0f}; {b:.0f}]", m, (a + b) / 2, v, (b - a) * (b - a) / 12, sk, ku, 0.0, -1.2))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    draw_hist(ax, data_u, a, b, 20, lambda x: pdf_uniform(x, a, b), (a - 0.5, b + 0.5), loc="upper right")
    ax.set_title(f"Рівномірний розподіл на [{a:.0f}; {b:.0f}], N = {N}")
    ax.set_ylim(0, 0.5)
    fig.tight_layout(); fig.savefig("figs/rys1_uniform.png", dpi=150); plt.close(fig)

    # ---------- 2. Експоненційний [0; 1] ----------
    lam = 3.0
    data_e = [exponential_01(lam) for _ in range(N)]
    pe = lambda x: pdf_exp(x, lam)
    tm, tv = theory_moments(pe, 0.0, 1.0)
    z_e = simpson(pe, 0.0, 1.0)
    m, v, sk, ku = describe(data_e)
    rows.append((f"Експоненційний, λ={lam:.0f}, [0; 1]", m, tm, v, tv, sk, ku, None, None))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    draw_hist(ax, data_e, 0.0, 1.0, 25, lambda x: pe(x) / z_e, (0.0, 1.0), loc="upper right")
    ax.set_title(f"Експоненційний розподіл (λ = {lam:.0f}) на [0; 1], N = {N}")
    fig.tight_layout(); fig.savefig("figs/rys2_exponential.png", dpi=150); plt.close(fig)

    # ---------- 3. Нормальний [0; 1] ----------
    mu, sg = 0.5, 0.15
    data_n = [normal_01(mu, sg) for _ in range(N)]
    pn = lambda x: pdf_normal(x, mu, sg)
    tm, tv = theory_moments(pn, 0.0, 1.0)
    z_n = simpson(pn, 0.0, 1.0)
    m, v, sk, ku = describe(data_n)
    rows.append((f"Нормальний μ={mu}, σ={sg}, [0; 1]", m, tm, v, tv, sk, ku, 0.0, None))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    draw_hist(axs[0], data_n, 0.0, 1.0, 25, lambda x: pn(x) / z_n, (0.0, 1.0))
    axs[0].set_title(f"N(μ={mu}; σ={sg}) на [0; 1], N = {N}")
    grid = linspace(-6.0, 8.0, 500)
    for (m_, s_, col) in ((0.0, 1.0, "#4C78A8"), (0.0, 2.0, "#F58518"), (2.0, 1.0, "#54A24B"), (-2.0, 0.5, "#B279A2")):
        axs[1].plot(grid, [pdf_normal(x, m_, s_) for x in grid], lw=2, color=col, label=f"μ={m_:g}, σ={s_:g}")
    axs[1].set_title("Сімейство нормальних кривих f(x)")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("f(x)"); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig("figs/rys3_normal01.png", dpi=150); plt.close(fig)

    # ---------- 4. Стандартна нормальна: три алгоритми ----------
    gens = (("Бокс-Мюллер (класичний)", normal_std_boxmuller, "#4C78A8"),
            ("Бокс-Мюллер (полярний)", normal_std_polar, "#F58518"),
            ("Зиккурат", normal_std_ziggurat, "#54A24B"))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
    std_stats = []
    rule = []
    for ax, (name, g, col) in zip(axs, gens):
        xs = [g() for _ in range(N)]
        m, v, sk, ku = describe(xs)
        std_stats.append((name, m, v, sk, ku))
        c1 = c2 = c3 = 0
        for x in xs:
            ax_ = my_abs(x)
            if ax_ < 1.0:
                c1 += 1
            if ax_ < 2.0:
                c2 += 1
            if ax_ < 3.0:
                c3 += 1
        rule.append((name, c1 / N * 100, c2 / N * 100, c3 / N * 100))
        draw_hist(ax, xs, -4.0, 4.0, 40, lambda x: pdf_normal(x, 0.0, 1.0), (-4.0, 4.0), color=col)
        ax.set_title(name)
    fig.suptitle(f"Стандартний нормальний розподіл N(0; 1), N = {N}")
    fig.tight_layout(); fig.savefig("figs/rys4_std_normal.png", dpi=150); plt.close(fig)
    for (name, m, v, sk, ku) in std_stats:
        rows.append((f"N(0;1): {name}", m, 0.0, v, 1.0, sk, ku, 0.0, 0.0))

    # ---------- 5. Хі-квадрат, Стьюдент, Фішер ----------
    M = 60000
    k_chi, k_t, d1, d2 = 4, 5, 5, 10
    data_c = [chi_square(k_chi) for _ in range(M)]
    data_t = [student_t(k_t) for _ in range(M)]
    data_f = [fisher_f(d1, d2) for _ in range(M)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    draw_hist(axs[0], data_c, 0.0, 20.0, 40, lambda x: pdf_chi2(x, k_chi), (0.01, 20.0), loc="upper right")
    axs[0].set_title(f"χ²-розподіл, k = {k_chi}")
    draw_hist(axs[1], data_t, -6.0, 6.0, 48, lambda x: pdf_t(x, k_t), (-6.0, 6.0))
    grid = linspace(-6.0, 6.0, 300)
    axs[1].plot(grid, [pdf_normal(x, 0.0, 1.0) for x in grid], "--", color="black", lw=1.4, label="N(0; 1)")
    axs[1].set_ylim(0.0, 0.55)
    axs[1].legend(fontsize=9, loc="upper left")
    axs[1].set_title(f"Розподіл Стьюдента, k = {k_t}")
    draw_hist(axs[2], data_f, 0.0, 6.0, 48, lambda x: pdf_f(x, d1, d2), (0.01, 6.0), loc="upper right")
    axs[2].set_title(f"Розподіл Фішера, d1 = {d1}, d2 = {d2}")
    fig.tight_layout(); fig.savefig("figs/rys5_chi_t_f.png", dpi=150); plt.close(fig)
    m, v, sk, ku = describe(data_c)
    rows.append((f"χ², k={k_chi}", m, float(k_chi), v, 2.0 * k_chi, sk, ku, my_sqrt(8.0 / k_chi), 12.0 / k_chi))
    m, v, sk, ku = describe(data_t)
    rows.append((f"Стьюдента, k={k_t}", m, 0.0, v, k_t / (k_t - 2.0), sk, ku, 0.0, 6.0 / (k_t - 4.0)))
    m, v, sk, ku = describe(data_f)
    tf_m = d2 / (d2 - 2.0)
    tf_v = 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
    rows.append((f"Фішера, d1={d1}, d2={d2}", m, tf_m, v, tf_v, sk, ku, None, None))

    # ---------- Вивід таблиці ----------
    say("ТАБЛИЦЯ 1. Числові характеристики вибірок (вибіркові / теоретичні значення)")
    say("-" * 100)
    say(f"{'Розподіл':<38}{'Сер.в':>9}{'Сер.т':>9}{'Дисп.в':>10}{'Дисп.т':>10}{'Асим.в':>9}{'Ексц.в':>9}")
    say("-" * 100)
    for r in rows:
        say(f"{r[0]:<38}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>10.4f}{r[4]:>10.4f}{r[5]:>9.4f}{r[6]:>9.4f}")
    say("-" * 100)
    say()
    say("ТАБЛИЦЯ 2. Правило «трьох сигм» (частка значень з |z| < 1, 2, 3 у %)")
    say("-" * 70)
    say(f"{'Алгоритм':<28}{'<1σ':>10}{'<2σ':>10}{'<3σ':>10}")
    for (name, p1, p2, p3) in rule:
        say(f"{name:<28}{p1:>10.2f}{p2:>10.2f}{p3:>10.2f}")
    say(f"{'Теорія':<28}{68.27:>10.2f}{95.45:>10.2f}{99.73:>10.2f}")
    say()

    # ---------- 9. Порівняння швидкості ----------
    NT = 200000
    REP = 3
    say(f"ТАБЛИЦЯ 3. Час генерації {NT} нормальних чисел (найкращий з {REP} запусків)")
    say("-" * 70)
    times = []
    for (name, g, col) in gens:
        best = None
        for _ in range(REP):
            t0 = time.perf_counter()
            for _ in range(NT):
                g()
            dt = time.perf_counter() - t0
            if best is None or dt < best:
                best = dt
        times.append((name, best, col))
    t_zig = times[2][1]
    say(f"{'Алгоритм':<28}{'Час, с':>10}{'мкс/число':>12}{'відносно зиккурату':>22}")
    for (name, t, col) in times:
        say(f"{name:<28}{t:>10.3f}{t / NT * 1e6:>12.3f}{t / t_zig:>20.2f}x")
    say()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([t[0] for t in times], [t[1] for t in times], color=[t[2] for t in times])
    for i, t in enumerate(times):
        ax.text(i, t[1], f"{t[1]:.2f} с", ha="center", va="bottom")
    ax.set_ylabel(f"Час генерації {NT} чисел, с")
    ax.set_title("Порівняння швидкодії алгоритмів")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout(); fig.savefig("figs/rys6_timing.png", dpi=150); plt.close(fig)
    say("Графіки збережено у каталозі figs/")


if __name__ == "__main__":
    main()
