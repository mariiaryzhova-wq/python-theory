# -*- coding: utf-8 -*-
"""
Лабораторна робота №4. Кореляційний аналіз.
Дослідження залежності річного обсягу виробництва Y (млн грн)
від вартості основних фондів X (млн грн) за даними n = 30 підприємств.
Дані генеруються випадково (фіксоване зерно SEED для відтворюваності).
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42          # зерно генератора випадкових чисел
N = 30             # обсяг вибірки
ALPHA = 0.05       # рівень значущості
OUT = "fig"        # папка для рисунків

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------- 1. Дані
def generate_data(n=N, seed=SEED):
    """X ~ N(20; 5^2) - основні фонди, Y = 15 + 0,30 X + випадкова похибка.
    Нормальний закон розподілу - вимога кореляційного аналізу."""
    rng = np.random.default_rng(seed)
    x = np.round(np.clip(rng.normal(20, 5, n), 8, 32), 1)
    y = np.round(15.0 + 0.30 * x + rng.normal(0, 0.9, n), 1)
    return x, y


# ------------------------------------------------- 2. Описова статистика
def variation_label(cv):
    """Оцінка варіації за таблицею 4.1."""
    if cv <= 5:
        return "слабка"
    if cv <= 10:
        return "помірна"
    if cv <= 20:
        return "значна"
    if cv <= 50:
        return "велика"
    return "дуже велика"


def describe(v):
    mean = v.mean()
    s = v.std(ddof=0)                  # S - середнє квадратичне відхилення
    cv = s / mean * 100                # коефіцієнт варіації, %
    return mean, s, cv


# ------------------------------------------- 3. Коефіцієнт кореляції r
def r_formula_41(x, y):
    """Формула (4.1): r = (mean(xy) - mean(x)*mean(y)) / (Sx * Sy)."""
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    return (np.mean(x * y) - x.mean() * y.mean()) / (sx * sy)


def r_sums(x, y):
    """Формула (4.2) для незгрупованих даних (через суми)."""
    n = len(x)
    num = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    den = math.sqrt(n * np.sum(x ** 2) - np.sum(x) ** 2) * \
          math.sqrt(n * np.sum(y ** 2) - np.sum(y) ** 2)
    return num / den


def r_grouped(xm, ym, nij):
    """Формула (4.3) для даних у вигляді кореляційної таблиці (формула 4.2 методички).
    xm, ym - середини інтервалів, nij - матриця частот (рядки - x, стовпці - y)."""
    nij = np.asarray(nij, dtype=float)
    n = nij.sum()
    ni, nj = nij.sum(axis=1), nij.sum(axis=0)
    sxy = sum(xm[i] * ym[j] * nij[i, j]
              for i in range(len(xm)) for j in range(len(ym)))
    sx, sy = np.sum(xm * ni), np.sum(ym * nj)
    sx2, sy2 = np.sum(xm ** 2 * ni), np.sum(ym ** 2 * nj)
    return (n * sxy - sx * sy) / (math.sqrt(n * sx2 - sx ** 2) *
                                  math.sqrt(n * sy2 - sy ** 2))


def strength_label(r):
    """Оцінка щільності зв'язку за шкалою з методичних вказівок."""
    a = abs(r)
    if a < 0.3:
        return "слабкий"
    if a < 0.5:
        return "помірний"
    if a < 0.7:
        return "помітний"
    if a < 0.9:
        return "досить тісний"
    if a < 1.0:
        return "надто тісний"
    return "функціональний (лінійна залежність)"


# ------------------------------------------- 4. Перевірка значущості r
def t_observed(r, n):
    return r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)


def t_critical(alpha, n):
    return stats.t.ppf(1 - alpha / 2, n - 2)       # двостороння область


def significance(r, n, alpha=ALPHA):
    t_obs = t_observed(r, n)
    t_cr = t_critical(alpha, n)
    p = 2 * (1 - stats.t.cdf(abs(t_obs), n - 2))
    return t_obs, t_cr, p, abs(t_obs) > t_cr


def confidence_interval(r, n, alpha=ALPHA):
    """Довірчий інтервал для r за допомогою z-перетворення Фішера."""
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return math.tanh(z - zc * se), math.tanh(z + zc * se)


# ------------------------------------------------ 5. Лінійна регресія
def regression(x, y):
    b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    se = math.sqrt(np.sum(resid ** 2) / (len(x) - 2))
    return a, b, se


def make_edges(v, width):
    """Межі інтервалів групування завширшки width, що охоплюють усі значення."""
    start = math.floor(v.min())
    k = math.floor((v.max() - start) / width) + 1
    return start + width * np.arange(k + 1)


# ------------------------------------------------------ 6. Візуалізація
def plot_series(x, y, path):
    k = np.arange(1, len(x) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    ax[0].plot(k, x, "o-", color="#1f77b4")
    ax[0].axhline(x.mean(), color="gray", ls="--", label=f"середнє = {x.mean():.2f}")
    ax[0].set_ylabel("X, млн грн"); ax[0].set_title("Основні фонди X за підприємствами")
    ax[0].legend(loc="upper right")
    ax[1].plot(k, y, "s-", color="#d62728")
    ax[1].axhline(y.mean(), color="gray", ls="--", label=f"середнє = {y.mean():.2f}")
    ax[1].set_ylabel("Y, млн грн"); ax[1].set_xlabel("Номер підприємства")
    ax[1].set_title("Річний обсяг виробництва Y за підприємствами")
    ax[1].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_hist(x, y, path):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for a_, v, c, name in ((ax[0], x, "#1f77b4", "X"), (ax[1], y, "#d62728", "Y")):
        a_.hist(v, bins=6, color=c, alpha=0.75, edgecolor="white")
        a_.axvline(v.mean(), color="black", ls="--", label=f"середнє = {v.mean():.2f}")
        a_.set_xlabel(f"{name}, млн грн"); a_.set_ylabel("Частота")
        a_.set_title(f"Гістограма {name}"); a_.legend()
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_field(x, y, r, a, b, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=45, color="#1f77b4", edgecolor="white", zorder=3,
               label="Спостереження (xi, yi)")
    xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    ax.plot(xs, a + b * xs, color="#d62728", lw=2,
            label=f"Регресія: ŷ = {a:.3f} + {b:.3f}·x")
    ax.axvline(x.mean(), color="gray", ls=":", lw=1)
    ax.axhline(y.mean(), color="gray", ls=":", lw=1, label="Середні значення")
    # еліпс розсіювання (1,5 сигма) - огинаюча кореляційного поля
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x.mean(), y.mean()), 3 * math.sqrt(vals[1]),
                         3 * math.sqrt(vals[0]), angle=ang, fill=False,
                         ec="#2ca02c", lw=1.5, ls="--",
                         label="Еліпс розсіювання (1,5σ)"))
    ax.set_xlabel("Основні фонди X, млн грн"); ax.set_ylabel("Річний обсяг виробництва Y, млн грн")
    ax.set_title(f"Кореляційне поле (r = {r:.4f})"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_directions(path, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    x = np.round(np.clip(rng.normal(20, 5, N), 8, 32), 1)
    sets = [("а) позитивний", 15 + 0.30 * x + rng.normal(0, 0.9, N)),
            ("б) негативний", 30 - 0.30 * x + rng.normal(0, 0.9, N)),
            ("в) відсутній", rng.uniform(18, 24, N))]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True)
    res = []
    for a_, (name, yy) in zip(ax, sets):
        r = r_formula_41(x, yy)
        res.append((name, r, *significance(r, N)))
        a_.scatter(x, yy, s=25, color="#1f77b4", edgecolor="white")
        b = np.polyfit(x, yy, 1)
        a_.plot(np.sort(x), np.polyval(b, np.sort(x)), color="#d62728")
        a_.set_title(f"{name}\nr = {r:.3f}"); a_.set_xlabel("X")
    ax[0].set_ylabel("Y")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return res


def plot_properties(path):
    x = np.linspace(-10, 10, 41)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    y1 = 2 * x + 1
    ax[0].scatter(x, y1, s=18); ax[0].set_title(f"Y = 2X + 1: r = {r_formula_41(x, y1):.3f}")
    y2 = x ** 2
    ax[1].scatter(x, y2, s=18, color="#d62728")
    ax[1].set_title(f"Y = X²: r = {r_formula_41(x, y2):.3f}")
    for a_ in ax:
        a_.set_xlabel("X"); a_.set_ylabel("Y")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_grouped(xm, ym, nij, r, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, xv in enumerate(xm):
        for j, yv in enumerate(ym):
            if nij[i][j] > 0:
                ax.scatter(xv, yv, s=120 * nij[i][j], color="#1f77b4", alpha=0.6,
                           edgecolor="black")
                ax.text(xv, yv, int(nij[i][j]), ha="center", va="center", fontsize=8)
    ax.set_xlabel("X (середини інтервалів)"); ax.set_ylabel("Y (середини інтервалів)")
    ax.set_title(f"Кореляційне поле за згрупованими даними (r = {r:.4f})")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# ---------------------------------------------------------------- main
def line(c="="):
    print(c * 72)


def main():
    x, y = generate_data()
    n = len(x)

    line(); print("ЛАБОРАТОРНА РОБОТА №4. КОРЕЛЯЦІЙНИЙ АНАЛІЗ"); line()
    print(f"Обсяг вибірки n = {n}; рівень значущості alpha = {ALPHA}; SEED = {SEED}\n")
    print("№   X, млн грн   Y, млн грн")
    for i in range(n):
        print(f"{i + 1:<3d} {x[i]:>9.1f} {y[i]:>12.1f}")

    # --- описова статистика
    print(); line("-"); print("1. ОПИСОВА СТАТИСТИКА"); line("-")
    for name, v in (("X", x), ("Y", y)):
        m, s, cv = describe(v)
        print(f"{name}: середнє = {m:.4f}; S = {s:.4f}; min = {v.min():.1f}; "
              f"max = {v.max():.1f}; Cv = {cv:.2f}% -> варіація {variation_label(cv)}")
        w, p = stats.shapiro(v)
        print(f"   тест Шапіро-Вілка: W = {w:.4f}, p = {p:.4f} -> "
              f"{'розподіл не суперечить нормальному' if p > ALPHA else 'нормальність відхиляється'}")

    # --- коефіцієнт кореляції
    print(); line("-"); print("2. ЛІНІЙНИЙ КОЕФІЦІЄНТ КОРЕЛЯЦІЇ"); line("-")
    r1 = r_formula_41(x, y)
    r2 = r_sums(x, y)
    r3 = np.corrcoef(x, y)[0, 1]
    r4 = stats.pearsonr(x, y)[0]
    print(f"mean(x*y) = {np.mean(x * y):.4f}; mean(x)*mean(y) = {x.mean() * y.mean():.4f}")
    print(f"cov(x,y) = {np.mean(x * y) - x.mean() * y.mean():.4f}")
    print(f"r за формулою (4.1)          = {r1:.6f}")
    print(f"r за формулою (4.2) (суми)   = {r2:.6f}")
    print(f"r за numpy.corrcoef          = {r3:.6f}")
    print(f"r за scipy.stats.pearsonr    = {r4:.6f}")
    r = r1
    print(f"Напрямок зв'язку: {'позитивний (прямий)' if r > 0 else 'негативний (обернений)'}")
    print(f"Щільність зв'язку: {strength_label(r)}")
    print(f"Коефіцієнт детермінації r^2 = {r ** 2:.4f} ({r ** 2 * 100:.2f}% варіації Y пояснюється X)")
    rho, p_rho = stats.spearmanr(x, y)
    print(f"Коефіцієнт рангової кореляції Спірмена = {rho:.4f} (p = {p_rho:.2e})")

    # --- перевірка гіпотези
    print(); line("-"); print("3. ПЕРЕВІРКА ЗНАЧУЩОСТІ КОЕФІЦІЄНТА КОРЕЛЯЦІЇ"); line("-")
    t_obs, t_cr, p, sig = significance(r, n)
    lo, hi = confidence_interval(r, n)
    print("H0: r_ген = 0 (лінійного зв'язку немає); H1: r_ген != 0")
    print(f"Число ступенів свободи k = n - 2 = {n - 2}")
    print(f"t_набл = {t_obs:.4f}; t_кр(двост., alpha={ALPHA}; k={n - 2}) = {t_cr:.4f}; p-value = {p:.3e}")
    if sig:
        print(f"|t_набл| > t_кр -> H0 відхиляємо: r значущий, лінійний кореляційний зв'язок існує")
    else:
        print(f"|t_набл| <= t_кр -> H0 не відхиляємо: r незначущий")
    print(f"95% довірчий інтервал для r_ген (z-перетворення Фішера): [{lo:.4f}; {hi:.4f}]")

    # --- регресія
    print(); line("-"); print("4. ЛІНІЙНА РЕГРЕСІЙНА МОДЕЛЬ Y = a + b*X"); line("-")
    a, b, se = regression(x, y)
    print(f"b = r*Sy/Sx = {r * y.std() / x.std():.4f} (за МНК: {b:.4f})")
    print(f"a = mean(y) - b*mean(x) = {a:.4f}")
    print(f"Рівняння регресії: y = {a:.4f} + {b:.4f} * x")
    print(f"Стандартна похибка оцінювання = {se:.4f}")
    for xv in (15, 20, 25):
        print(f"   прогноз при X = {xv}: Y = {a + b * xv:.3f}")

    # --- згруповані дані
    print(); line("-"); print("5. КОРЕЛЯЦІЙНА ТАБЛИЦЯ (ЗГРУПОВАНІ ДАНІ) І ФОРМУЛА (4.3)"); line("-")
    xb = make_edges(x, 2.5)                                 # інтервали X завширшки 2,5
    yb = make_edges(y, 0.8)                                 # інтервали Y завширшки 0,8
    nij, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    xm, ym = (xb[:-1] + xb[1:]) / 2, (yb[:-1] + yb[1:]) / 2
    print("X \\ Y        " + " ".join(f"{yb[j]:.1f}-{yb[j + 1]:.1f}" for j in range(len(ym))) + "    n_i")
    for i in range(len(xm)):
        print(f"{xb[i]:>5.1f}-{xb[i + 1]:<5.1f}  " +
              " ".join(f"{int(nij[i, j]):>9d}" for j in range(len(ym))) + f"   {int(nij[i].sum()):>4d}")
    print("n_j          " + " ".join(f"{int(nij[:, j].sum()):>9d}" for j in range(len(ym))) + f"   {int(nij.sum()):>4d}")
    rg = r_grouped(xm, ym, nij)
    print(f"r (згруповані дані, формула 4.3) = {rg:.4f}; r (вихідні дані) = {r:.4f}; різниця = {abs(rg - r):.4f}")

    # --- перевірка на прикладі з методичних вказівок
    print(); line("-"); print("6. ПЕРЕВІРКА ПРОГРАМИ НА ПРИКЛАДІ З МЕТОДИЧНИХ ВКАЗІВОК (табл. 4.2)"); line("-")
    ex_x = np.array([12.5, 17.5, 22.5, 27.5])
    ex_y = np.array([20.5, 21.5, 22.5, 23.5, 24.5])
    ex_n = np.array([[1, 0, 0, 0, 0],
                     [0, 2, 1, 0, 0],
                     [0, 0, 2, 3, 0],
                     [0, 0, 0, 3, 8]])
    r_ex = r_grouped(ex_x, ex_y, ex_n)
    t_ex, tc_ex, p_ex, sig_ex = significance(r_ex, 20)
    print(f"r = {r_ex:.4f}          (у методичці: 0,924)")
    print(f"t_набл = {t_ex:.2f}     (у методичці: 10,25)")
    print(f"t_кр = {tc_ex:.2f}       (у методичці: 2,1)")
    verdict = "зв'язок значущий" if sig_ex else "зв'язок незначущий"
    print(f"Висновок: {verdict}; щільність - {strength_label(r_ex)}")

    # --- три випадки напрямку
    print(); line("-"); print("7. ПОРІВНЯННЯ НАПРЯМКІВ ЗВ'ЯЗКУ (додаткові згенеровані вибірки)"); line("-")
    res = plot_directions(f"{OUT}/fig_directions.png")
    for name, rr, tt, tc, pp, ss in res:
        print(f"{name:<14s} r = {rr:>7.4f}; t_набл = {tt:>7.3f}; t_кр = {tc:.3f}; p = {pp:.3e}; "
              f"{'значущий' if ss else 'незначущий'}")

    # --- властивості r
    print(); line("-"); print("8. ПЕРЕВІРКА ВЛАСТИВОСТЕЙ КОЕФІЦІЄНТА КОРЕЛЯЦІЇ"); line("-")
    print(f"r(X, Y)            = {r_formula_41(x, y):.6f}")
    print(f"r(10*X, 3*Y)       = {r_formula_41(10 * x, 3 * y):.6f}   (масштаб не впливає на r)")
    print(f"r(X+100, Y-50)     = {r_formula_41(x + 100, y - 50):.6f}   (зсув не впливає на r)")
    print(f"r(-X, Y)           = {r_formula_41(-x, y):.6f}   (зміна знака X змінює знак r)")
    xt = np.linspace(-10, 10, 41)
    print(f"r(X, 2X+1)         = {r_formula_41(xt, 2 * xt + 1):.6f}   (|r| = 1: лінійна залежність)")
    print(f"r(X, X^2), X симетр.= {r_formula_41(xt, xt ** 2):.6f}   (зв'язок нелінійний, але r = 0)")

    # --- графіки
    plot_series(x, y, f"{OUT}/fig_series.png")
    plot_hist(x, y, f"{OUT}/fig_hist.png")
    plot_field(x, y, r, a, b, f"{OUT}/fig_field.png")
    plot_properties(f"{OUT}/fig_properties.png")
    plot_grouped(xm, ym, nij, rg, f"{OUT}/fig_grouped.png")
    print(); line()
    print("Рисунки збережено в папку:", OUT)
    print("ВИСНОВОК: між X та Y існує "
          f"{'прямий' if r > 0 else 'обернений'} {strength_label(r)}")
    print(f"лінійний кореляційний зв'язок (r = {r:.3f}), який є "
          f"{'статистично значущим' if sig else 'статистично незначущим'}.")
    line()


if __name__ == "__main__":
    main()
