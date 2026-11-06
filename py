# -*- coding: utf-8 -*-
"""
Лабораторна робота №1. Описові статистики.
Запуск:  python lab1.py  
"""
import json
import math   # використовується ЛИШЕ для перевірки результатів
import os
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT = os.environ.get("OUT_DIR", "out")
os.makedirs(OUT, exist_ok=True)

LOG = {}          # текстові виводи програми (для звіту)
_current = None


def section(name):
    global _current
    _current = name
    LOG[name] = []


def out(text=""):
    print(text)
    LOG[_current].append(text)


# ======================================================================
# ЧАСТИНА 1. ВЛАСНІ РЕАЛІЗАЦІЇ ОПИСОВИХ СТАТИСТИК
# ======================================================================

def count(x):
    """Кількість елементів (лічильник у циклі замість len)."""
    n = 0
    for _ in x:
        n += 1
    return n


def total(x):
    """Сума елементів (накопичення в циклі замість sum)."""
    s = 0
    for v in x:
        s += v
    return s


def absolute(v):
    """Модуль числа без abs()."""
    return -v if v < 0 else v


def square_root(a):
    """Квадратний корінь методом Ньютона (замість math.sqrt)."""
    if a == 0:
        return 0.0
    r = a if a >= 1 else 1.0
    for _ in range(100):
        nxt = (r + a / r) / 2
        if absolute(nxt - r) < 1e-15 * nxt:
            return nxt
        r = nxt
    return r


def floor_int(h):
    """Ціла частина невід'ємного числа без math.floor."""
    k = 0
    while k + 1 <= h:
        k += 1
    return k


def sort_values(x):
    """Сортування злиттям (merge sort) замість sorted()."""
    n = count(x)
    if n <= 1:
        return list(x)
    mid = n // 2
    left = sort_values(x[:mid])
    right = sort_values(x[mid:])
    res = []
    i = j = 0
    nl, nr = count(left), count(right)
    while i < nl and j < nr:
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    while i < nl:
        res.append(left[i])
        i += 1
    while j < nr:
        res.append(right[j])
        j += 1
    return res


def mean(x):
    """Вибіркове середнє, формула (1.1)."""
    return total(x) / count(x)


def sum_squares(x, c):
    """Сума квадратів відхилень від точки c."""
    s = 0
    for v in x:
        d = v - c
        s += d * d
    return s


def sum_abs(x, c):
    """Сума модулів відхилень від точки c."""
    s = 0
    for v in x:
        s += absolute(v - c)
    return s


def variance(x):
    """Вибіркова дисперсія з поправкою Бесселя (n - 1), формула (1.3)."""
    return sum_squares(x, mean(x)) / (count(x) - 1)


def std(x):
    """Стандартне відхилення = корінь з вибіркової дисперсії."""
    return square_root(variance(x))


def median(x):
    """Медіана: середній елемент варіаційного ряду (або півсума двох середніх)."""
    s = sort_values(x)
    n = count(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def frequencies(x):
    """Частоти значень (словник значення -> кількість)."""
    freq = {}
    for v in x:
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
    return freq


def modes(x):
    """Список мод. Якщо всі значення унікальні, моди немає (порожній список)."""
    freq = frequencies(x)
    top = 0
    for c in freq.values():
        if c > top:
            top = c
    if top == 1:
        return []
    res = []
    for v, c in freq.items():
        if c == top:
            res.append(v)
    return sort_values(res)


def minimum(x):
    r = x[0]
    for v in x:
        if v < r:
            r = v
    return r


def maximum(x):
    r = x[0]
    for v in x:
        if v > r:
            r = v
    return r


def value_range(x):
    """Розмах = max - min."""
    return maximum(x) - minimum(x)


def quantile(x, p):
    """Квантиль рівня p; лінійна інтерполяція між сусідніми порядковими
    статистиками (тип 7 за Hyndman–Fan, використовується в numpy та pandas)."""
    s = sort_values(x)
    n = count(s)
    h = (n - 1) * p
    lo = floor_int(h)
    hi = lo + 1 if lo + 1 < n else n - 1
    return s[lo] + (h - lo) * (s[hi] - s[lo])


QUANTILES = (0.1, 0.25, 0.5, 0.75)


def describe(x):
    """Повний набір описових статистик з завдання."""
    res = {
        "n": count(x),
        "mean": mean(x),
        "variance": variance(x),
        "std": std(x),
        "median": median(x),
        "mode": modes(x),
        "min": minimum(x),
        "max": maximum(x),
        "range": value_range(x),
    }
    for p in QUANTILES:
        res["q%g" % p] = quantile(x, p)
    return res


def show(name, x):
    d = describe(x)
    out("Вибірка %s: %s" % (name, x))
    out("  n                     = %d" % d["n"])
    out("  вибіркове середнє     = %.4f" % d["mean"])
    out("  вибіркова дисперсія   = %.4f" % d["variance"])
    out("  стандартне відхилення = %.4f" % d["std"])
    out("  медіана               = %.4f" % d["median"])
    out("  мода                  = %s" % (d["mode"] if d["mode"] else "відсутня (усі значення унікальні)"))
    out("  мінімум / максимум    = %s / %s" % (d["min"], d["max"]))
    out("  розмах                = %s" % d["range"])
    for p in QUANTILES:
        out("  квантиль %-5g       = %.4f" % (p, d["q%g" % p]))
    return d


section("basic")
A = [100, 102, 81, 105, 96]                       # доходи фірми, табл. 1.1
B = [12, 15, 11, 15, 18, 20, 12, 14, 15, 12]      # парна вибірка з двома модами
out("=== Вхідні вибірки ===")
dA = show("A (доходи фірми, млн)", A)
out("")
dB = show("B (n парне, дві моди)", B)
out("")

# приклад з теоретичної частини: час підйому 7:00, 8:30, 6:15
wake = [7 * 60, 8 * 60 + 30, 6 * 60 + 15]
mw = mean(wake)
out("=== Приклад із теорії: час підйому ===")
out("  середній час = %d хв = %d год %02d хв" % (mw, mw // 60, mw % 60))
out("")

# властивості середнього та медіани (точна арифметика Fraction)
out("=== Властивості вибіркового середнього (вибірка A) ===")
FA = [Fraction(v) for v in A]
FM = total(FA) / count(FA)
dev = [FM - v for v in FA]
out("  M = %s" % float(FM))
out("  відхилення (M - X_i) = %s" % [float(d) for d in dev])
out("  сума відхилень      = %s" % total(dev))
out("")
out("  c      сума квадратів відхилень   сума модулів відхилень")
cands = [("M-2", FM - 2), ("M-1", FM - 1), ("M", FM), ("M+1", FM + 1), ("M+2", FM + 2),
         ("Me", Fraction(median(A)))]
rows_prop = []
for lab, c in cands:
    sq = sum_squares(FA, c)
    ab = sum_abs(FA, c)
    rows_prop.append((lab, float(c), float(sq), float(ab)))
    out("  %-3s=%6.1f   %24.2f   %20.2f" % (lab, float(c), float(sq), float(ab)))
out("  Мінімум суми квадратів – при c = M; мінімум суми модулів – при c = Me.")

# графік властивостей
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cs = [75 + i * 0.25 for i in range(0, 161)]
sq = [sum_squares(A, c) for c in cs]
ab = [sum_abs(A, c) for c in cs]
ax[0].plot(cs, sq, color="#3b6ea5")
ax[0].axvline(dA["mean"], color="#c0392b", ls="--", label="M = %.1f" % dA["mean"])
ax[0].set_title("Сума квадратів відхилень S(c)")
ax[0].set_xlabel("c"); ax[0].set_ylabel("S(c)"); ax[0].legend()
ax[1].plot(cs, ab, color="#3b6ea5")
ax[1].axvline(dA["median"], color="#2e8b57", ls="--", label="Me = %.0f" % dA["median"])
ax[1].set_title("Сума модулів відхилень D(c)")
ax[1].set_xlabel("c"); ax[1].set_ylabel("D(c)"); ax[1].legend()
for a in ax:
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_properties.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# перевірка власних функцій еталонними бібліотеками
# ---------------------------------------------------------------------
section("verify")
import statistics
import numpy as np

def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)   # лише для перевірки

out("=== Звірка з statistics / numpy / pandas ===")
all_ok = True
checks = []
for name, x in (("A", A), ("B", B)):
    s = pd.Series(x, dtype=float)
    pairs = [
        ("середнє", mean(x), statistics.mean(x), float(np.mean(x))),
        ("дисперсія", variance(x), statistics.variance(x), float(np.var(x, ddof=1))),
        ("ст. відхилення", std(x), statistics.stdev(x), float(np.std(x, ddof=1))),
        ("медіана", median(x), statistics.median(x), float(np.median(x))),
    ]
    for p in QUANTILES:
        pairs.append(("квантиль %g" % p, quantile(x, p),
                      float(s.quantile(p)), float(np.quantile(x, p))))
    for lab, mine, ref1, ref2 in pairs:
        ok = close(mine, ref1) and close(mine, ref2)
        all_ok &= ok
        checks.append((name, lab, mine, ref1, ref2, ok))
        out("  [%s] %-15s власне=%-10.4f бібліотеки=%-10.4f %-10.4f  %s"
            % (name, lab, mine, ref1, ref2, "OK" if ok else "РОЗБІЖНІСТЬ"))
    mm = sorted(statistics.multimode(x))
    mine = modes(x)
    ok = (mine == mm) or (not mine and len(mm) == len(set(x)))
    all_ok &= ok
    out("  [%s] %-15s власне=%s бібліотека=%s  %s" % (name, "мода", mine or "—", mm if mine else "усі унікальні", "OK" if ok else "РОЗБІЖНІСТЬ"))
out("Підсумок: %s" % ("усі значення збігаються" if all_ok else "є розбіжності"))

# ======================================================================
# ЧАСТИНА 2. НАБІР ДАНИХ TMDB 5000
# ======================================================================
movies = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_movies.csv"))
credits = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_credits.csv"))


def names(cell, key="name"):
    """JSON-рядок -> список значень поля key."""
    return [d[key] for d in json.loads(cell)] if isinstance(cell, str) else []


section("tmdb_overview")
out("=== Структура набору даних ===")
out("tmdb_5000_movies.csv : %d рядків, %d стовпців" % movies.shape)
out("tmdb_5000_credits.csv: %d рядків, %d стовпців" % credits.shape)
out("id збігаються з movie_id: %s; дублікатів id: %d"
    % ((movies["id"].values == credits["movie_id"].values).all(), movies["id"].duplicated().sum()))
out("Пропуски: " + ", ".join("%s=%d" % (k, v) for k, v in movies.isna().sum().items() if v))
out("Нульові значення: " + ", ".join("%s=%d" % (c, (movies[c] == 0).sum())
                                     for c in ("budget", "revenue", "runtime", "vote_average", "vote_count")))

movies["release_year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
movies["release_month"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.month

NUM = ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]


def clean(col):
    """Значення без пропусків; нулі в budget/revenue/runtime/vote_average
    трактуються як «невідомо» і відкидаються."""
    s = movies[col].dropna()
    if col in ("budget", "revenue", "runtime"):
        s = s[s > 0]
    if col == "vote_average":
        s = s[movies.loc[s.index, "vote_count"] > 0]
    return [float(v) for v in s]


stats_raw = {c: describe([float(v) for v in movies[c].dropna()]) for c in NUM}
stats_raw["release_year"] = describe([float(v) for v in movies["release_year"].dropna()])
stats_clean = {c: describe(clean(c)) for c in ("budget", "revenue", "runtime", "vote_average")}

section("tmdb_stats")
out("=== Описові статистики (власні функції), усі значення ===")
hdr = "%-14s" % "показник" + "".join("%16s" % c for c in NUM)
out(hdr)
for key in ("n", "mean", "variance", "std", "median", "min", "max", "range", "q0.1", "q0.25", "q0.5", "q0.75"):
    out("%-14s" % key + "".join("%16.4g" % stats_raw[c][key] for c in NUM))
out("%-14s" % "mode" + "".join("%16s" % (("%.4g" % stats_raw[c]["mode"][0]) + ("+%d" % (len(stats_raw[c]["mode"]) - 1) if len(stats_raw[c]["mode"]) > 1 else "") if stats_raw[c]["mode"] else "—") for c in NUM))

# ---------- гістограми числових ознак ----------
BLUE, GREY = "#3b6ea5", "#9aa5b1"
LAB = {"budget": "Бюджет, USD", "revenue": "Збори, USD", "runtime": "Тривалість, хв",
       "popularity": "Популярність (TMDB)", "vote_average": "Середня оцінка", "vote_count": "Кількість голосів"}
TTL = {"budget": "Бюджет (>0)", "revenue": "Збори (>0)", "runtime": "Тривалість (>0)",
       "popularity": "Популярність", "vote_average": "Середня оцінка (v>0)", "vote_count": "Кількість голосів"}


def hist_panel(ax, vals, col, d, logx=False, bins=40):
    if logx:
        vv = [math.log10(v) for v in vals if v > 0]
        ax.hist(vv, bins=bins, color=BLUE, edgecolor="white", linewidth=.4)
        ax.axvline(math.log10(d["mean"]), color="#c0392b", ls="--", lw=1.3, label="середнє")
        ax.axvline(math.log10(d["median"]), color="#2e8b57", ls="-.", lw=1.3, label="медіана")
        ax.set_xlabel("log10(%s)" % LAB[col])
    else:
        ax.hist(vals, bins=bins, color=BLUE, edgecolor="white", linewidth=.4)
        ax.axvline(d["mean"], color="#c0392b", ls="--", lw=1.3, label="середнє")
        ax.axvline(d["median"], color="#2e8b57", ls="-.", lw=1.3, label="медіана")
        ax.set_xlabel(LAB[col])
    ax.set_title(TTL[col], fontsize=11)
    ax.set_ylabel("Кількість фільмів")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
for ax, col in zip(axes.flat, NUM):
    vals = clean(col) if col in stats_clean else [float(v) for v in movies[col].dropna()]
    d = stats_clean.get(col, stats_raw[col])
    hist_panel(ax, vals, col, d)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hist_linear.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
for ax, col in zip(axes.flat, ["budget", "revenue", "popularity", "vote_count"]):
    vals = clean(col) if col in stats_clean else [float(v) for v in movies[col].dropna() if v > 0]
    d = stats_clean.get(col, stats_raw[col])
    hist_panel(ax, vals, col, d, logx=True, bins=35)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hist_log.png"), dpi=150)
plt.close(fig)

# рік і місяць виходу
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
yrs = [float(v) for v in movies["release_year"].dropna()]
ax[0].hist(yrs, bins=range(1915, 2020, 5), color=BLUE, edgecolor="white")
ax[0].axvline(stats_raw["release_year"]["mean"], color="#c0392b", ls="--", label="середнє")
ax[0].axvline(stats_raw["release_year"]["median"], color="#2e8b57", ls="-.", label="медіана")
ax[0].set_title("Рік виходу (інтервали по 5 років)", fontsize=11)
ax[0].set_xlabel("Рік"); ax[0].set_ylabel("Кількість фільмів"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
mf = frequencies([int(v) for v in movies["release_month"].dropna()])
mn = ["січ", "лют", "бер", "кві", "тра", "чер", "лип", "сер", "вер", "жов", "лис", "гру"]
ax[1].bar(mn, [mf.get(i, 0) for i in range(1, 13)], color=BLUE)
ax[1].set_title("Місяць виходу", fontsize=11); ax[1].set_ylabel("Кількість фільмів"); ax[1].grid(alpha=.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_year_month.png"), dpi=150)
plt.close(fig)

# ---------- частоти нечислових ознак ----------
section("tmdb_freq")


def freq_sorted(items):
    f = frequencies(items)
    return sorted(f.items(), key=lambda kv: (-kv[1], str(kv[0])))


LANG_EN = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
           "ru": "Russian", "ja": "Japanese", "zh": "Chinese", "pt": "Portuguese", "hi": "Hindi",
           "ar": "Arabic", "ko": "Korean", "la": "Latin", "sv": "Swedish", "pl": "Polish"}
genres_lists = [names(c) for c in movies["genres"]]
freqs = {
    "original_language": freq_sorted(list(movies["original_language"])),
    "status": freq_sorted(list(movies["status"])),
    "genres": freq_sorted([g for l in genres_lists for g in l]),
    "production_countries": freq_sorted([g for c in movies["production_countries"] for g in names(c)]),
    "spoken_languages": freq_sorted([LANG_EN.get(g, g) for c in movies["spoken_languages"] for g in names(c, "iso_639_1")]),
    "production_companies": freq_sorted([g for c in movies["production_companies"] for g in names(c)]),
    "keywords": freq_sorted([g for c in movies["keywords"] for g in names(c)]),
    "actors": freq_sorted([g for c in credits["cast"] for g in set(names(c))]),
    "directors": freq_sorted([d["name"] for c in credits["crew"] for d in json.loads(c) if d["job"] == "Director"]),
}
for k, v in freqs.items():
    out("%-22s унікальних значень: %5d; топ-5: %s" % (k, len(v), ", ".join("%s (%d)" % kv for kv in v[:5])))


def barh(ax, pairs, title, top=10, color=BLUE):
    pairs = pairs[:top][::-1]
    ax.barh([str(p[0]) for p in pairs], [p[1] for p in pairs], color=color)
    for i, p in enumerate(pairs):
        ax.text(p[1], i, " %d" % p[1], va="center", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Кількість фільмів")
    ax.grid(alpha=.25, axis="x")
    ax.margins(x=.12)
    ax.tick_params(axis="y", labelsize=8)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
barh(axes[0, 0], freqs["original_language"], "Оригінальна мова (код), топ-10")
barh(axes[0, 1], freqs["status"], "Статус фільму", top=3)
barh(axes[1, 0], freqs["genres"], "Жанри (усі 20)", top=20)
barh(axes[1, 1], freqs["production_countries"], "Країни виробництва, топ-10")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cat1.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
barh(axes[0, 0], freqs["spoken_languages"], "Мови, якими розмовляють у фільмі, топ-10")
barh(axes[0, 1], freqs["production_companies"], "Кінокомпанії, топ-15", top=15)
barh(axes[1, 0], freqs["keywords"], "Ключові слова, топ-15", top=15)
barh(axes[1, 1], freqs["directors"], "Режисери, топ-15", top=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cat2.png"), dpi=150)
plt.close(fig)

# ---------- зважений рейтинг ----------
section("tmdb_wr")
vc_all = [float(v) for v in movies["vote_count"]]
va_all = [float(v) for v in movies["vote_average"]]
m_min = quantile(vc_all, 0.90)          # поріг m
C_mean = mean(va_all)
C_med = median(va_all)
C_rated = mean([float(r) for r, v in zip(movies["vote_average"], movies["vote_count"]) if v > 0])
out("m (90-й перцентиль кількості голосів) = %.1f" % m_min)
out("C (середнє vote_average)              = %.4f" % C_mean)
out("C (медіана vote_average)              = %.4f" % C_med)
out("Довідково: C без фільмів з v = 0      = %.4f" % C_rated)
out("Фільмів з v >= m: %d" % count([v for v in vc_all if v >= m_min]))


def wr(v, R, C, m=m_min):
    """Зважений рейтинг (формула 1.5)."""
    return v / (v + m) * R + m / (v + m) * C


movies["wr_mean"] = [wr(v, R, C_mean) for v, R in zip(movies["vote_count"], movies["vote_average"])]
movies["wr_med"] = [wr(v, R, C_med) for v, R in zip(movies["vote_count"], movies["vote_average"])]
movies["wr_rated"] = [wr(v, R, C_rated) for v, R in zip(movies["vote_count"], movies["vote_average"])]
movies["genre_list"] = genres_lists


def top_n(df, col, n):
    return df.sort_values([col, "vote_count"], ascending=[False, False]).head(n)


top10_mean = top_n(movies, "wr_mean", 10)
top10_med = top_n(movies, "wr_med", 10)
top10_rated = top_n(movies, "wr_rated", 10)
qualified = movies[movies["vote_count"] >= m_min]
out("")
out("Top-10 (C = середнє):")
for i, (_, r) in enumerate(top10_mean.iterrows(), 1):
    out("%2d. %-45s v=%5d  R=%.1f  WR=%.4f" % (i, r["title"], r["vote_count"], r["vote_average"], r["wr_mean"]))
out("")
out("Top-10 (C = медіана):")
for i, (_, r) in enumerate(top10_med.iterrows(), 1):
    out("%2d. %-45s v=%5d  R=%.1f  WR=%.4f" % (i, r["title"], r["vote_count"], r["vote_average"], r["wr_med"]))
out("")
out("Той самий Top-10 з фільтром v >= m: %s" % (list(top_n(qualified, "wr_mean", 10)["title"]) == list(top10_mean["title"])))
out("Top-10 за C без нерейтингованих збігається: %s" % (list(top10_rated["title"]) == list(top10_mean["title"])))

# Top-5 за жанрами
all_genres = [g for g, _ in freqs["genres"]]
genre_top = {}
for g in sorted(all_genres):
    sub = movies[movies["genre_list"].apply(lambda l: g in l)]
    genre_top[g] = {
        "n": len(sub),
        "n_q": int((sub["vote_count"] >= m_min).sum()),
        "mean": top_n(sub, "wr_mean", 5),
        "med": top_n(sub, "wr_med", 5),
    }
out("")
out("Top-5 за жанрами (C = середнє) – перший фільм кожного жанру:")
for g, d in genre_top.items():
    r = d["mean"].iloc[0]
    out("  %-16s фільмів: %4d (v>=m: %3d)  №1: %s (WR=%.3f)" % (g, d["n"], d["n_q"], r["title"], r["wr_mean"]))

changed = []
for g, d in genre_top.items():
    a, b = list(d["mean"]["title"]), list(d["med"]["title"])
    if a != b:
        changed.append((g, a, b))
out("")
out("Жанрів, де Top-5 змінився при C = медіана: %d з %d" % (len(changed), len(genre_top)))

# графік WR
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].scatter(movies["vote_count"], movies["vote_average"], s=6, color=GREY, alpha=.5, label="усі фільми")
ax[0].scatter(top10_mean["vote_count"], top10_mean["vote_average"], s=28, color="#c0392b", label="Top-10 за WR")
ax[0].axhline(C_mean, color=BLUE, ls="--", lw=1, label="C = %.2f" % C_mean)
ax[0].axvline(m_min, color="#2e8b57", ls="-.", lw=1, label="m = %.0f" % m_min)
ax[0].set_xscale("log")
ax[0].set_xlabel("Кількість голосів v (логарифмічна шкала)")
ax[0].set_ylabel("Середня оцінка R")
ax[0].set_title("Оцінка та кількість голосів", fontsize=11)
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=.25)
t = top10_mean.iloc[::-1]
ax[1].barh(t["title"], t["wr_mean"], color=BLUE)
for i, w in enumerate(t["wr_mean"]):
    ax[1].text(w, i, " %.3f" % w, va="center", fontsize=8)
ax[1].set_xlim(7.0, 8.5)
ax[1].set_title("Top-10 за зваженим рейтингом (C = середнє)", fontsize=11)
ax[1].set_xlabel("WR"); ax[1].tick_params(axis="y", labelsize=8); ax[1].grid(alpha=.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_wr.png"), dpi=150)
plt.close(fig)

# ---------- збереження результатів для звіту ----------
import pickle
with open(os.path.join(OUT, "results.pkl"), "wb") as fh:
    pickle.dump({
        "LOG": LOG, "dA": dA, "dB": dB, "rows_prop": rows_prop, "checks": checks,
        "stats_raw": stats_raw, "stats_clean": stats_clean, "freqs": freqs,
        "m": m_min, "C_mean": C_mean, "C_med": C_med, "C_rated": C_rated,
        "top10_mean": top10_mean[["title", "vote_count", "vote_average", "wr_mean"]],
        "top10_med": top10_med[["title", "vote_count", "vote_average", "wr_med"]],
        "genre_top": {g: {"n": d["n"], "n_q": d["n_q"],
                          "mean": d["mean"][["title", "vote_count", "vote_average", "wr_mean"]],
                          "med": d["med"][["title", "vote_count", "vote_average", "wr_med"]]}
                      for g, d in genre_top.items()},
        "changed": changed,
        "n_qualified": int(len(qualified)),
        "zeros": {c: int((movies[c] == 0).sum()) for c in ("budget", "revenue", "runtime", "vote_average", "vote_count", "popularity")},
        "nan_runtime": int(movies["runtime"].isna().sum()),
    }, fh)
for k, v in LOG.items():
    with open(os.path.join(OUT, k + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(v))
