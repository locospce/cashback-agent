import numpy as np
import pandas as pd


COLUMN_KEYWORDS = {
    "amount":   ["summa", "сумма", "amount", "sum", "price"],
    "cashback": ["cashback", "кэшбэк", "кэшбек", "кешбэк", "кешбек", "cb", "cashbackkb"],
    "client":   ["clnt", "client", "клиент", "customer"],
    "date":     ["date", "дата", "opdate"],
    "city":     ["city", "город", "mrccity"],
}

SYSTEM_PROMPT = (
    "Ты аналитик кэшбэк-программы банка. "
    "Если данные по одной кампании — напиши вывод (3–4 предложения): что работало хорошо, "
    "на что обратить внимание, есть ли что-то необычное. "
    "Если данных несколько — сравни кампании: что изменилось, есть ли тренд, "
    "какая кампания эффективнее и почему. "
    "Отвечай по-русски, без заголовков и маркеров."
)

_SEG_TOL = 3  # допуск ±3% при определении сегмента


def detect_params(df: pd.DataFrame) -> dict:
    """Автоматически определяет new_pct, old_pct и cap из данных транзакций."""
    col_amount   = find_col(df, COLUMN_KEYWORDS["amount"])
    col_cashback = find_col(df, COLUMN_KEYWORDS["cashback"])

    defaults = {"new_pct": 15, "old_pct": 10, "cap": 4500}
    if not col_amount or not col_cashback:
        return defaults

    clean = df[(df[col_amount] > 0) & (df[col_cashback] > 0)]
    if len(clean) < 10:
        return defaults

    # ── Проценты: ищем пики в распределении cashback/amount ──────────────────
    pct     = (clean[col_cashback] / clean[col_amount] * 100).clip(1, 50)
    pct_int = pct.round(0).astype(int)
    counts  = pct_int.value_counts()

    min_count = max(5, len(clean) * 0.06)
    valid = counts[(counts.index >= 3) & (counts.index <= 30) & (counts >= min_count)]

    # Группируем близкие пики (±2%) — берём самый частый из группы
    peaks: list[int] = []
    for rate in valid.sort_values(ascending=False).index.tolist():
        rate = int(rate)
        if not peaks or all(abs(rate - p) > 2 for p in peaks):
            peaks.append(rate)
    peaks.sort()

    if len(peaks) >= 2:
        old_pct, new_pct = peaks[0], peaks[-1]
    elif len(peaks) == 1:
        r = peaks[0]
        if r > 12:          # скорее всего новые клиенты (высокая ставка)
            new_pct = r
            old_pct = max(3, r - 5)
        else:               # скорее всего действующие клиенты (низкая ставка)
            old_pct = r
            new_pct = r + 5
    else:
        old_pct, new_pct = defaults["old_pct"], defaults["new_pct"]

    # ── Кэп: ищем кластер в верхнем хвосте распределения кэшбэка ────────────
    cb  = clean[col_cashback]
    p90 = cb.quantile(0.90)
    top = cb[cb >= p90 * 0.95]
    top_rounded = (top / 50).round() * 50
    top_counts  = top_rounded.value_counts()

    if len(top_counts) > 0 and top_counts.iloc[0] >= max(3, len(clean) * 0.02):
        cap = int(round(top_counts.index[0] / 500) * 500)
    else:
        p99 = cb.quantile(0.99)
        cap = int(np.ceil(p99 / 500) * 500)

    return {"new_pct": int(new_pct), "old_pct": int(old_pct), "cap": max(500, cap)}


def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        col_clean = col.lower().replace("_", "").replace(" ", "")
        for kw in keywords:
            if kw in col_clean:
                return col
    return None


def _build_prompt(metrics: dict, params: dict) -> str:
    m = metrics
    p = params

    seg = p.get("segment_type", "Оба сегмента")
    lines = [
        f"Кампания: {p['name']} (тип: {seg})",
        f"Настройки: новые клиенты {p['new_pct']}%, действующие {p['old_pct']}%, "
        f"кэп {p['cap']} руб ({p['cap_type']})",
        "",
        f"Транзакций: {m['total_transactions']}",
        f"Оборот: {m['total_amount']:,.0f} руб",
        f"Кэшбек выплачен: {m['total_cashback']:,.0f} руб",
        f"Нагрузка кэшбэка на оборот: {m['cashback_load_pct']:.1f}%",
    ]

    if m["unique_clients"] is not None:
        lines.append(f"Уникальных клиентов: {m['unique_clients']}")

    lines += [
        "",
        f"Новые клиенты ({p['new_pct']}%): {m['new_transactions']} транзакций, "
        f"оборот {m['new_amount']:,.0f} руб, средний чек {m['new_avg_check']:,.0f} руб",
        f"Действующие ({p['old_pct']}%): {m['old_transactions']} транзакций, "
        f"оборот {m['old_amount']:,.0f} руб, средний чек {m['old_avg_check']:,.0f} руб",
        f"Кэпнутых транзакций: {m['capped_count']} ({m['capped_pct']:.1f}%)",
    ]

    if m["top_cities"]:
        lines.append("\nТоп-3 города по обороту:")
        for city, amount, cashback in m["top_cities"]:
            lines.append(f"  {city}: {amount:,.0f} руб (кэшбэк {cashback:,.0f} руб)")

    if m["days_active"] is not None:
        lines.append(f"\nАктивных дней: {m['days_active']}")
    if m["best_day_amount"] is not None:
        lines.append(
            f"Лучший день: {m['best_day_date']} — {m['best_day_amount']:,.0f} руб оборота"
        )

    return "\n".join(lines)


def _calc_metrics(df: pd.DataFrame, params: dict) -> dict:
    col_amount   = find_col(df, COLUMN_KEYWORDS["amount"])
    col_cashback = find_col(df, COLUMN_KEYWORDS["cashback"])
    col_client   = find_col(df, COLUMN_KEYWORDS["client"])
    col_date     = find_col(df, COLUMN_KEYWORDS["date"])
    col_city     = find_col(df, COLUMN_KEYWORDS["city"])

    if not col_amount or not col_cashback:
        raise ValueError(
            f"Не найдены обязательные колонки (сумма/кэшбэк). "
            f"Колонки в файле: {list(df.columns)}"
        )

    total_amount   = float(df[col_amount].sum())
    total_cashback = float(df[col_cashback].sum())
    total          = len(df)

    df = df.copy()
    df["_pct"] = df[col_cashback] / df[col_amount] * 100

    new_pct, old_pct, cap = params["new_pct"], params["old_pct"], params["cap"]

    df_new    = df[abs(df["_pct"] - new_pct) <= _SEG_TOL]
    df_old    = df[abs(df["_pct"] - old_pct) <= _SEG_TOL]
    df_capped = df[df[col_cashback] >= cap * 0.99]

    new_amount = float(df_new[col_amount].sum())
    old_amount = float(df_old[col_amount].sum())

    top_cities = None
    if col_city:
        city_agg = (
            df.groupby(col_city)
            .agg(**{col_amount: (col_amount, "sum"), col_cashback: (col_cashback, "sum")})
            .sort_values(col_amount, ascending=False)
            .head(3)
            .reset_index()
        )
        top_cities = [
            (row[col_city], float(row[col_amount]), float(row[col_cashback]))
            for _, row in city_agg.iterrows()
        ]

    days_active = best_day_date = best_day_amount = None
    if col_date:
        dates = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True).dt.date
        daily = df.groupby(dates)[col_amount].sum()
        days_active     = int(daily.count())
        best_day_date   = str(daily.idxmax())
        best_day_amount = float(daily.max())

    return {
        "total_transactions": total,
        "total_amount":       total_amount,
        "total_cashback":     total_cashback,
        "cashback_load_pct":  total_cashback / total_amount * 100 if total_amount else 0,
        "unique_clients":     int(df[col_client].nunique()) if col_client else None,
        "new_transactions":   len(df_new),
        "new_clients":        int(df_new[col_client].nunique()) if col_client else len(df_new),
        "new_amount":         new_amount,
        "new_avg_check":      new_amount / len(df_new) if len(df_new) else 0,
        "old_transactions":   len(df_old),
        "old_clients":        int(df_old[col_client].nunique()) if col_client else len(df_old),
        "old_amount":         old_amount,
        "old_avg_check":      old_amount / len(df_old) if len(df_old) else 0,
        "capped_count":       len(df_capped),
        "capped_pct":         len(df_capped) / total * 100 if total else 0,
        "top_cities":         top_cities,
        "days_active":        days_active,
        "best_day_date":      best_day_date,
        "best_day_amount":    best_day_amount,
    }


def _build_comparative_prompt(metrics_list: list[dict], params_list: list[dict]) -> str:
    lines = [f"Сравнительный анализ {len(metrics_list)} кампаний:\n"]
    for i, (m, p) in enumerate(zip(metrics_list, params_list), 1):
        lines += [
            f"Кампания {i}: {p['name']} (тип: {p.get('segment_type', 'Оба сегмента')})",
            f"  Оборот: {m['total_amount']:,.0f} руб",
            f"  Кэшбек: {m['total_cashback']:,.0f} руб ({m['cashback_load_pct']:.1f}%)",
            f"  Транзакций: {m['total_transactions']}",
        ]
        if m["unique_clients"] is not None:
            lines.append(f"  Клиентов: {m['unique_clients']}")
        lines += [
            f"  Новые клиенты: {m['new_transactions']} транз., {m['new_amount']:,.0f} руб, "
            f"ср. чек {m['new_avg_check']:,.0f} руб",
            f"  Действующие: {m['old_transactions']} транз., {m['old_amount']:,.0f} руб, "
            f"ср. чек {m['old_avg_check']:,.0f} руб",
            f"  Кэпнутых: {m['capped_count']} ({m['capped_pct']:.1f}%)",
            "",
        ]
    lines.append(
        "Сравни кампании: что изменилось, есть ли тренд, какая эффективнее и почему?"
    )
    return "\n".join(lines)


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Суммирует метрики нескольких кампаний для передачи в downstream-агентов."""
    sum_keys = [
        "total_transactions", "total_amount", "total_cashback",
        "new_transactions", "new_clients", "new_amount",
        "old_transactions", "old_clients", "old_amount", "capped_count",
    ]
    agg: dict = {}
    for k in sum_keys:
        agg[k] = sum(m.get(k) or 0 for m in metrics_list)

    agg["cashback_load_pct"] = (
        agg["total_cashback"] / agg["total_amount"] * 100 if agg["total_amount"] else 0
    )
    agg["unique_clients"]    = sum(m.get("unique_clients") or 0 for m in metrics_list)
    agg["new_avg_check"]     = agg["new_amount"] / agg["new_transactions"] if agg["new_transactions"] else 0
    agg["old_avg_check"]     = agg["old_amount"] / agg["old_transactions"] if agg["old_transactions"] else 0
    agg["capped_pct"]        = agg["capped_count"] / agg["total_transactions"] * 100 if agg["total_transactions"] else 0

    last = metrics_list[-1]
    agg["days_active"]     = last.get("days_active")
    agg["best_day_date"]   = last.get("best_day_date")
    agg["best_day_amount"] = last.get("best_day_amount")
    agg["top_cities"]      = None
    return agg


def run_analyst(dfs: list[pd.DataFrame], params_list: list[dict], client) -> dict:
    metrics_list = [_calc_metrics(df, p) for df, p in zip(dfs, params_list)]

    if len(dfs) == 1:
        prompt = _build_prompt(metrics_list[0], params_list[0])
    else:
        prompt = _build_comparative_prompt(metrics_list, params_list)

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=512,
    )

    aggregated = metrics_list[0] if len(metrics_list) == 1 else _aggregate_metrics(metrics_list)
    return {
        "metrics":      aggregated,
        "metrics_list": metrics_list,
        "params_list":  params_list,
        "summary":      response.choices[0].message.content,
    }


if __name__ == "__main__":
    test_df = pd.DataFrame(columns=["SUMMA_RUE", "CASHBACK_KB", "CLNT_ID", "OP_DATE", "MRC_CITY"])

    assert find_col(test_df, COLUMN_KEYWORDS["amount"])   == "SUMMA_RUE",   "amount: не найдено"
    assert find_col(test_df, COLUMN_KEYWORDS["cashback"]) == "CASHBACK_KB", "cashback: не найдено"
    assert find_col(test_df, COLUMN_KEYWORDS["client"])   == "CLNT_ID",     "client: не найдено"
    assert find_col(test_df, COLUMN_KEYWORDS["date"])     == "OP_DATE",     "date: не найдено"
    assert find_col(test_df, COLUMN_KEYWORDS["city"])     == "MRC_CITY",    "city: не найдено"
    assert find_col(test_df, ["nonexistent"])             is None,          "несуществующая: должна вернуть None"

    print("Все тесты прошли.")
