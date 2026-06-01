import pandas as pd
import numpy as np
from agents.analyst import find_col, COLUMN_KEYWORDS


def run_anomaly(df: pd.DataFrame, params: dict) -> list[dict]:
    col_amount   = find_col(df, COLUMN_KEYWORDS["amount"])
    col_cashback = find_col(df, COLUMN_KEYWORDS["cashback"])
    col_client   = find_col(df, COLUMN_KEYWORDS["client"])
    col_date     = find_col(df, COLUMN_KEYWORDS["date"])

    if not col_amount or not col_cashback:
        return []

    anomalies = []
    total     = len(df)
    cap       = params["cap"]
    max_rate  = max(params["new_pct"], params["old_pct"]) / 100

    # 1. Переплата: кэшбэк > сумма * max_rate * 1.1
    overpaid     = df[df[col_cashback] > df[col_amount] * max_rate * 1.1]
    overpaid_n   = len(overpaid)
    if overpaid_n > 0:
        overpaid_pct   = overpaid_n / total * 100
        excess_amount  = (overpaid[col_cashback] - overpaid[col_amount] * max_rate).sum()
        severity       = "высокая" if overpaid_pct > 5 else ("средняя" if overpaid_pct > 1 else "низкая")
        anomalies.append({
            "type":           "Переплата кэшбэка",
            "severity":       severity,
            "description":    (
                f"{overpaid_n} транзакций ({overpaid_pct:.1f}%) с кэшбэком выше "
                f"{params['new_pct']}% + 10%. Избыточная выплата: ~{excess_amount:,.0f} руб."
            ),
            "recommendation": (
                "Проверить правила начисления в процессинге. "
                "Возможны двойное начисление или ошибка в ставке."
            ),
        })

    # 2. Клиенты с аномально высоким числом транзакций (среднее + 3σ)
    if col_client:
        counts    = df.groupby(col_client).size()
        threshold = counts.mean() + 3 * counts.std()
        suspects  = counts[counts > threshold]
        if len(suspects) > 0:
            anomalies.append({
                "type":           "Подозрительные клиенты",
                "severity":       "высокая",
                "description":    (
                    f"{len(suspects)} клиент(ов) с аномальным числом транзакций "
                    f"(порог: {threshold:.0f}, макс: {suspects.max()})."
                ),
                "recommendation": (
                    "Проверить клиентов на фрод или технические дубли транзакций."
                ),
            })

    # 3. Дни с резким изменением оборота (> ±50% от среднего)
    if col_date:
        tmp         = df.copy()
        tmp["_d"]   = pd.to_datetime(tmp[col_date], errors="coerce", dayfirst=True).dt.date
        daily       = tmp.groupby("_d")[col_amount].sum()

        if len(daily) > 1:
            mean_d = daily.mean()

            drop_days = daily[daily < mean_d * 0.5]
            if len(drop_days) > 0:
                worst_val = drop_days.min()
                worst_pct = worst_val / mean_d * 100
                severity  = "высокая" if worst_pct < 30 else "средняя"
                anomalies.append({
                    "type":           "Падение оборота",
                    "severity":       severity,
                    "description":    (
                        f"{len(drop_days)} дней с оборотом ниже 50% от среднего "
                        f"({mean_d:,.0f} руб). "
                        f"Худший день: {drop_days.idxmin()} — {worst_val:,.0f} руб "
                        f"({worst_pct:.0f}% от среднего)."
                    ),
                    "recommendation": (
                        "Проверить причину: технический сбой, праздник или отток трафика."
                    ),
                })

            spike_days = daily[daily > mean_d * 1.5]
            if len(spike_days) > 0:
                best_val = spike_days.max()
                best_pct = best_val / mean_d * 100
                severity = "средняя" if best_pct > 300 else "низкая"
                anomalies.append({
                    "type":           "Скачок оборота",
                    "severity":       severity,
                    "description":    (
                        f"{len(spike_days)} дней с оборотом выше 150% от среднего. "
                        f"Пик: {spike_days.idxmax()} — {best_val:,.0f} руб "
                        f"({best_pct:.0f}% от среднего)."
                    ),
                    "recommendation": (
                        "Изучить причину всплеска и использовать опыт при планировании следующей кампании."
                    ),
                })

    # 4. Высокий процент кэпнутых транзакций (> 30%)
    capped_n   = int((df[col_cashback] >= cap * 0.99).sum())
    capped_pct = capped_n / total * 100
    if capped_pct > 30:
        severity = "высокая" if capped_pct > 50 else "средняя"
        anomalies.append({
            "type":           "Высокий процент кэпа",
            "severity":       severity,
            "description":    (
                f"{capped_pct:.1f}% транзакций достигли кэпа {cap:,.0f} руб. "
                f"Клиенты систематически не дополучают ожидаемый кэшбэк."
            ),
            "recommendation": (
                f"Рассмотреть увеличение кэпа — текущий порог занижен для аудитории кампании."
            ),
        })

    return anomalies


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    n = 300
    amounts   = np.random.uniform(500, 10_000, n)
    cashbacks = amounts * 0.10

    # Вставляем переплату в 10 транзакций
    cashbacks[:10] = amounts[:10] * 0.17

    # Один подозрительный клиент (100 транзакций)
    clients = [f"C{i % 50}" for i in range(n)]
    clients[:100] = ["FRAUD"] * 100

    # Один день с падением
    dates = pd.date_range("2024-01-01", periods=n, freq="h").date.tolist()
    for i in range(20, 30):
        amounts[i] *= 0.1

    df_test = pd.DataFrame({
        "SUMMA_RUE":   amounts,
        "CASHBACK_KB": cashbacks,
        "CLNT_ID":     clients,
        "OP_DATE":     dates,
    })

    test_params = {"name": "Тест", "new_pct": 15, "old_pct": 10, "cap": 4500, "cap_type": "на транзакцию"}
    result = run_anomaly(df_test, test_params)

    assert any(a["type"] == "Переплата кэшбэка"      for a in result), "переплата не найдена"
    assert any(a["type"] == "Подозрительные клиенты" for a in result), "фрод не найден"
    assert any(a["type"] == "Падение оборота"         for a in result), "падение не найдено"

    for a in result:
        print(f"[{a['severity'].upper()}] {a['type']}: {a['description']}")

    print("\nВсе тесты прошли.")
