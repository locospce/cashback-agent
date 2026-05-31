import numpy as np
import pandas as pd

np.random.seed(42)

CAP    = 4500
CITIES = (
    ["Москва"] * 60 +
    ["Санкт-Петербург"] * 20 +
    ["Новосибирск"] * 5 +
    ["Екатеринбург"] * 5 +
    ["Казань"] * 4 +
    ["Нижний Новгород"] * 3 +
    ["Краснодар"] * 3
)


def _dates(n: int, start="2024-01-01", days=30) -> list:
    base = pd.Timestamp(start)
    return [base + pd.Timedelta(days=int(d)) for d in np.random.randint(0, days, n)]


def _cities(n: int) -> np.ndarray:
    return np.random.choice(CITIES, n)


def _add_overpayment(df: pd.DataFrame, pct: float = 0.05) -> pd.DataFrame:
    """Переплата: кэшбек > сумма * 16.5% (выше max_rate * 1.1 при new_pct=15)."""
    idx = np.random.choice(df.index, max(1, int(len(df) * pct)), replace=False)
    df.loc[idx, "CASHBACK_KB"] = (df.loc[idx, "SUMMA_RUE"] * 0.18).round(2)
    return df


def make_new_clients() -> pd.DataFrame:
    """200 строк — только новые клиенты (15%), уникальные CLNT_ID."""
    n       = 200
    amounts = np.random.uniform(500, 20_000, n).round(2)

    df = pd.DataFrame({
        "SUMMA_RUE":   amounts,
        "CASHBACK_KB": np.minimum(amounts * 0.15, CAP).round(2),
        "CLNT_ID":     [f"NEW{i:04d}" for i in range(1, n + 1)],
        "OP_DATE":     _dates(n),
        "CITY":        _cities(n),
    })

    # Аномалия — день с резким падением оборота:
    # последние 3 строки переводим на один день с маленькими суммами
    drop_date = pd.Timestamp("2024-01-28")
    df.loc[df.index[-3:], ["OP_DATE", "SUMMA_RUE", "CASHBACK_KB"]] = [
        [drop_date, 510.0,  76.5],
        [drop_date, 620.0,  93.0],
        [drop_date, 580.0,  87.0],
    ]

    return _add_overpayment(df)


def make_existing_clients() -> pd.DataFrame:
    """180 строк — только действующие клиенты (10%), повторяющиеся CLNT_ID."""
    n    = 180
    pool = [f"EXI{i:04d}" for i in range(1, 51)]   # 50 постоянных клиентов

    amounts = np.random.uniform(500, 20_000, n).round(2)

    # Аномалия — клиент EXI9999 с 15 транзакциями (> среднее + 3σ)
    n_normal  = n - 15
    clients   = list(np.random.choice(pool, n_normal)) + ["EXI9999"] * 15
    np.random.shuffle(clients)

    df = pd.DataFrame({
        "SUMMA_RUE":   amounts,
        "CASHBACK_KB": np.minimum(amounts * 0.10, CAP).round(2),
        "CLNT_ID":     clients,
        "OP_DATE":     _dates(n),
        "CITY":        _cities(n),
    })

    return _add_overpayment(df)


def make_both_segments() -> pd.DataFrame:
    """300 строк — новые (15%) и действующие (10%) клиенты вперемешку."""
    n_new, n_old = 150, 150

    amounts_new = np.random.uniform(500, 20_000, n_new).round(2)
    amounts_old = np.random.uniform(500, 20_000, n_old).round(2)
    pool_old    = [f"EXI{i:04d}" for i in range(51, 101)]

    # Аномалия — скачок оборота: несколько транзакций на один день с крупными суммами
    spike_date   = pd.Timestamp("2024-02-14")
    spike_n      = 8
    spike_amounts= np.full(spike_n, 19_500.0)

    n = n_new + n_old + spike_n
    df = pd.DataFrame({
        "SUMMA_RUE": np.concatenate([amounts_new, amounts_old, spike_amounts]),
        "CASHBACK_KB": np.concatenate([
            np.minimum(amounts_new * 0.15, CAP),
            np.minimum(amounts_old * 0.10, CAP),
            np.minimum(spike_amounts * 0.15, CAP),
        ]).round(2),
        "CLNT_ID": (
            [f"NEW{i:04d}" for i in range(201, 201 + n_new)] +
            list(np.random.choice(pool_old, n_old)) +
            [f"NEW{i:04d}" for i in range(401, 401 + spike_n)]
        ),
        "OP_DATE": _dates(n_new + n_old) + [spike_date] * spike_n,
        "CITY":    _cities(n),
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    return _add_overpayment(df)


FILES = {
    "demo_новые_клиенты.xlsx":      make_new_clients,
    "demo_действующие_клиенты.xlsx": make_existing_clients,
    "demo_оба_сегмента.xlsx":       make_both_segments,
}

if __name__ == "__main__":
    for filename, builder in FILES.items():
        df = builder()
        df.to_excel(filename, index=False)
        cashback_pct = df["CASHBACK_KB"].sum() / df["SUMMA_RUE"].sum() * 100
        print(
            f"{filename}: {len(df)} строк | "
            f"оборот {df['SUMMA_RUE'].sum():>12,.0f} руб | "
            f"кэшбек {cashback_pct:.1f}%"
        )
    print("\nФайлы созданы.")
