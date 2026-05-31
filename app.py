import os

from openai import OpenAI, APIStatusError
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.analyst import COLUMN_KEYWORDS, detect_params, find_col, run_analyst
from agents.anomaly import run_anomaly
from agents.forecaster import run_forecaster
from agents.segmentor import run_segmentor
from agents.strategist import run_strategist

load_dotenv()

st.set_page_config(page_title="Кэшбек агент", layout="wide")
st.title("AI-агент кэшбек кампаний")

_SEG_LABELS = {"new": "Новые клиенты", "existing": "Действующие", "both": "Оба сегмента"}
_SEVERITY_FN = {"высокая": st.error, "средняя": st.warning, "низкая": st.info}

_MONTH_NAMES = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def _detect_period(df: pd.DataFrame) -> tuple[int, int, str]:
    """Определяет период кампании по датам в данных."""
    col_date = find_col(df, COLUMN_KEYWORDS["date"])
    if col_date:
        dates = pd.to_datetime(df[col_date], errors="coerce").dropna()
        if not dates.empty:
            period = dates.dt.to_period("M").value_counts().index[0]
            label  = f"{_MONTH_NAMES.get(period.month, str(period.month))} {period.year}"
            return period.year, period.month, label
    return 9999, 99, "Неизвестный период"


def load_file(f) -> pd.DataFrame:
    if f.name.endswith(".csv"):
        return pd.read_csv(f, sep=None, engine="python")
    return pd.read_excel(f)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Загрузка данных")
uploaded_files = st.sidebar.file_uploader(
    "Загрузи до 3 файлов", type=["xlsx", "xls", "csv"], accept_multiple_files=True
)

# ── File loading ──────────────────────────────────────────────────────────────
if not uploaded_files:
    st.stop()

if len(uploaded_files) > 3:
    st.error("Максимум 3 файла одновременно.")
    st.stop()

# Загружаем файлы, определяем период и параметры из данных каждого файла
_loaded = [(load_file(f), f) for f in uploaded_files]
_loaded.sort(key=lambda x: _detect_period(x[0])[:2])

dfs, params_list = [], []
for df, f in _loaded:
    dfs.append(df)
    _, _, display_name = _detect_period(df)
    det = detect_params(df)
    st.sidebar.write(f"**{display_name}** — {len(df):,} строк")
    st.sidebar.caption(
        f"Новые клиенты: {det['new_pct']}% · "
        f"Действующие: {det['old_pct']}% · "
        f"Кэп: {det['cap']:,} руб"
    )
    params_list.append({
        "name":     display_name,
        "new_pct":  det["new_pct"],
        "old_pct":  det["old_pct"],
        "cap":      det["cap"],
        "cap_type": "на транзакцию",
    })

df_all = pd.concat(dfs, ignore_index=True)

# ── Column detection ──────────────────────────────────────────────────────────
col_amount   = find_col(df_all, COLUMN_KEYWORDS["amount"])
col_cashback = find_col(df_all, COLUMN_KEYWORDS["cashback"])
col_client   = find_col(df_all, COLUMN_KEYWORDS["client"])
col_date     = find_col(df_all, COLUMN_KEYWORDS["date"])
col_city     = find_col(df_all, COLUMN_KEYWORDS["city"])

if not col_amount or not col_cashback:
    st.error(f"Не найдены колонки суммы/кэшбека. Колонки: {list(df_all.columns)}")
    st.stop()

st.sidebar.markdown("**Найденные колонки:**")
st.sidebar.write(f"Сумма: `{col_amount}`")
st.sidebar.write(f"Кэшбек: `{col_cashback}`")
st.sidebar.write(f"Клиент: `{col_client or 'не найден'}`")
st.sidebar.write(f"Дата: `{col_date or 'не найдена'}`")
st.sidebar.write(f"Город: `{col_city or 'не найден'}`")

# ── Overview ──────────────────────────────────────────────────────────────────
def _campaign_metrics(df: pd.DataFrame, p: dict) -> dict:
    a = find_col(df, COLUMN_KEYWORDS["amount"])
    cb = find_col(df, COLUMN_KEYWORDS["cashback"])
    cl = find_col(df, COLUMN_KEYWORDS["client"])
    if not a or not cb:
        return {}
    total_amt = df[a].sum()
    total_cb  = df[cb].sum()
    df2 = df.copy()
    df2["_pct"] = df2[cb] / df2[a] * 100
    tol = 3
    df_new    = df2[abs(df2["_pct"] - p["new_pct"]) <= tol]
    df_old    = df2[abs(df2["_pct"] - p["old_pct"]) <= tol]
    df_capped = df2[df2[cb] >= p["cap"] * 0.99]
    return {
        "total":          len(df),
        "total_amount":   total_amt,
        "total_cashback": total_cb,
        "load_pct":       total_cb / total_amt * 100 if total_amt else 0,
        "clients":        df[cl].nunique() if cl else 0,
        "new_clients":    df_new[cl].nunique() if cl else len(df_new),
        "old_clients":    df_old[cl].nunique() if cl else len(df_old),
        "capped_n":       len(df_capped),
        "capped_pct":     len(df_capped) / len(df) * 100 if len(df) else 0,
    }


if len(dfs) == 1:
    # ── Одна кампания: стандартный вид ───────────────────────────────────────
    st.subheader("Данные")
    st.dataframe(df_all.head(10))

    cm = _campaign_metrics(dfs[0], params_list[0])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Транзакций",          f"{cm['total']:,}")
    m2.metric("Оборот",              f"{cm['total_amount']:,.0f} руб")
    m3.metric("Кэшбек выплачен",     f"{cm['total_cashback']:,.0f} руб")
    m4.metric("Уникальных клиентов", f"{cm['clients']:,}")

    st.subheader("Разбивка по сегментам")
    s1, s2, s3 = st.columns(3)
    s1.metric("Новые клиенты",       cm["new_clients"])
    s2.metric("Действующие клиенты", cm["old_clients"])
    s3.metric("Кэпнутых транзакций", f"{cm['capped_n']} ({cm['capped_pct']:.1f}%)")

else:
    # ── Несколько кампаний: сравнение по столбцам ─────────────────────────────
    st.subheader("Сравнение кампаний")
    cols = st.columns(len(dfs))
    for col_ui, df, p in zip(cols, dfs, params_list):
        cm = _campaign_metrics(df, p)
        with col_ui:
            st.markdown(f"**{p['name']}**")
            st.metric("Оборот",              f"{cm['total_amount']:,.0f} руб")
            st.metric("Кэшбек выплачен",     f"{cm['total_cashback']:,.0f} руб")
            st.metric("Нагрузка",            f"{cm['load_pct']:.1f}%")
            st.caption(f"С каждых 100 руб покупки банк выплатил {cm['load_pct']:.2f} руб кэшбека")
            st.metric("Уникальных клиентов", f"{cm['clients']:,}")
            st.metric("Новые клиенты",       cm["new_clients"])
            st.metric("Действующие клиенты", cm["old_clients"])
            st.metric("Кэпнутых транзакций", f"{cm['capped_n']} ({cm['capped_pct']:.1f}%)")

# ── Города: всегда по объединённым данным ────────────────────────────────────
df_all = df_all.copy()
df_all["_pct"] = df_all[col_cashback] / df_all[col_amount] * 100

if col_city:
    st.subheader("Топ-города по обороту")
    city_df = (
        df_all.groupby(col_city)
        .agg(оборот=(col_amount, "sum"), кэшбек=(col_cashback, "sum"), транзакций=(col_amount, "count"))
        .sort_values("оборот", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={col_city: "Город"})
    )
    city_df["оборот"] = city_df["оборот"].map("{:,.0f} руб".format)
    city_df["кэшбек"] = city_df["кэшбек"].map("{:,.0f} руб".format)
    st.dataframe(city_df, use_container_width=True)

# ── Agent pipeline ────────────────────────────────────────────────────────────
if not st.button("Запустить анализ"):
    st.stop()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY не найден в .env")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

total_steps = 5   # аналитик + аномалии + прогноз + сегментатор + стратег
steps_done  = 0

progress = st.progress(0)
status   = st.empty()


def _advance(label: str):
    global steps_done
    steps_done += 1
    progress.progress(steps_done / total_steps)
    status.text(label)


combined_params = {**params_list[0], "name": " + ".join(p["name"] for p in params_list)}

try:
    # 1. Аналитик — один вызов Claude со всеми таблицами сразу
    status.text("Аналитик анализирует данные...")
    result       = run_analyst(dfs, params_list, client)
    analyst_results = [result]
    metrics_list = result["metrics_list"]
    p_list       = result["params_list"]

    with st.expander("Результаты анализа", expanded=True):
        if len(metrics_list) == 1:
            m = metrics_list[0]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Оборот",    f"{m['total_amount']:,.0f} руб")
            a2.metric("Кэшбек",    f"{m['total_cashback']:,.0f} руб")
            a3.metric("Нагрузка",  f"{m['cashback_load_pct']:.1f}%")
            a3.caption(f"С каждых 100 руб покупки банк выплатил {m['cashback_load_pct']:.2f} руб кэшбека")
            if m["unique_clients"] is not None:
                a4.metric("Клиентов", f"{m['unique_clients']:,}")
            b1, b2, b3 = st.columns(3)
            b1.metric("Новые клиенты",       f"{m['new_clients']:,}")
            b2.metric("Действующие клиенты", f"{m['old_clients']:,}")
            b3.metric("Кэпнутых",            f"{m['capped_count']} ({m['capped_pct']:.1f}%)")
            if m["best_day_amount"]:
                st.caption(f"Лучший день: {m['best_day_date']} — {m['best_day_amount']:,.0f} руб")
        else:
            cols_ui = st.columns(len(metrics_list))
            for col_ui, m, p in zip(cols_ui, metrics_list, p_list):
                with col_ui:
                    st.markdown(f"**{p['name']}**")
                    st.metric("Оборот",              f"{m['total_amount']:,.0f} руб")
                    st.metric("Кэшбек",              f"{m['total_cashback']:,.0f} руб")
                    st.metric("Нагрузка",            f"{m['cashback_load_pct']:.1f}%")
                    st.caption(f"С каждых 100 руб — {m['cashback_load_pct']:.2f} руб кэшбека")
                    if m["unique_clients"] is not None:
                        st.metric("Клиентов",        f"{m['unique_clients']:,}")
                    st.metric("Новые клиенты",       f"{m['new_clients']:,}")
                    st.metric("Действующие клиенты", f"{m['old_clients']:,}")
                    st.metric("Кэпнутых",            f"{m['capped_count']} ({m['capped_pct']:.1f}%)")
                    if m["best_day_amount"]:
                        st.caption(f"Лучший день: {m['best_day_date']} — {m['best_day_amount']:,.0f} руб")

        params_caption = " | ".join(
            f"{p['name']}: новые {p['new_pct']}%, действующие {p['old_pct']}%, кэп {p['cap']:,} руб"
            for p in p_list
        )
        st.caption(f"Параметры сегментации: {params_caption}")
        st.markdown("**Вывод аналитика:**")
        st.write(result["summary"])

    _advance("Анализ завершён.")

    # 2. Аномалии
    status.text("Проверяю аномалии...")
    anomalies = run_anomaly(df_all, combined_params)
    _advance("Аномалии проверены.")

    with st.expander("Аномалии", expanded=bool(anomalies)):
        if not anomalies:
            st.success("Аномалий не обнаружено.")
        else:
            for a in anomalies:
                fn = _SEVERITY_FN.get(a["severity"], st.info)
                fn(f"**{a['type']}** [{a['severity']}] — {a['description']}  \n*{a['recommendation']}*")

    # 3. Прогноз
    status.text("Строю прогноз...")
    forecast = run_forecaster(analyst_results, [combined_params], client)
    _advance("Прогноз готов.")

    with st.expander("Прогноз оборота", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Линейный тренд",   f"{forecast['trend_forecast']:,} руб")
        f2.metric("Средневзвешенный", f"{forecast['weighted_forecast']:,} руб")
        f3.metric("LLM-прогноз",      f"{forecast['llm_forecast']['value']:,} руб")
        f4.metric("Консенсус",        f"{forecast['consensus']:,} руб")
        st.caption(forecast["llm_forecast"]["reasoning"])

    # 4. Сегментатор
    status.text("Анализирую сегменты...")
    seg = run_segmentor(analyst_results, [combined_params], client)
    _advance("Сегменты проанализированы.")

    with st.expander("Анализ сегментов", expanded=True):
        g1, g2, g3 = st.columns(3)
        new_roi = seg['new_clients_roi']
        old_roi = seg['existing_clients_roi']
        g1.metric("ROI новых клиентов",    f"{new_roi:.2f}x")
        g1.caption(f"На каждые 100 руб кэшбека новые клиенты принесли {new_roi * 100:,.0f} руб оборота")
        g2.metric("ROI действующих клиентов", f"{old_roi:.2f}x")
        g2.caption(f"На каждые 100 руб кэшбека действующие клиенты принесли {old_roi * 100:,.0f} руб оборота")
        g3.metric("Рекомендация", _SEG_LABELS.get(seg["recommendation"], seg["recommendation"]))
        st.write(seg["reasoning"])

    # 5. Стратег
    status.text("Формирую стратегию...")
    strategy = run_strategist(analyst_results, forecast, seg, anomalies, [combined_params], client)
    _advance("Стратегия готова.")

    with st.expander("Стратегия", expanded=True):
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Сегмент",             _SEG_LABELS.get(strategy["target_segment"], strategy["target_segment"]))
        h2.metric("% кэшбека (новые)",   f"{strategy['cashback_new_pct']}%")
        h3.metric("% кэшбека (действ.)", f"{strategy['cashback_old_pct']}%")
        h4.metric("Кэп",                 f"{strategy['recommended_cap']:,} руб")

        if strategy["key_risks"]:
            st.markdown("**Ключевые риски:**")
            for risk in strategy["key_risks"]:
                st.warning(risk)

        st.markdown("**Итоговая рекомендация:**")
        st.write(strategy["summary"])

    status.text("Анализ завершён.")

except APIStatusError as e:
    progress.empty()
    if e.status_code == 429:
        st.error(
            "Превышена квота OpenRouter (429).  \n"
            "Проверь баланс и лимиты на [openrouter.ai/settings](https://openrouter.ai/settings/credits)"
        )
    else:
        st.error(f"Ошибка OpenRouter ({e.status_code}): {e.message}")
    status.empty()
