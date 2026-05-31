import re
import json
import numpy as np


# Базовые веса [первая, предпоследняя, последняя]; для n<3 берём правый хвост и нормируем
_BASE_WEIGHTS = np.array([0.2, 0.3, 0.5])

SYSTEM_PROMPT = (
    "Ты финансовый аналитик кэшбек-программы банка. "
    "На основе данных о прошлых кампаниях спрогнозируй оборот следующей. "
    "Ответь строго в формате JSON — только JSON, без лишнего текста:\n"
    '{"value": <целое число в рублях>, "reasoning": "<2–3 предложения на русском>"}'
)


def _linear_trend(amounts: list[float]) -> float:
    n = len(amounts)
    if n == 1:
        return amounts[0]
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, amounts, deg=1)
    return float(np.polyval(coeffs, n))


def _weighted_forecast(amounts: list[float]) -> float:
    n = len(amounts)
    w = _BASE_WEIGHTS[-n:]          # правый хвост для n < 3
    w = w / w.sum()                 # нормируем, чтобы сумма = 1
    return float(np.dot(w, amounts[-n:]))


def _build_llm_prompt(results: list[dict], params: list[dict]) -> str:
    lines = ["Данные по прошлым кэшбек-кампаниям:\n"]
    for i, (r, p) in enumerate(zip(results, params), 1):
        m = r["metrics"]
        lines += [
            f"Кампания {i}: {p['name']}",
            f"  Оборот: {m['total_amount']:,.0f} руб",
            f"  Кэшбек: {m['total_cashback']:,.0f} руб ({m['cashback_load_pct']:.1f}%)",
            f"  Транзакций: {m['total_transactions']}",
        ]
        if m["unique_clients"] is not None:
            lines.append(f"  Клиентов: {m['unique_clients']}")
        lines += [
            f"  Новые клиенты: {m['new_transactions']} транз., {m['new_amount']:,.0f} руб",
            f"  Действующие: {m['old_transactions']} транз., {m['old_amount']:,.0f} руб",
            f"  Кэпнутых: {m['capped_count']} ({m['capped_pct']:.1f}%)",
        ]
        if m["days_active"] is not None:
            lines.append(f"  Активных дней: {m['days_active']}, "
                         f"лучший день {m['best_day_date']} — {m['best_day_amount']:,.0f} руб")
        lines.append("")
    lines.append("Спрогнозируй оборот следующей кампании в рублях.")
    return "\n".join(lines)


def _parse_llm_response(raw: str, fallback: float) -> dict:
    # Claude может обернуть JSON в ```json ... ```, ищем первый {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {"value": float(data["value"]), "reasoning": data.get("reasoning", "")}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return {"value": fallback, "reasoning": raw.strip()}


def run_forecaster(results: list[dict], params: list[dict], client) -> dict:
    amounts = [r["metrics"]["total_amount"] for r in results]

    trend_forecast    = _linear_trend(amounts)
    weighted_forecast = _weighted_forecast(amounts)

    prompt  = _build_llm_prompt(results, params)
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=512,
    )
    llm = _parse_llm_response(response.choices[0].message.content, fallback=weighted_forecast)

    consensus = (trend_forecast + weighted_forecast + llm["value"]) / 3

    return {
        "trend_forecast":    round(trend_forecast),
        "weighted_forecast": round(weighted_forecast),
        "llm_forecast":      {"value": round(llm["value"]), "reasoning": llm["reasoning"]},
        "consensus":         round(consensus),
    }
