import re
import json


SYSTEM_PROMPT = (
    "Ты аналитик кэшбэк-программы банка. "
    "На основе данных о сегментах клиентов по нескольким кампаниям "
    "дай конкретную рекомендацию: запускать следующую кампанию "
    "на новых клиентов, действующих или оба сегмента.\n"
    "Ответь строго в формате JSON — только JSON, без лишнего текста:\n"
    '{"recommendation": "new" | "existing" | "both", '
    '"reasoning": "<3–4 предложения на русском>"}'
)


def _aggregate_segments(results: list[dict], params: list[dict]) -> dict:
    new_amount_total = new_cashback_est = 0.0
    new_transactions_total = campaigns_with_new = 0
    new_avg_checks = []

    old_amount_total = old_cashback_est = 0.0
    old_transactions_total = campaigns_with_old = 0
    old_avg_checks = []

    for r, p in zip(results, params):
        m = r["metrics"]

        if m["new_transactions"] > 0:
            campaigns_with_new  += 1
            new_amount_total    += m["new_amount"]
            new_cashback_est    += m["new_amount"] * p["new_pct"] / 100
            new_transactions_total += m["new_transactions"]
            new_avg_checks.append(m["new_avg_check"])

        if m["old_transactions"] > 0:
            campaigns_with_old  += 1
            old_amount_total    += m["old_amount"]
            old_cashback_est    += m["old_amount"] * p["old_pct"] / 100
            old_transactions_total += m["old_transactions"]
            old_avg_checks.append(m["old_avg_check"])

    return {
        "new_amount_total":      new_amount_total,
        "new_cashback_est":      new_cashback_est,
        "new_transactions_total": new_transactions_total,
        "new_roi":               new_amount_total / new_cashback_est if new_cashback_est else 0.0,
        "avg_new_check":         sum(new_avg_checks) / len(new_avg_checks) if new_avg_checks else 0.0,
        "campaigns_with_new":    campaigns_with_new,
        "old_amount_total":      old_amount_total,
        "old_cashback_est":      old_cashback_est,
        "old_transactions_total": old_transactions_total,
        "old_roi":               old_amount_total / old_cashback_est if old_cashback_est else 0.0,
        "avg_old_check":         sum(old_avg_checks) / len(old_avg_checks) if old_avg_checks else 0.0,
        "campaigns_with_old":    campaigns_with_old,
    }


def _build_prompt(agg: dict, results: list[dict], params: list[dict]) -> str:
    n = len(results)
    lines = [f"Анализ сегментов по {n} кампаниям:\n"]

    for i, (r, p) in enumerate(zip(results, params), 1):
        m = r["metrics"]
        lines += [
            f"Кампания {i}: {p['name']}",
            f"  Новые клиенты ({p['new_pct']}%): {m['new_transactions']} транз., "
            f"оборот {m['new_amount']:,.0f} руб, средний чек {m['new_avg_check']:,.0f} руб",
            f"  Действующие ({p['old_pct']}%): {m['old_transactions']} транз., "
            f"оборот {m['old_amount']:,.0f} руб, средний чек {m['old_avg_check']:,.0f} руб",
            "",
        ]

    lines += [
        "Итого по всем кампаниям:",
        f"  Новые клиенты — присутствовали в {agg['campaigns_with_new']}/{n} кампаниях",
        f"    Оборот: {agg['new_amount_total']:,.0f} руб",
        f"    Кэшбек (оценка): {agg['new_cashback_est']:,.0f} руб",
        f"    ROI (оборот / кэшбэк): {agg['new_roi']:.2f}x",
        f"    Средний чек: {agg['avg_new_check']:,.0f} руб",
        f"  Действующие — присутствовали в {agg['campaigns_with_old']}/{n} кампаниях",
        f"    Оборот: {agg['old_amount_total']:,.0f} руб",
        f"    Кэшбек (оценка): {agg['old_cashback_est']:,.0f} руб",
        f"    ROI (оборот / кэшбэк): {agg['old_roi']:.2f}x",
        f"    Средний чек: {agg['avg_old_check']:,.0f} руб",
        "",
        "На основе этих данных — запускать следующую кампанию на новых клиентов, "
        "действующих клиентов или оба сегмента? Дай конкретную рекомендацию с обоснованием.",
    ]
    return "\n".join(lines)


def _parse_response(raw: str, new_roi: float, old_roi: float) -> tuple[str, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            rec = data.get("recommendation", "").lower()
            if rec not in ("new", "existing", "both"):
                rec = "both"
            return rec, data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            pass
    # Фолбэк: выбираем сегмент с лучшим ROI
    if new_roi > old_roi:
        return "new", raw.strip()
    if old_roi > new_roi:
        return "existing", raw.strip()
    return "both", raw.strip()


def run_segmentor(results: list[dict], params: list[dict], client) -> dict:
    agg = _aggregate_segments(results, params)
    prompt = _build_prompt(agg, results, params)

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=512,
    )

    recommendation, reasoning = _parse_response(
        response.choices[0].message.content, agg["new_roi"], agg["old_roi"]
    )

    return {
        "new_clients_roi":      round(agg["new_roi"], 2),
        "existing_clients_roi": round(agg["old_roi"], 2),
        "recommendation":       recommendation,
        "reasoning":            reasoning,
    }
