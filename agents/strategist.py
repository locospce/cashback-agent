import re
import json


SYSTEM_PROMPT = (
    "Ты старший стратег кэшбек-программы банка. "
    "Изучи результаты аналитических агентов и дай финальную рекомендацию по следующей кампании.\n"
    "Ответь строго в формате JSON — только JSON, без лишнего текста:\n"
    "{\n"
    '  "target_segment": "new" | "existing" | "both",\n'
    '  "cashback_new_pct": <число от 5 до 25>,\n'
    '  "cashback_old_pct": <число от 3 до 15>,\n'
    '  "recommended_cap": <целое число в рублях>,\n'
    '  "key_risks": ["риск 1", "риск 2"],\n'
    '  "summary": "<итоговая рекомендация 4–6 предложений на русском>"\n'
    "}"
)

_SEG_LABELS = {"new": "новые клиенты", "existing": "действующие", "both": "оба сегмента"}


def _build_prompt(
    analyst_results: list[dict],
    forecast: dict,
    segmentor_result: dict,
    anomalies: list[dict],
    params: list[dict],
) -> str:
    lines = ["АНАЛИЗ КАМПАНИЙ:"]

    for i, (r, p) in enumerate(zip(analyst_results, params), 1):
        m = r["metrics"]
        lines += [
            f"  Кампания {i} «{p['name']}»: "
            f"оборот {m['total_amount']:,.0f} руб, "
            f"кэшбек {m['total_cashback']:,.0f} руб ({m['cashback_load_pct']:.1f}%), "
            f"{m['total_transactions']} транз.",
            f"  Вывод аналитика: {r['summary']}",
            "",
        ]

    f = forecast
    lines += [
        "ПРОГНОЗ ОБОРОТА СЛЕДУЮЩЕЙ КАМПАНИИ:",
        f"  Линейный тренд:   {f['trend_forecast']:>15,} руб",
        f"  Средневзвешенный: {f['weighted_forecast']:>15,} руб",
        f"  LLM-прогноз:      {f['llm_forecast']['value']:>15,} руб",
        f"  Консенсус:        {f['consensus']:>15,} руб",
        f"  Обоснование: {f['llm_forecast']['reasoning']}",
        "",
    ]

    seg = segmentor_result
    lines += [
        "АНАЛИЗ СЕГМЕНТОВ:",
        f"  ROI новых клиентов:      {seg['new_clients_roi']:.2f}x",
        f"  ROI действующих:   {seg['existing_clients_roi']:.2f}x",
        f"  Рекомендация: {_SEG_LABELS.get(seg['recommendation'], seg['recommendation'])}",
        f"  Обоснование: {seg['reasoning']}",
        "",
    ]

    if anomalies:
        lines.append("АНОМАЛИИ:")
        for a in anomalies:
            lines.append(
                f"  [{a['severity'].upper()}] {a['type']}: {a['description']} "
                f"→ {a['recommendation']}"
            )
    else:
        lines.append("АНОМАЛИИ: не выявлены")

    last = params[-1]
    lines += [
        "",
        f"Текущие параметры (последняя кампания): "
        f"новые клиенты {last['new_pct']}%, действующие {last['old_pct']}%, кэп {last['cap']:,} руб.",
        "",
        "Дай финальную рекомендацию по следующей кампании:\n"
        "1. Целевой сегмент (new / existing / both)\n"
        "2. Рекомендуемый % кэшбека для каждого сегмента\n"
        "3. Рекомендуемый кэп в рублях\n"
        "4. Ключевые риски — на что обратить внимание",
    ]

    return "\n".join(lines)


def _parse_response(raw: str, params: list[dict]) -> dict:
    last = params[-1]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            rec = data.get("target_segment", "both").lower()
            if rec not in ("new", "existing", "both"):
                rec = "both"
            return {
                "target_segment":   rec,
                "cashback_new_pct": float(data.get("cashback_new_pct", last["new_pct"])),
                "cashback_old_pct": float(data.get("cashback_old_pct", last["old_pct"])),
                "recommended_cap":  int(data.get("recommended_cap",   last["cap"])),
                "key_risks":        list(data.get("key_risks",         [])),
                "summary":          str(data.get("summary",            "")),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return {
        "target_segment":   "both",
        "cashback_new_pct": float(last["new_pct"]),
        "cashback_old_pct": float(last["old_pct"]),
        "recommended_cap":  int(last["cap"]),
        "key_risks":        [],
        "summary":          raw.strip(),
    }


def run_strategist(
    analyst_results: list[dict],
    forecast: dict,
    segmentor_result: dict,
    anomalies: list[dict],
    params: list[dict],
    client,
) -> dict:
    prompt = _build_prompt(analyst_results, forecast, segmentor_result, anomalies, params)

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
    )

    return _parse_response(response.choices[0].message.content, params)


if __name__ == "__main__":
    # --- фикстуры ---
    _analyst_results = [
        {
            "metrics": {
                "total_amount": 5_000_000, "total_cashback": 500_000,
                "cashback_load_pct": 10.0, "total_transactions": 1000,
                "unique_clients": 400, "new_transactions": 300, "new_amount": 1_500_000,
                "new_avg_check": 5_000, "old_transactions": 700, "old_amount": 3_500_000,
                "old_avg_check": 5_000, "capped_count": 50, "capped_pct": 5.0,
                "days_active": 30, "best_day_date": "2024-01-15",
                "best_day_amount": 250_000, "top_cities": None,
            },
            "summary": "Кампания прошла стабильно, нагрузка кэшбека в норме.",
        },
    ]
    _forecast = {
        "trend_forecast": 5_500_000, "weighted_forecast": 5_200_000,
        "llm_forecast": {"value": 5_300_000, "reasoning": "Стабильный рост."},
        "consensus": 5_333_333,
    }
    _segmentor = {
        "new_clients_roi": 6.67, "existing_clients_roi": 10.0,
        "recommendation": "both",
        "reasoning": "Оба сегмента активны, действующие дают лучший ROI.",
    }
    _anomalies = [
        {
            "type": "высокий_процент_кэпа", "severity": "средняя",
            "description": "35% транзакций достигли кэпа.",
            "recommendation": "Увеличить кэп.",
        }
    ]
    _params = [{"name": "Тест", "new_pct": 15, "old_pct": 10, "cap": 4500, "cap_type": "на транзакцию"}]

    # --- тест _build_prompt ---
    prompt = _build_prompt(_analyst_results, _forecast, _segmentor, _anomalies, _params)
    assert "АНАЛИЗ КАМПАНИЙ"   in prompt
    assert "ПРОГНОЗ"           in prompt
    assert "5,333,333"         in prompt
    assert "АНОМАЛИИ"          in prompt
    assert "высокий_процент"   in prompt

    # --- тест _parse_response: валидный JSON ---
    raw_ok = (
        '{"target_segment": "both", "cashback_new_pct": 14, "cashback_old_pct": 9, '
        '"recommended_cap": 5000, "key_risks": ["риск А"], "summary": "Всё хорошо."}'
    )
    r = _parse_response(raw_ok, _params)
    assert r["target_segment"]   == "both"
    assert r["cashback_new_pct"] == 14.0
    assert r["recommended_cap"]  == 5000
    assert r["key_risks"]        == ["риск А"]

    # --- тест _parse_response: невалидный JSON → фолбэк ---
    r2 = _parse_response("Не смог ответить в JSON.", _params)
    assert r2["target_segment"]  == "both"
    assert r2["cashback_new_pct"] == 15.0
    assert r2["summary"]         == "Не смог ответить в JSON."

    # --- тест _parse_response: недопустимый target_segment → "both" ---
    raw_bad_seg = '{"target_segment": "unknown", "cashback_new_pct": 12, "cashback_old_pct": 8, "recommended_cap": 4000, "key_risks": [], "summary": "?"}'
    r3 = _parse_response(raw_bad_seg, _params)
    assert r3["target_segment"] == "both"

    print("Все тесты прошли.")
