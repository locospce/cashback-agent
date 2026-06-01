import re
import json


_LIMITS = {"push": 60, "banner": 120, "sms": 160}

_SEGMENT_FOCUS = {
    "new":      "акцент на первой покупке у партнёра и приветственном кэшбэке",
    "existing": "акцент на лояльности, вознаграждении постоянных клиентов",
    "both":     "нейтральный текст, подходящий для любой аудитории",
}

SYSTEM_PROMPT = (
    "Ты копирайтер банковских кэшбэк-кампаний. "
    "Пиши живо и конкретно: указывай процент и сумму кэшбэка. "
    "Без общих фраз вроде «выгодное предложение» или «не упустите шанс».\n"
    "Ответь строго в формате JSON — только JSON, без лишнего текста:\n"
    '{"push": "<текст>", "banner": "<текст>", "sms": "<текст>"}'
)


def _build_prompt(strategy: dict) -> str:
    seg       = strategy.get("target_segment", "both")
    new_pct   = strategy.get("cashback_new_pct", 15)
    old_pct   = strategy.get("cashback_old_pct", 10)
    cap       = strategy.get("recommended_cap", 4500)
    summary   = strategy.get("summary", "")
    focus     = _SEGMENT_FOCUS.get(seg, _SEGMENT_FOCUS["both"])

    if seg == "new":
        pct_line = f"Кэшбек {new_pct}% на первую покупку, кэп {cap:,} руб."
    elif seg == "existing":
        pct_line = f"Кэшбек {old_pct}% для постоянных клиентов, кэп {cap:,} руб."
    else:
        pct_line = f"Кэшбек до {new_pct}% (новые) и {old_pct}% (действующие), кэп {cap:,} руб."

    lines = [
        f"Параметры кампании: {pct_line}",
        f"Акцент: {focus}",
    ]
    if summary:
        lines.append(f"Контекст от стратега: {summary}")

    lines += [
        "",
        "Напиши три варианта текста оффера:",
        f"1. Push-уведомление — строго до {_LIMITS['push']} символов",
        f"2. Баннер в приложении — строго до {_LIMITS['banner']} символов",
        f"3. SMS — строго до {_LIMITS['sms']} символов",
    ]

    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            result = {}
            for channel, limit in _LIMITS.items():
                text = str(data.get(channel, "")).strip()
                # Обрезаем если Claude превысил лимит
                if len(text) > limit:
                    text = text[:limit - 1].rstrip() + "…"
                result[channel] = text
            return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    # Фолбэк: возвращаем сырой текст в push, остальные пустые
    return {"push": raw.strip()[:_LIMITS["push"]], "banner": "", "sms": ""}


def run_copywriter(strategy: dict, client) -> dict:
    prompt = _build_prompt(strategy)

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=512,
    )

    return _parse_response(response.choices[0].message.content)


if __name__ == "__main__":
    # --- тест _build_prompt ---
    strategy_new = {
        "target_segment": "new", "cashback_new_pct": 15, "cashback_old_pct": 10,
        "recommended_cap": 4500, "key_risks": [], "summary": "Фокус на новых клиентов.",
    }
    p = _build_prompt(strategy_new)
    assert "15%" in p
    assert "первой покупк" in p
    assert "60 символов" in p

    strategy_existing = {**strategy_new, "target_segment": "existing"}
    p2 = _build_prompt(strategy_existing)
    assert "10%" in p2
    assert "лояльност" in p2

    strategy_both = {**strategy_new, "target_segment": "both"}
    p3 = _build_prompt(strategy_both)
    assert "до 15%" in p3

    # --- тест _parse_response: валидный JSON ---
    raw_ok = json.dumps({
        "push":   "Кэшбек 15% на первую покупку у партнёра!",
        "banner": "Получите 15% кэшбэка на первую покупку у партнёра. Максимум 4 500 руб.",
        "sms":    "Альфа-Банк: кэшбэк 15% на первую покупку у партнёра. Лимит 4 500 руб. Совершите покупку до конца месяца.",
    }, ensure_ascii=False)
    r = _parse_response(raw_ok)
    assert len(r["push"])   <= _LIMITS["push"]
    assert len(r["banner"]) <= _LIMITS["banner"]
    assert len(r["sms"])    <= _LIMITS["sms"]
    assert "15%" in r["push"]

    # --- тест _parse_response: превышение лимита → обрезка ---
    raw_long = json.dumps({
        "push":   "А" * 100,
        "banner": "Б" * 50,
        "sms":    "В" * 30,
    }, ensure_ascii=False)
    r2 = _parse_response(raw_long)
    assert len(r2["push"]) == _LIMITS["push"], f"push не обрезан: {len(r2['push'])}"

    # --- тест _parse_response: сломанный JSON → фолбэк ---
    r3 = _parse_response("Не могу ответить в JSON.")
    assert r3["banner"] == ""
    assert len(r3["push"]) <= _LIMITS["push"]

    print("Все тесты прошли.")
