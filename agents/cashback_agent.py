import os
import anthropic


SYSTEM_PROMPT = """Ты — аналитик партнёрских кэшбек-кампаний Альфа-Банка.
Твоя задача — на основе числовых данных о транзакциях дать чёткую, практичную рекомендацию менеджеру.

Структура ответа:
1. **Вывод** — какой сегмент (новички, старички или оба) рекомендуешь для следующей кампании и почему.
2. **Ключевые наблюдения** — 2–3 факта из данных, которые повлияли на решение.
3. **На что обратить внимание** — риски или аномалии (высокий % кэпнутых, низкий оборот и т.д.).
4. **Текст оффера** — готовый короткий оффер для выбранного сегмента (1–2 предложения).

Отвечай по-русски, лаконично и по делу."""


def build_prompt(stats: dict) -> str:
    s = stats
    city_block = ""
    if s.get("top_cities"):
        city_lines = "\n".join(
            f"  - {city}: оборот {amt:,.0f} руб, кэшбек {cb:,.0f} руб"
            for city, amt, cb in s["top_cities"]
        )
        city_block = f"\nТоп-города по обороту:\n{city_lines}"

    return f"""Данные кэшбек-кампании партнёра:

Общая статистика:
- Всего транзакций: {s['total']}
- Уникальных клиентов: {s['unique_clients']}
- Оборот: {s['total_amount']:,.0f} руб
- Кэшбек выплачен: {s['total_cashback']:,.0f} руб
- Нагрузка кэшбека на оборот: {s['cashback_load']:.1f}%
- Средний кэшбек на клиента: {s['avg_cashback_per_client']:,.0f} руб

Сегменты:
- Новички ({s['new_pct']}%): {s['new_count']} транзакций, оборот {s['new_amount']:,.0f} руб
- Старички ({s['old_pct']}%): {s['old_count']} транзакций, оборот {s['old_amount']:,.0f} руб
- Кэпнутых транзакций: {s['capped_count']} ({s['capped_share']:.1f}%){city_block}

Дай рекомендацию по структуре ответа из системного промпта."""


def run_analysis(stats: dict) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(stats)}],
    )
    return message.content[0].text
