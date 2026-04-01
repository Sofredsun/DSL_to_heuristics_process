from openai import OpenAI
import json
import time
from pathlib import Path
import os

llm_client = OpenAI(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url="https://api.duckduck.cloud/v1",
)

SYSTEM_PROMPT = """Ты эксперт-исследователь в области комбинаторных задач построения расписаний в условиях ограниченных ресурсов.
Тебе передаются метахарактеристики проекта в формате DSL описания. 
На их основе выбери и реализуй подходящий эвристический алгоритм. 
Эвристика на вход должна принимать такие аргументы: 
- jobs: list[dict] (список работ)
- resources: list[dict] (список ресурсов)
Отвечай строго в JSON формате:
{
    "code": "полный Python код функции solve_scheduling",
    "method": "название выбранного метода",
    "reasoning": "почему выбрано именно это правило",
    "description": "краткое описание реализации"
}"""


def load_dsl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_heuristic(dsl_content, run_index):
    response = llm_client.chat.completions.create(
        model="gpt-5.4-nano-2026-03-17",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Построй эвристику для этого проекта:\n{json.dumps(dsl_content, ensure_ascii=False, indent=2)}"
            }
        ],
        temperature=0.7,  # случайность (0.0 — детерминированно, 2.0 — максимум)
        max_tokens=4000,  # максимум токенов в ответе
        n=1,  # сколько вариантов ответа вернуть
        timeout=60,
        response_format={"type": "json_object"}  # принудительный JSON
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    parsed["run_index"] = run_index
    parsed["tokens_used"] = response.usage.total_tokens
    return parsed


def run_experiment(dsl_filepath, num_runs=10):
    dsl_content = load_dsl(dsl_filepath)
    results = []

    for i in range(num_runs):
        print(f"Запуск {i + 1}/{num_runs}...")
        try:
            result = generate_heuristic(dsl_content, run_index=i + 1)
            results.append(result)
            print(f"Правило: {result.get('method')} | "
                  f"Токены: {result.get('tokens_used')}")
        except Exception as e:
            print(f"Ошибка на запуске {i + 1}: {e}")
            results.append({"run_index": i + 1, "error": str(e)})

        time.sleep(1)

    # Сохранение результатов
    output_path = dsl_filepath.replace(".json", "_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Сохранено: {output_path}")
    return results


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    results = run_experiment(
        dsl_filepath=" ",
        num_runs=10
    )

    # Краткая статистика по выбранным правилам
    rules = [r.get("method", "N/A") for r in results if "error" not in r]
    print(f"\nВыбранные правила за 10 запусков: {rules}")
