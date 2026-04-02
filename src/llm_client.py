from openai import OpenAI
import json
import time
from datetime import datetime
from pathlib import Path
import os

api_key = os.environ.get("LITELLM_API_KEY")
if not api_key:
    raise ValueError("Не найден LITELLM_API_KEY! Установите переменную окружения.")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://api.duckduck.cloud/v1",
)

NUM_RUNS = 1

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
        model="iairlab/qwen3.5-35b-a3b",
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
        max_tokens=5000,  # максимум токенов в ответе
        n=1,  # сколько вариантов ответа вернуть
        timeout=300,
        response_format={"type": "json_object"}  # принудительный JSON
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return {
        "run_index": run_index,
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
        "method": parsed.get("method"),
        "reasoning": parsed.get("reasoning"),
        "description": parsed.get("description"),
        "code": parsed.get("code"),
    }


def run_experiment(dsl_filepath, output_dir: Path, num_runs=NUM_RUNS):
    dsl_content = load_dsl(dsl_filepath)
    results = []

    for i in range(num_runs):
        print(f"Запуск {i + 1}/{num_runs}...")
        try:
            result = generate_heuristic(dsl_content, run_index=i + 1)
            results.append(result)
            print(
                f"Правило: {result['method']} | "
                f"Вход: {result['tokens']['prompt']} tok | "
                f"Выход: {result['tokens']['completion']} tok | "
                f"Всего: {result['tokens']['total']} tok"
            )
        except Exception as e:
            print(f"Ошибка на запуске {i + 1}: {e}")
            results.append({"run_index": i + 1, "error": str(e)})

        time.sleep(1)

    json_path = output_dir / (Path(dsl_filepath).stem + "_results.json")
    output = {
        "dsl_file": Path(dsl_filepath).name,
        "num_runs": num_runs,
        "runs": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nГотово. Сохранено: {json_path}")

    return results


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DSL_DIR = BASE_DIR/"data"/"references"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    OUTPUT_DIR = BASE_DIR/"experiments"/"test run"/timestamp
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dsl_files = sorted([
        f for f in DSL_DIR.glob("*.json")
        if not f.name.endswith("_results.json")
    ])

    if not dsl_files:
        print(f"JSON-файлы не найдены в {DSL_DIR}")
    else:
        print(f"Найдено файлов: {len(dsl_files)}")
        print(f"Результаты будут сохранены в: {OUTPUT_DIR}\n")

        all_results = {}

        for dsl_file in dsl_files:
            print(f"{'-' * 55}")
            print(f"Обработка: {dsl_file.name}")
            runs = run_experiment(
                dsl_filepath=dsl_file,
                output_dir=OUTPUT_DIR
            )
            all_results[dsl_file.name] = runs

        summary_path = OUTPUT_DIR/"all_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nОбщий будут сохранены в: {summary_path}")

        print(f"\n{'-' * 55}")
        print("Итого по методам:")
        for filename, runs in all_results.items():
            rules = [r.get("method", "N/A") for r in runs if "error" not in r]
            print(f"  {filename}: {rules}")
