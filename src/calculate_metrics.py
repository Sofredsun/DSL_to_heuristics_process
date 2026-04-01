import json
import networkx as nx
import os
import pandas as pd
from pathlib import Path


def build_graph_from_json(filepath):
    """
    Функция для парсинга разных типов RCPSP
    и построения направленного графа NetworkX.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()

    case_info = data.get("case_info", {})
    case_type = case_info.get("type", os.path.basename(filepath))

    # Case 5
    if 'jobs' in data and len(data['jobs']) > 0 and 'activities' in data['jobs'][0]:
        for proj in data['jobs']:
            prefix = f"p{proj['project_id']}_"  # Чтобы узлы разных проектов не слипались
            for act in proj['activities']:
                u = prefix + str(act['id'])
                G.add_node(u)
                for succ in act.get('successors', []):
                    v = prefix + str(succ)
                    G.add_edge(u, v)
        return case_type, G

    # Cases 1-4
    jobs = data.get('jobs', [])
    for job in jobs:
        u = str(job['id'])
        G.add_node(u)

        # Case 1
        if 'successors' in job:
            for succ in job['successors']:
                G.add_edge(u, str(succ))

        # Cases 3, 4
        elif 'precedences' in job and 'time_successors' in job['precedences']:
            for succ in job['precedences']['time_successors']:
                G.add_edge(u, str(succ))

        # Case 2
        elif 'predecessors' in job:
            for pred in job['predecessors']:
                G.add_edge(str(pred), u)

    return case_type, G


def calculate_graph_metrics(G, per_project=False, project_subgraphs=None):
    """
    per_project=True + project_subgraphs — для Case 5, где несколько проектов
    слиты в один граф. BFS-уровни и OS считаются по каждому подграфу отдельно,
    затем агрегируются.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_edges == 0:
        return {
            "nodes": num_nodes,
            "edges": 0,
            "connected_components": num_nodes,
            "average_degree": 0.0,
            "density": 0.0,
            "cpl": "N/A",
            "relative_cpl": "N/A",
            "order_strength": 0.0,
            "num_levels_bfs": 1,
            "avg_parallelism": "N/A",
            "max_parallelism": 1,
            "num_roots": "N/A",
            "num_leaves": "N/A",
        }

    # Базовые характеристики
    connected_components = nx.number_weakly_connected_components(G)
    average_degree = sum(dict(G.degree()).values()) / num_nodes
    density = nx.density(G)

    # Критический путь
    cpl = len(nx.dag_longest_path(G))
    relative_cpl = round(cpl / num_nodes, 4)

    # Сила упорядоченности
    tc = nx.transitive_closure(G)
    os_edges = tc.number_of_edges()

    max_pairs = num_nodes * (num_nodes - 1)
    order_strength = round(os_edges / max_pairs, 4) if max_pairs > 0 else 0

    # BFS-параллелизм
    # Если граф многопроектный (Case 5) - рассчет по каждому подграфу,
    # иначе корни слитого графа дадут некорректную картину ширин
    if per_project and project_subgraphs:
        all_widths = []
        for sg in project_subgraphs:
            sg_roots = [n for n, d in sg.in_degree() if d == 0]
            if sg_roots:
                layers = list(nx.bfs_layers(sg, sg_roots))
                all_widths.extend(len(layer) for layer in layers)
        bfs_widths = all_widths
        num_levels = len(bfs_widths)
    else:
        roots = [n for n, d in G.in_degree() if d == 0]
        if roots:
            layers = list(nx.bfs_layers(G, roots))
            bfs_widths = [len(layer) for layer in layers]
            num_levels = len(bfs_widths)
        else:
            bfs_widths = []
            num_levels = 0

    avg_parallelism = round(sum(bfs_widths) / num_levels, 3) if num_levels > 0 else 0
    max_parallelism = max(bfs_widths) if bfs_widths else 0

    num_roots = sum(1 for n, d in G.in_degree() if d == 0)
    num_leaves = sum(1 for n, d in G.out_degree() if d == 0)

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "connected_components": connected_components,
        "average_degree": round(average_degree, 3),
        "density": round(density, 4),
        "cpl": cpl,
        "relative_cpl": relative_cpl,
        "order_strength": order_strength,
        "num_levels_bfs": num_levels,
        "avg_parallelism": avg_parallelism,
        "max_parallelism": max_parallelism,
        "num_roots": num_roots,
        "num_leaves": num_leaves,
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    output_dir = BASE_DIR/"data"/"references"/"case_metrics.xlsx"
    data_dir = BASE_DIR/"data"/"processed"
    files = [
        (str(data_dir/"Case1_PSPLIB_j301_1.json"), False),
        (str(data_dir/"Case2_sch50.json"), False),
        (str(data_dir/"Case3_nonrenewable.json"), False),
        (str(data_dir/"Case4_renewable.json"), False),
        (str(data_dir/"Case5_mp_j90_a2_nr1.json"), True)
    ]

    all_results = []

    print(
        f"{'Кейс':<20} | {'Узлы':<5} | {'Ребра':<5} | {'CPL':<5} | {'OS':<6} | {'AvgPar':<8} | {'Корни':<6} | {'Листья':<6}")
    print("-" * 80)

    for filepath, is_multiproject in files:
        if not os.path.exists(filepath):
            print(f"Файл не найден: {filepath}")
            continue

        case_type, graph = build_graph_from_json(filepath)

        project_subgraphs = None
        if is_multiproject:
            components = list(nx.weakly_connected_components(graph))
            project_subgraphs = [
                graph.subgraph(c).copy() for c in components
            ]

        metrics = calculate_graph_metrics(
            graph,
            per_project=is_multiproject,
            project_subgraphs=project_subgraphs,
        )

        row = {"Case ID": case_type, **metrics}
        all_results.append(row)

        print(
            f"{case_type[:18]:<20} | "
            f"{metrics['nodes']:<5} | "
            f"{metrics['edges']:<5} | "
            f"{metrics['cpl']:<5} | "
            f"{metrics['order_strength']:<6} | "
            f"{metrics['avg_parallelism']:<8} | "
            f"{metrics['num_roots']:<6} | "
            f"{metrics['num_leaves']:<6}"
        )

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(output_dir, index=False)
        print(f"\nСохранено: {output_dir}")
