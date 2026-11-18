"""Render the paper's tables (Table 1-5) as Markdown and LaTeX."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

from ..config import ensure_dir, project_path

TABLE_COLUMNS = {
    "table1": ["Acc", "F1", "AUC", "P@10", "NDCG@10", "Cold-Start AUC"],
    "table2": ["Acc", "F1", "AUC", "P@10", "NDCG@10", "Cold-Start AUC"],
    "table3": ["Acc", "F1_neg", "F1_neu", "F1_pos", "Kappa"],
    "table4": ["HR@5", "HR@10", "NDCG@5", "NDCG@10"],
    "table5": ["BLEU-4", "ROUGE-L", "Human-Coherence"],
    "table_external": ["HR@5", "HR@10", "NDCG@5", "NDCG@10"],
}

TABLE_TITLES = {
    "table1": "Main results on Yelp (business-level ranking)",
    "table2": "Ablation study on Yelp",
    "table3": "Implicit sentiment classification on EmoMeta",
    "table4": "Ranking transfer on Taobao-MM",
    "table5": "Generated explanation quality and human evaluation",
    "table_external": "External baselines on Yelp (user-level ranking)",
}


def _numeric(value) -> bool:
    return isinstance(value, (int, float))


def _rank(rows: List[Tuple[str, Dict]], column: str) -> Dict[str, int]:
    values = [(name, row.get(column)) for name, row in rows if _numeric(row.get(column))]
    values.sort(key=lambda item: -float(item[1]))
    ranking = {}
    for position, (name, _) in enumerate(values):
        ranking[name] = position
    return ranking


def render_markdown(title: str, columns: Sequence[str], rows: List[Tuple[str, Dict]], decimals: int = 4) -> str:
    rankings = {column: _rank(rows, column) for column in columns}
    header = "| Method | " + " | ".join(columns) + " |"
    separator = "|" + "---|" * (len(columns) + 1)
    lines = ["### " + title, "", header, separator]
    for name, row in rows:
        cells = [name]
        for column in columns:
            value = row.get(column)
            if not _numeric(value):
                cells.append("-" if value is None else str(value))
                continue
            text = "{:.{d}f}".format(float(value), d=decimals)
            position = rankings[column].get(name)
            if position == 0:
                text = "**{}**".format(text)
            elif position == 1:
                text = "_{}_".format(text)
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_latex(title: str, columns: Sequence[str], rows: List[Tuple[str, Dict]], decimals: int = 4) -> str:
    rankings = {column: _rank(rows, column) for column in columns}
    header = " & ".join(["Method"] + list(columns)) + " \\\\"
    lines = ["% " + title, "\\begin{tabular}{l" + "c" * len(columns) + "}", "\\toprule", header, "\\midrule"]
    for name, row in rows:
        cells = [name]
        for column in columns:
            value = row.get(column)
            if not _numeric(value):
                cells.append("-" if value is None else str(value))
                continue
            text = "{:.{d}f}".format(float(value), d=decimals)
            position = rankings[column].get(name)
            if position == 0:
                text = "\\textbf{" + text + "}"
            elif position == 1:
                text = "\\underline{" + text + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_tables(results: Dict[str, Dict[str, Dict]], decimals: int = 4) -> Dict[str, Dict[str, str]]:
    tables: Dict[str, Dict[str, str]] = {}
    for key, columns in TABLE_COLUMNS.items():
        variants = results.get(key, {})
        if not variants:
            continue
        rows = [(name, values) for name, values in variants.items()]
        tables[key] = {
            "markdown": render_markdown(TABLE_TITLES[key], columns, rows, decimals),
            "latex": render_latex(TABLE_TITLES[key], columns, rows, decimals),
        }
    return tables


def write_tables(cfg: Dict, results: Dict[str, Dict[str, Dict]]) -> Dict[str, str]:
    tables = build_tables(results)
    output_dir = ensure_dir(os.path.join(project_path(cfg, "output_dir"), "tables"))
    markdown_path = os.path.join(output_dir, "tables.md")
    latex_path = os.path.join(output_dir, "tables.tex")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        for key in TABLE_COLUMNS:
            if key in tables:
                handle.write(tables[key]["markdown"] + "\n\n")
    with open(latex_path, "w", encoding="utf-8") as handle:
        for key in TABLE_COLUMNS:
            if key in tables:
                handle.write(tables[key]["latex"] + "\n")
    return {"markdown": markdown_path, "latex": latex_path}
