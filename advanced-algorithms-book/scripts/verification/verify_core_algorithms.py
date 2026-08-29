#!/usr/bin/env python3
"""Behavioral checks for representative core algorithms printed in the book."""

from __future__ import annotations

import ast
import contextlib
import io
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_block(chapter: str, symbol: str) -> dict:
    path = ROOT / "chapters" / chapter
    lines = path.read_text(encoding="utf-8").splitlines()
    in_python = False
    body: list[str] = []
    for line in lines + ["```"]:
        if not in_python and re.match(r"^```(?:python|\{[^}]*\.python(?:\s|\}))", line):
            in_python = True
            body = []
        elif in_python and line.startswith("```"):
            code = "\n".join(body)
            try:
                tree = ast.parse(code)
            except SyntaxError:
                in_python = False
                continue
            names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            if symbol in names:
                kept = [
                    node for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                    or (isinstance(node, ast.Assign) and not any(isinstance(x, ast.Call) for x in ast.walk(node.value)))
                ]
                namespace: dict = {"__name__": "book_example"}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), namespace)
                return namespace
            in_python = False
        elif in_python:
            body.append(line)
    raise LookupError(f"{symbol} not found in {chapter}")


def check(name: str, operation) -> tuple[str, str]:
    try:
        operation()
        return name, "PASS"
    except Exception as exc:  # reports failures without hiding later checks
        return name, f"FAIL: {type(exc).__name__}: {exc}"


def sorting_checks() -> None:
    ns = load_block("02-Divide-and-Conquer.qmd", "merge_sort")
    for values in ([], [1], [2, 1], [3, 1, 2, 1], [-2, 5, 0, -2], list(range(10)), list(range(9, -1, -1))):
        assert ns["merge_sort"](values) == sorted(values)
    ns = load_block("02-Divide-and-Conquer.qmd", "quicksort")
    for values in ([], [1], [2, 1], [3, 1, 2, 1], [-2, 5, 0, -2], list(range(10)), list(range(9, -1, -1))):
        actual = values.copy()
        ns["quicksort"](actual, 0, len(actual) - 1)
        assert actual == sorted(values)


def selection_checks() -> None:
    for chapter in ("01-introduction.qmd", "06-Randomized-Algorithms.qmd"):
        ns = load_block(chapter, "quickselect")
        values = [7, -1, 3, 3, 9, 0]
        for k, expected in enumerate(sorted(values)):
            assert ns["quickselect"](values.copy(), k) == expected


def heap_checks() -> None:
    ns = load_block("03-Data-Structures-for-Efficiency.qmd", "MaxHeap")
    heap = ns["MaxHeap"]()
    for value in [3, 1, 7, 7, -2, 4]:
        heap.insert(value)
    assert [heap.extract_max() for _ in range(6)] == [7, 7, 4, 3, 1, -2]


def union_find_checks() -> None:
    ns = load_block("03-Data-Structures-for-Efficiency.qmd", "UnionFind")
    uf = ns["UnionFind"](6)
    uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    assert uf.connected(0, 2) and not uf.connected(0, 4)


def activity_checks() -> None:
    ns = load_block("04-Greedy-Algorithms.qmd", "activity_selection")
    chosen = ns["activity_selection"]([(1, 4, "A"), (3, 5, "B"), (0, 6, "C"), (5, 7, "D"), (8, 9, "E"), (5, 9, "F")])
    assert chosen == ["A", "D", "E"]


def knapsack_checks() -> None:
    ns = load_block("05-Dynamic-Programming.qmd", "knapsack_01")
    result = ns["knapsack_01"]([10, 20, 30], [60, 100, 120], 50)
    assert result == 220
    assert ns["knapsack_01"]([], [], 10) == 0


def primality_checks() -> None:
    ns = load_block("06-Randomized-Algorithms.qmd", "miller_rabin_primality")
    random.seed(2026)
    for value in [2, 3, 5, 97, 997]: assert ns["miller_rabin_primality"](value)
    for value in [0, 1, 4, 9, 21, 341, 561]: assert not ns["miller_rabin_primality"](value)


def string_checks() -> None:
    cases = [("", ""), ("abc", ""), ("abc", "d"), ("aaaa", "aa"), ("ababcabc", "abc")]
    for symbol in ("naive_search", "kmp_search"):
        ns = load_block("10-String-Processing-Algorithms.qmd", symbol)
        if symbol == "kmp_search":
            ns.update(load_block("10-String-Processing-Algorithms.qmd", "compute_failure_function"))
            ns[symbol].__globals__.update(ns)
        for text, pattern in cases:
            expected = [] if pattern == "" else [i for i in range(len(text) + 1) if text.startswith(pattern, i)]
            assert ns[symbol](text, pattern) == expected


def numerical_checks() -> None:
    ns = load_block("11-Numerical-Algorithms.qmd", "fft_recursive")
    np = ns["np"]
    for values in ([1], [1, 2], [1, 2, 3, 4], [0, -1, 2, 3, 4, 5, 6, 7]):
        assert np.allclose(ns["fft_recursive"](np.array(values)), np.fft.fft(values))
    ns = load_block("11-Numerical-Algorithms.qmd", "strassen_matmul")
    ns.update(load_block("11-Numerical-Algorithms.qmd", "matmul_naive"))
    ns["np"] = np
    ns["strassen_matmul"].__globals__.update(ns)
    ns["matmul_naive"].__globals__.update(ns)
    for size in (1, 2, 4, 65):
        a = np.arange(size * size).reshape(size, size)
        b = np.flip(a, axis=0)
        assert np.allclose(ns["strassen_matmul"](a, b), a @ b)


def range_structure_checks() -> None:
    ns = load_block("12-Advanced-Data-Structures.qmd", "SegmentTree")
    tree = ns["SegmentTree"]([1, 3, 5, 7, 9, 11])
    assert tree.query(1, 3) == 15
    tree.update(2, -5)
    assert tree.query(0, 5) == 26
    ns = load_block("12-Advanced-Data-Structures.qmd", "FenwickTree")
    bit = ns["FenwickTree"].from_array([1, 3, 5, 7, 9, 11])
    assert bit.range_sum(1, 3) == 15
    bit.update(2, -10)
    assert bit.range_sum(0, 5) == 26


def flow_checks() -> None:
    ns = load_block("09-Advanced-Graph-Algorithms.qmd", "EdmondsKarp")
    flow = ns["EdmondsKarp"](6)
    for edge in [(0,1,16),(0,2,13),(1,2,10),(2,1,4),(1,3,12),(3,2,9),(2,4,14),(4,3,7),(3,5,20),(4,5,4)]:
        flow.add_edge(*edge)
    assert flow.max_flow(0, 5) == 23
    empty = ns["EdmondsKarp"](2)
    assert empty.max_flow(0, 1) == 0


def smoothed_demo_checks() -> None:
    ns = load_block("13-Research-Industry-Applications.qmd", "demonstrate_smoothed_analysis")
    with contextlib.redirect_stdout(io.StringIO()):
        ns["demonstrate_smoothed_analysis"]()


def main() -> int:
    checks = [sorting_checks, selection_checks, heap_checks, union_find_checks,
              activity_checks, knapsack_checks, primality_checks, string_checks,
              numerical_checks, range_structure_checks, flow_checks,
              smoothed_demo_checks]
    results = [check(fn.__name__, fn) for fn in checks]
    for name, status in results:
        print(f"{name}: {status}")
    passed = sum(status == "PASS" for _, status in results)
    print(f"groups={len(results)} passed={passed} failed={len(results)-passed}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
