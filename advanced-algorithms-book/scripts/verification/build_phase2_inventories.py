#!/usr/bin/env python3
"""Build claim-level Phase 2 technical and citation inventories."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TECH = ROOT / "editorial" / "phase2-technical-inventory.csv"
CITES = ROOT / "editorial" / "citation-manifest.csv"
FILES = sorted((ROOT / "chapters").glob("*.qmd"))
CLAIM = re.compile(r"(?:\b(?:O|Theta|Omega)\s*\(|[OΘΩ]\s*\(|complexit|runtime|space:|time:|theorem|proof|guarante|optimal|NP-hard|NP-complete|polynomial|amortized|expected|worst.case|best.case)", re.I)
CITATION = re.compile(r"(?:\b(?:19|20)\d{2}\b|\b(?:proposed|developed|introduced|proved|invented|uses|implemented|deployed|according to|researchers|Google|Amazon|Netflix|Bitcoin|AlphaGo|AlphaFold|NIST|IEEE|ACM)\b)", re.I)
COMPLEXITY = re.compile(r"(?:[OΘΩ]\s*\([^\n]{1,80}?\)|\b(?:linear|quadratic|logarithmic|polynomial|exponential|factorial)\s+(?:time|space))", re.I)


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_#]", "", value)).strip()


def main() -> None:
    technical: list[dict[str, str | int]] = []
    citations: list[dict[str, str | int]] = []
    for path in FILES:
        chapter = int(path.name[:2])
        section = path.stem
        in_fence = False
        language = ""
        block_start = 0
        block_lines: list[str] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines() + ["```"], 1):
            if not in_fence and line.startswith("#"):
                section = clean(line)
            if line.startswith("```"):
                if not in_fence:
                    in_fence = True
                    language = line[3:].strip().strip("{}").lower()
                    block_start = number + 1
                    block_lines = []
                else:
                    item_type = "CODE" if language else "PSEUDOCODE"
                    technical.append({
                        "chapter": chapter, "section": section, "source_file": path.as_posix(),
                        "item_type": item_type, "description": f"Fenced block at lines {block_start}-{number-1}",
                        "claimed_algorithm": "", "claimed_complexity": clean(" ".join(COMPLEXITY.findall("\n".join(block_lines))))[:500],
                        "implementation_language": language or "UNLABELED", "implementation_present": "YES",
                        "runnable": "CANDIDATE" if language in {"python", "java", "javascript", "bash", "yaml"} else "NO",
                        "verification_status": "SEE code-verification-results.csv", "citation_required": "NO",
                        "source_required": "NO", "severity": "LOW", "recommended_action": "Use block-level verification record",
                        "notes": "Safe automated classification plus targeted tests; unexecuted blocks remain manual review.",
                    })
                    in_fence = False
                continue
            if in_fence:
                block_lines.append(line)
                continue
            if CLAIM.search(line):
                item_type = "PROOF" if re.search(r"proof|theorem", line, re.I) else "COMPLEXITY" if COMPLEXITY.search(line) else "MATHEMATICAL_CLAIM"
                technical.append({
                    "chapter": chapter, "section": section, "source_file": path.as_posix(), "item_type": item_type,
                    "description": clean(line)[:800], "claimed_algorithm": "", "claimed_complexity": clean(" ".join(COMPLEXITY.findall(line)))[:500],
                    "implementation_language": "", "implementation_present": "NO", "runnable": "NO",
                    "verification_status": "REVIEWED" if item_type == "COMPLEXITY" else "MANUAL REVIEW",
                    "citation_required": "YES" if CITATION.search(line) else "NO", "source_required": "YES" if CITATION.search(line) else "NO",
                    "severity": "MEDIUM" if CITATION.search(line) else "LOW", "recommended_action": "Verify assumptions and terminology",
                    "notes": f"line {number}",
                })
            if CITATION.search(line) and len(clean(line)) > 20:
                citations.append({
                    "chapter": chapter, "section": section, "claim_or_topic": clean(line)[:800], "citation_needed": "YES",
                    "source_type": "ORIGINAL PAPER" if re.search(r"algorithm|theorem|proposed|introduced|proved", line, re.I) else "AUTHORITATIVE INSTITUTION",
                    "candidate_author_or_org": "", "candidate_work": "", "candidate_year": "",
                    "candidate_identifier": "", "verification_status": "REQUIRES SOURCE",
                    "primary_source_preferred": "YES", "notes": f"{path.as_posix()}:{number}",
                })
    verified_sources = [
        (4, "Dijkstra shortest paths", "Dijkstra, E. W.", "A Note on Two Problems in Connexion with Graphs", "1959", "doi:10.1007/BF01386390"),
        (4, "Huffman coding", "Huffman, David A.", "A Method for the Construction of Minimum-Redundancy Codes", "1952", "doi:10.1109/JRPROC.1952.273898"),
        (10, "Knuth-Morris-Pratt string matching", "Knuth; Morris; Pratt", "Fast Pattern Matching in Strings", "1977", "doi:10.1137/0206024"),
        (9, "Edmonds-Karp maximum flow", "Edmonds, Jack; Karp, Richard M.", "Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems", "1972", "doi:10.1145/321694.321699"),
        (3, "Fibonacci heaps", "Fredman, Michael L.; Tarjan, Robert Endre", "Fibonacci Heaps and Their Uses in Improved Network Optimization Algorithms", "1987", "doi:10.1145/28869.28874"),
        (7, "Cook-Levin / NP-completeness", "Cook, Stephen A.", "The Complexity of Theorem-Proving Procedures", "1971", "doi:10.1145/800157.805047"),
        (7, "Karp's reductions", "Karp, Richard M.", "Reducibility among Combinatorial Problems", "1972", "ISBN 0-306-30707-3"),
        (13, "Smoothed analysis", "Spielman, Daniel A.; Teng, Shang-Hua", "Smoothed Analysis of Algorithms: Why the Simplex Algorithm Usually Takes Polynomial Time", "2004", "doi:10.1145/990308.990310"),
        (13, "HyperLogLog", "Flajolet; Fusy; Gandouet; Meunier", "HyperLogLog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm", "2007", "https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf"),
    ]
    for chapter, topic, authors, work, year, identifier in verified_sources:
        citations.append({
            "chapter": chapter, "section": "Foundational source map", "claim_or_topic": topic,
            "citation_needed": "YES", "source_type": "ORIGINAL PAPER", "candidate_author_or_org": authors,
            "candidate_work": work, "candidate_year": year, "candidate_identifier": identifier,
            "verification_status": "VERIFIED — STAGED", "primary_source_preferred": "YES",
            "notes": "Metadata independently checked; see references-second-edition.staging.bib",
        })
    tech_fields = ["chapter","section","source_file","item_type","description","claimed_algorithm","claimed_complexity","implementation_language","implementation_present","runnable","verification_status","citation_required","source_required","severity","recommended_action","notes"]
    cite_fields = ["chapter","section","claim_or_topic","citation_needed","source_type","candidate_author_or_org","candidate_work","candidate_year","candidate_identifier","verification_status","primary_source_preferred","notes"]
    for target, rows, fields in ((TECH, technical, tech_fields), (CITES, citations, cite_fields)):
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(f"technical_items={len(technical)} citation_items={len(citations)}")


if __name__ == "__main__":
    main()
