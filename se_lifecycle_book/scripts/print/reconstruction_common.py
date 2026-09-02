#!/usr/bin/env python3
"""Source and PDF parity helpers for the neutral complete-edition build."""
from __future__ import annotations
import csv, hashlib, re, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = [ROOT / "chapters" / f"{n:02d}-{slug}.qmd" for n, slug in enumerate([
 "introduction","requirements-engineering","systems-modeling","software-architecture","ui-ux",
 "agile-methodologies","version-control","testing-quality","cicd","data-management","cloud-services",
 "security","maintenance-evolution","ethics-professionalism","final-project-integration"], 1)]
PDF = ROOT / "output/reconstruction/The-Complete-Software-Engineering-Lifecycle-NEUTRAL-REVIEW.pdf"
EDITORIAL = ROOT / "editorial"

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def slugify(s: str) -> str:
    s = re.sub(r"[`*_~]", "", s).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")
def plain(s: str) -> str:
    s = re.sub(r"!\[([^]]*)\]\([^)]*\)(?:\{[^}]*\})?", r"\1", s)
    s = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", s)
    return re.sub(r"[`*_~]", "", s)
def norm(s: str) -> str: return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

@dataclass
class Block:
    line:int; language:str; text:str; section:str

def scan(path: Path):
    lines=path.read_text(encoding="utf-8").splitlines(); headings=[]; blocks=[]; images=[]; tables=[]
    in_fence=False; fence=""; start=0; lang=""; buf=[]; section=""; in_yaml=False; divs=rawhtml=rawlatex=0
    i=0
    while i < len(lines):
        line=lines[i]; ln=i+1
        if ln==1 and line.strip()=="---": in_yaml=True; i+=1; continue
        if in_yaml:
            if line.strip()=="---": in_yaml=False
            i+=1; continue
        if in_fence:
            if re.match(r"^\s*"+re.escape(fence)+r"\s*$", line):
                blocks.append(Block(start,lang,"\n".join(buf),section)); in_fence=False; buf=[]
            else: buf.append(line)
            i+=1; continue
        fm=re.match(r"^\s*(`{3,}|~{3,})\s*([^`]*)$",line)
        if fm:
            in_fence=True; fence=fm.group(1)[0]*len(fm.group(1)); start=ln; lang=fm.group(2).strip().strip("{}").split()[0] if fm.group(2).strip() else ""; i+=1; continue
        hm=re.match(r"^(#{1,4})\s+(.+?)\s*$",line)
        if hm:
            text=re.sub(r"\s*\{[^}]*\}\s*$","",hm.group(2)); attrs=re.search(r"\{#([^ }]+)",line)
            ident=attrs.group(1) if attrs else slugify(plain(text)); headings.append((len(hm.group(1)),plain(text),ident,ln)); section=plain(text); i+=1; continue
        if re.match(r"^\s*:::",line): divs+=1
        if re.match(r"^\s*<[^>]+>",line): rawhtml+=1
        if re.match(r"^\s*\\(?:begin|end|newpage|clearpage)",line): rawlatex+=1
        for im in re.finditer(r"!\[([^]]*)\]\(([^ )]+)(?:\s+\"([^\"]*)\")?\)(\{[^}]*\})?",line): images.append((ln,section,*im.groups()))
        if "|" in line and i+1<len(lines) and re.match(r"^\s*\|?\s*:?-{3,}",lines[i+1]):
            startrow=i; rows=[line]; i+=1
            while i<len(lines) and "|" in lines[i] and lines[i].strip(): rows.append(lines[i]); i+=1
            cols=max(len([x for x in r.strip().strip('|').split('|')]) for r in rows); tables.append((startrow+1,section,len(rows)-1,cols)); continue
        # Four-space indented verbatim outside lists. Rare, but required by manifest.
        if re.match(r"^( {4}|\t)\S",line) and (i==0 or not re.match(r"^\s*(?:[-*+] |\d+[.)] )",lines[i-1])):
            ib=[re.sub(r"^( {4}|\t)","",line)]; st=ln; i+=1
            while i<len(lines) and (not lines[i].strip() or re.match(r"^( {4}|\t)",lines[i])):
                ib.append(re.sub(r"^( {4}|\t)","",lines[i])); i+=1
            blocks.append(Block(st,"", "\n".join(ib).rstrip(),section)); continue
        i+=1
    return lines,headings,blocks,images,tables,(rawhtml,rawlatex,divs)

def classify(language,text,context=""):
    l=language.lower().lstrip('.'); t=text.strip(); low=t.lower()
    if l in {"python","py","javascript","js","java","c","cpp","typescript","tsx","jsx","ruby","go","rust","php","gherkin"}: return "PROGRAM_CODE"
    if l in {"yaml","yml","dockerfile","terraform","hcl","toml","ini","properties","nginx","xml"}: return "CONFIGURATION"
    if l in {"json","csv"}: return "DATA"
    if l in {"sql"}: return "PROGRAM_CODE"
    if l in {"pseudocode","textual"} or re.search(r"\bBEGIN\b.*\bEND\b",t,re.S): return "PSEUDOCODE"
    if re.search(r"[┌┐└┘├┤┬┴┼│─═╔╗╚╝]|(?:^|\n)\s*[+|].*[+|]",t): return "ASCII_DIAGRAM"
    if l in {"bash","sh","shell","console","zsh"}:
        return "TERMINAL_SESSION" if re.search(r"(?m)^\s*[$#>]\s+",t) or "output" in context.lower() else "TERMINAL_COMMAND"
    if re.search(r"(?m)^\s*[$>]\s+",t): return "TERMINAL_SESSION"
    if re.search(r"(?m)^(?:PASS|FAIL|ERROR|INFO|WARNING|coverage|tests?:)",t,re.I): return "PROGRAM_OUTPUT"
    return "PLAIN_VERBATIM" if t else "UNKNOWN"

def pdf_text(pdf=PDF):
    if not pdf.exists(): raise SystemExit(f"PDF not found: {pdf}")
    p=subprocess.run(["pdftotext","-layout",str(pdf),"-"],text=True,capture_output=True,check=True)
    return p.stdout
def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

if __name__ == "__main__": print("library")
