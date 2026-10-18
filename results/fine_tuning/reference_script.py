#!/bin/python3
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # sense pantalla: funciona també al cluster
import matplotlib.pyplot as plt

# ---- Dades de referència: Wilma (nodo.q), 10 èpoques
REF_NODE = "Wilma, \\textit{nodo.q}"
REF_SEQ = 40.737795
REF_TIME = {2: 22.542747, 4: 11.730357, 6: 8.308845, 12: 5.263955}

RE_TIME = re.compile(r"Goodbye!\s*\(([\d.,]+)\s*sec\)")
RE_THREADS = re.compile(r"^\s*Threads:\s*(\d+)")
RE_LABEL = re.compile(r"^\s*LABEL=(.+)$")
RE_HITS = re.compile(r"Total encerts\s*=\s*(\d+)")
RE_SEQ = re.compile(r"stats for '\.?/?exec_seq")


def num(s):
    return float(s.replace(",", "."))


def mean_std(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fc(x, d=2):
    """Número amb coma decimal (per a LaTeX en català)."""
    return "{:.{}f}".format(x, d).replace(".", ",")


def sg(x, d=2):
    """Número amb coma decimal i signe menys tipogràfic."""
    return ("$-$" if x < -0.5 * 10 ** (-d) else "") + fc(abs(x), d)


def parse_text(txt):
    """Retorna una llista de dicts, una per cada execució trobada al text."""
    runs, last = [], None
    threads = label = None
    hits = None
    for line in txt.splitlines():
        m = RE_THREADS.match(line)
        if m:
            threads = m.group(1)
            continue
        m = RE_LABEL.match(line)
        if m:
            label = m.group(1).strip()
            continue
        if line.startswith("Training"):
            hits = None
            continue
        m = RE_HITS.search(line)
        if m:
            hits = int(m.group(1))
            continue
        m = RE_TIME.search(line)
        if m:
            last = dict(label=label or "", threads=threads, time=num(m.group(1)), hits=hits)
            runs.append(last)
            threads = None  # la propera execució ha de portar la seva pròpia línia "Threads:"
            continue
        # el bloc de perf (després del Goodbye) diu si era l'executable seqüencial
        if RE_SEQ.search(line) and last is not None and last["threads"] is None:
            last["threads"] = "seq"
    return runs


def parse(folder):
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "slurm-*.out"))):
        with open(path, errors="replace") as f:
            found = parse_text(f.read())
        if not found:
            print("[avís] {}: no hi ha la línia 'Goodbye!' (execució fallida?)".format(path), file=sys.stderr)
            continue
        for r in found:
            if r["threads"] is None:
                print("[avís] {}: no sé si és seqüencial ni quants fils té".format(path), file=sys.stderr)
                continue
            runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta amb els fitxers slurm-*.out")
    ap.add_argument("--label", help="només les execucions amb aquesta etiqueta (LABEL=...)")
    ap.add_argument("--out", default="comparacio", help="prefix dels fitxers de sortida")
    args = ap.parse_args()

    runs = parse(args.carpeta)
    if not runs:
        sys.exit("No s'ha trobat cap execució vàlida.")
    labels = sorted({r["label"] for r in runs if r["label"]})
    if args.label is not None:
        runs = [r for r in runs if r["label"] == args.label]
    elif len(labels) > 1:
        print("[avís] Hi ha diverses etiquetes ({}): es barregen. Usa --label per triar-ne una."
              .format(", ".join(labels)), file=sys.stderr)

    own = defaultdict(list)  # fils -> [temps]
    seq = []
    ignorades = Counter()
    hits_par = Counter()
    for r in runs:
        if r["threads"].lower() == "seq":
            seq.append(r["time"])
        elif int(r["threads"]) in REF_TIME:
            own[int(r["threads"])].append(r["time"])
            if r["hits"] is not None:
                hits_par[r["hits"]] += 1
        else:
            ignorades[r["threads"]] += 1
    if ignorades:
        print("[info] Execucions ignorades (sense referència): " +
              ", ".join("{} fils ({})".format(k, v) for k, v in sorted(ignorades.items(), key=lambda x: int(x[0]))))
    if len(hits_par) > 1:
        print("[avís] Els encerts no coincideixen entre execucions {}: possible carrera de dades."
              .format(dict(hits_par)), file=sys.stderr)

    # ---- estadístiques
    Tq = Sq = None
    if seq:
        Tq, Sq = mean_std(seq)
    rows = []
    for p in sorted(REF_TIME):
        ts = own.get(p)
        if not ts:
            print("[avís] No hi ha execucions amb {} fils.".format(p), file=sys.stderr)
            continue
        m, s = mean_std(ts)
        ref = REF_TIME[p]
        S_ref = REF_SEQ / ref
        S = REF_SEQ / m
        row = dict(p=p, n=len(ts), ref=ref, mean=m, std=s, diff=m - ref, pct=(m - ref) / ref * 100,
                   below=sum(1 for t in ts if t < ref), times=ts,
                   S_ref=S_ref, E_ref=S_ref / p, S=S, Serr=S * s / m, E=S / p)
        if Tq:
            Sp = Tq / m
            row.update(S_seq=Sp, Serr_seq=Sp * ((Sq / Tq) ** 2 + (s / m) ** 2) ** 0.5, E_seq=Sp / p)
        rows.append(row)
    if not rows:
        sys.exit("Cap execució amb 2, 4, 6 o 12 fils.")

    # ---- CSV
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "n", "ref_s", "mean_s", "std_s", "diff_s", "diff_pct", "runs_below_ref",
                    "speedup_ref", "speedup", "speedup_err", "efficiency_ref", "efficiency"])
        for r in rows:
            w.writerow([r["p"], r["n"], r["ref"], r["mean"], r["std"], r["diff"], r["pct"], r["below"],
                        r["S_ref"], r["S"], r["Serr"], r["E_ref"], r["E"]])

    # ---- resum per pantalla
    print("Referència: Wilma nodo.q, seqüencial {:.3f} s".format(REF_SEQ))
    print("{:>5}{:>4}{:>11}{:>20}{:>9}{:>9}{:>10}{:>10}{:>10}".format(
        "fils", "n", "ref (s)", "propi (s)", "dif %", "< ref", "S ref", "S propi", "E propi"))
    for r in rows:
        print("{:>5}{:>4}{:>11.3f}{:>11.3f} ± {:<6.3f}{:>9.2f}{:>6}/{:<3}{:>10.2f}{:>10.2f}{:>10.2f}".format(
            r["p"], r["n"], r["ref"], r["mean"], r["std"], r["pct"], r["below"], r["n"],
            r["S_ref"], r["S"], r["E"]))

    # ---- taules LaTeX
    ns = {r["n"] for r in rows}
    ntxt = "%d execucions" % ns.pop() if len(ns) == 1 else "diverses execucions"
    L = []

    L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline")
    L.append("\\textbf{Fils} & \\textbf{Ref. (s)} & \\textbf{Propi (s)} & \\textbf{Dif. (s)} & "
             "\\textbf{Dif. (\\%)} & \\textbf{Execs. $<$ ref.} \\\\ \\hline")
    for r in rows:
        L.append("%d & %s & %s $\\pm$ %s & %s & %s & %d/%d \\\\ \\hline" % (
            r["p"], fc(r["ref"], 3), fc(r["mean"], 3), fc(r["std"], 3),
            sg(r["diff"], 3), sg(r["pct"], 2), r["below"], r["n"]))
    L.append("\\end{tabular}")
    L.append("\\caption{Comparació del temps d'execució propi (mitjana $\\pm$ desviació estàndard de %s) "
             "amb el temps de referència (%s, 10 èpoques). Els valors negatius indiquen un temps "
             "inferior al de referència.}" % (ntxt, REF_NODE))
    L.append("\\label{tab:comp_temps}\n\\end{table}\n")

    L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline")
    L.append("\\textbf{Fils} & \\textbf{Speedup ref.} & \\textbf{Speedup propi} & "
             "\\textbf{Eficiència ref.} & \\textbf{Eficiència pròpia} \\\\ \\hline")
    for r in rows:
        L.append("%d & %s & %s $\\pm$ %s & %s & %s \\\\ \\hline" % (
            r["p"], fc(r["S_ref"]), fc(r["S"]), fc(r["Serr"]), fc(r["E_ref"]), fc(r["E"])))
    L.append("\\end{tabular}")
    L.append("\\caption{Speedup i eficiència propis i de referència (%s), calculats amb el temps "
             "seqüencial de referència (%s s).}" % (REF_NODE, fc(REF_SEQ, 3)))
    L.append("\\label{tab:comp_speedup}\n\\end{table}\n")

    if Tq:
        L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline")
        L.append("\\textbf{Fils} & \\textbf{Speedup} & \\textbf{Eficiència} \\\\ \\hline")
        for r in rows:
            L.append("%d & %s $\\pm$ %s & %s \\\\ \\hline" % (
                r["p"], fc(r["S_seq"]), fc(r["Serr_seq"]), fc(r["E_seq"])))
        L.append("\\end{tabular}")
        L.append("\\caption{Speedup i eficiència propis respecte al seqüencial propi "
                 "(%s $\\pm$ %s s, %d execucions).}" % (fc(Tq, 3), fc(Sq, 3), len(seq)))
        L.append("\\label{tab:comp_speedup_propi}\n\\end{table}\n")

    with open(args.out + "_taules.tex", "w") as f:
        f.write("\n".join(L))

    # ---- gràfics
    ths = [r["p"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4))

    ax = axes[0]
    ax.plot(ths, [r["ref"] for r in rows], "s--", color="gray", label="Referència")
    ax.errorbar(ths, [r["mean"] for r in rows], yerr=[r["std"] for r in rows], marker="o", capsize=3, label="Propi")
    for r in rows:
        ax.scatter([r["p"]] * len(r["times"]), r["times"], s=10, alpha=0.3, color="C0")
    ax.set_ylabel("Temps (s)")
    ax.set_title("Temps d'execució")

    ax = axes[1]
    ax.plot(ths, [r["S_ref"] for r in rows], "s--", color="gray", label="Referència")
    ax.errorbar(ths, [r["S"] for r in rows], yerr=[r["Serr"] for r in rows], marker="o", capsize=3, label="Propi")
    ax.plot(ths, ths, "k:", label="ideal")
    ax.set_ylabel("Speedup (respecte al seqüencial de referència)")
    ax.set_title("Speedup")

    ax = axes[2]
    ax.plot(ths, [r["E_ref"] for r in rows], "s--", color="gray", label="Referència")
    ax.plot(ths, [r["E"] for r in rows], "o-", label="Propi")
    ax.axhline(1, ls=":", color="k", label="ideal")
    ax.set_ylabel("Eficiència")
    ax.set_title("Eficiència")

    for ax in axes:
        ax.set_xlabel("Fils")
        ax.set_xticks(ths)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=200)
    fig.savefig(args.out + ".pdf")

    print("\nGenerats: {0}_taules.tex, {0}.csv, {0}.png, {0}.pdf".format(args.out))


if __name__ == "__main__":
    main()
