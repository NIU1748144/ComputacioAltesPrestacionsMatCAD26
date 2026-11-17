#!/usr/bin/env python3
"""
taules_mpi.py - Taules (temps mitjà, speedup i eficiència) i gràfics de la versió
MPI a partir dels fitxers slurm-<id>.out.

Format dels .out que llegeix
    Processos MPI: 6          nombre de processos de l'execució
    Particio: nodo.q          partició (s'avisa si no és nodo.q o si se'n barregen)
    Goodbye! (18.70 sec)      temps de l'execució
    Total encerts = 928 (/946)
  (el bloc d'èpoques i qualsevol altra línia s'ignoren). Un mateix fitxer pot
  contenir diverses execucions seguides. Opcionalment, una línia  LABEL=<text>
  permet filtrar amb --label.

Càlcul del speedup i l'eficiència (el mateix criteri que a la part d'OpenMP)
    speedup = T_seq_ref / T_propi          eficiència = speedup / processos
amb T_seq_ref = 40,737795 s (Wilma, nodo.q, 10 èpoques). Es pot canviar amb --ref-seq.

Temps de referència MPI (opcional; el PDF de la pràctica només en dona d'OpenMP)
    --ref "2=22.5,4=11.7,6=8.3,12=5.2"      (decimals amb punt, parelles procs=temps)
Amb --ref s'afegeixen les taules de comparació amb la referència.

Ús
    python3 taules_mpi.py carpeta_amb_els_out/ [--ref "2=..,4=.."] [--label TEXT] [--out mpi]

Sortides: <out>_taules.tex, <out>.csv, <out>.png i <out>.pdf
Només necessita Python 3 i matplotlib.
"""
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

REF_NODE = "Wilma, \\textit{new-nodo.q}"
REF_PART = "new-nodo.q"
REF_SEQ = 12.0959

RE_TIME = re.compile(r"Goodbye!\s*\(([\d.,]+)\s*sec\)")
RE_PROCS = re.compile(r"^\s*Processos MPI:\s*(\d+)")
RE_PART = re.compile(r"^\s*Particio:\s*(\S+)")
RE_LABEL = re.compile(r"^\s*LABEL=(.+)$")
RE_HITS = re.compile(r"Total encerts\s*=\s*(\d+)")
RE_LOSS = re.compile(r"Loss:\s*([-+\d.,eE]+)")


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


def parse_ref(s):
    d = {}
    try:
        for item in s.split(","):
            k, v = item.split("=")
            d[int(k)] = float(v)
    except ValueError:
        sys.exit('Format de --ref incorrecte. Exemple: --ref "2=22.5,4=11.7" (decimals amb punt).')
    return d


def parse_text(txt):
    """Retorna una llista de dicts, una per cada execució trobada al text."""
    runs = []
    procs = part = label = None
    hits, losses = None, []
    for line in txt.splitlines():
        m = RE_PROCS.match(line)
        if m:
            procs = int(m.group(1))
            continue
        m = RE_PART.match(line)
        if m:
            part = m.group(1)
            continue
        m = RE_LABEL.match(line)
        if m:
            label = m.group(1).strip()
            continue
        if line.startswith("Training"):
            hits, losses = None, []
            continue
        m = RE_LOSS.search(line)
        if m:
            losses.append(num(m.group(1)))
            continue
        m = RE_HITS.search(line)
        if m:
            hits = int(m.group(1))
            continue
        m = RE_TIME.search(line)
        if m:
            runs.append(dict(label=label or "", procs=procs, part=part, time=num(m.group(1)),
                             hits=hits, loss=losses[-1] if losses else None))
            procs = part = None  # la propera execució ha de portar les seves pròpies línies
    return runs


def parse(folder):
    runs, sense_procs = [], 0
    for path in sorted(glob.glob(os.path.join(folder, "slurm-*.out"))):
        with open(path, errors="replace") as f:
            found = parse_text(f.read())
        if not found:
            print("[avís] {}: no hi ha la línia 'Goodbye!' (execució fallida?)".format(path), file=sys.stderr)
            continue
        for r in found:
            if r["procs"] is None:
                sense_procs += 1
            else:
                runs.append(r)
    if sense_procs:
        print("[info] {} execucions sense la línia 'Processos MPI:' ignorades.".format(sense_procs))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta amb els fitxers slurm-*.out")
    ap.add_argument("--ref", help='temps de referència MPI, p. ex. "2=22.5,4=11.7,6=8.3,12=5.2"')
    ap.add_argument("--ref-seq", type=float, default=REF_SEQ,
                    help="temps seqüencial de referència (per defecte %(default)s s)")
    ap.add_argument("--label", help="només les execucions amb aquesta etiqueta (LABEL=...)")
    ap.add_argument("--out", default="mpi", help="prefix dels fitxers de sortida")
    args = ap.parse_args()
    ref_seq = args.ref_seq
    ref_time = parse_ref(args.ref) if args.ref else {}

    runs = parse(args.carpeta)
    if not runs:
        sys.exit("No s'ha trobat cap execució vàlida.")
    labels = sorted({r["label"] for r in runs if r["label"]})
    if args.label is not None:
        runs = [r for r in runs if r["label"] == args.label]
        if not runs:
            sys.exit("Cap execució amb aquesta etiqueta.")
    elif len(labels) > 1:
        print("[avís] Hi ha diverses etiquetes ({}): es barregen. Usa --label per triar-ne una."
              .format(", ".join(labels)), file=sys.stderr)

    # ---- comprovacions
    parts = Counter(r["part"] for r in runs if r["part"])
    if len(parts) > 1:
        print("[avís] Hi ha execucions a particions diferents {}: els temps no són comparables."
              .format(dict(parts)), file=sys.stderr)
    for part in parts:
        if part != REF_PART:
            print("[avís] Hi ha execucions a la partició '{}': el temps seqüencial de referència és de {}."
                  .format(part, REF_PART), file=sys.stderr)

    by_p = defaultdict(list)
    for r in runs:
        by_p[r["procs"]].append(r)
    for p in sorted(by_p):
        hits = {r["hits"] for r in by_p[p] if r["hits"] is not None}
        loss = {r["loss"] for r in by_p[p] if r["loss"] is not None}
        if len(hits) > 1 or len(loss) > 1:
            print("[avís] Amb {} processos, el resultat canvia entre execucions (encerts {}, loss final {}): "
                  "possible indeterminisme.".format(p, sorted(hits), sorted(loss)), file=sys.stderr)

    # ---- estadístiques
    rows = []
    for p in sorted(by_p):
        ts = [r["time"] for r in by_p[p]]
        m, s = mean_std(ts)
        S = ref_seq / m
        hits = sorted({r["hits"] for r in by_p[p] if r["hits"] is not None})
        row = dict(p=p, n=len(ts), mean=m, std=s, times=ts, S=S, Serr=S * s / m, E=S / p, hits=hits)
        if p in ref_time:
            ref = ref_time[p]
            S_ref = ref_seq / ref
            row.update(ref=ref, diff=m - ref, pct=(m - ref) / ref * 100,
                       below=sum(1 for t in ts if t < ref), S_ref=S_ref, E_ref=S_ref / p)
        rows.append(row)
    con_ref = [r for r in rows if "ref" in r]

    # ---- CSV
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["procs", "n", "mean_s", "std_s", "speedup", "speedup_err", "efficiency",
                    "ref_s", "diff_s", "diff_pct", "runs_below_ref", "speedup_ref", "efficiency_ref"])
        for r in rows:
            w.writerow([r["p"], r["n"], r["mean"], r["std"], r["S"], r["Serr"], r["E"],
                        r.get("ref"), r.get("diff"), r.get("pct"), r.get("below"),
                        r.get("S_ref"), r.get("E_ref")])

    # ---- resum per pantalla
    print("Speedup i eficiència respecte al seqüencial de referència: {:.3f} s".format(ref_seq))
    print("{:>6}{:>4}{:>20}{:>9}{:>9}   {}".format("procs", "n", "temps (s)", "speedup", "efic.", "encerts"))
    for r in rows:
        print("{:>6}{:>4}{:>11.3f} ± {:<6.3f}{:>9.2f}{:>9.2f}   {}".format(
            r["p"], r["n"], r["mean"], r["std"], r["S"], r["E"], ", ".join(str(h) for h in r["hits"])))

    # ---- taules LaTeX
    ns = {r["n"] for r in rows}
    ntxt = "%d execucions" % ns.pop() if len(ns) == 1 else "diverses execucions"
    L = []

    L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|}\n\\hline")
    L.append("\\textbf{Processos MPI} & \\textbf{Temps (s)} & \\textbf{Speedup} & "
             "\\textbf{Eficiència} \\\\ \\hline")
    for r in rows:
        L.append("%d & %s $\\pm$ %s & %s $\\pm$ %s & %s \\\\ \\hline" % (
            r["p"], fc(r["mean"], 3), fc(r["std"], 3), fc(r["S"]), fc(r["Serr"]), fc(r["E"])))
    L.append("\\end{tabular}")
    L.append("\\caption{Temps d'execució (mitjana $\\pm$ desviació estàndard de %s), speedup i "
             "eficiència de la versió MPI. El speedup i l'eficiència es calculen amb el temps seqüencial "
             "de referència (%s s, %s).}" % (ntxt, fc(ref_seq, 3), REF_NODE))
    L.append("\\label{tab:mpi_resultats}\n\\end{table}\n")

    if con_ref:
        L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline")
        L.append("\\textbf{Processos} & \\textbf{Ref. (s)} & \\textbf{Propi (s)} & \\textbf{Dif. (s)} & "
                 "\\textbf{Dif. (\\%)} & \\textbf{Execs. $<$ ref.} \\\\ \\hline")
        for r in con_ref:
            L.append("%d & %s & %s $\\pm$ %s & %s & %s & %d/%d \\\\ \\hline" % (
                r["p"], fc(r["ref"], 3), fc(r["mean"], 3), fc(r["std"], 3),
                sg(r["diff"], 3), sg(r["pct"], 2), r["below"], r["n"]))
        L.append("\\end{tabular}")
        L.append("\\caption{Comparació del temps d'execució MPI propi (mitjana $\\pm$ desviació estàndard) "
                 "amb el temps de referència. Els valors negatius indiquen un temps inferior al de "
                 "referència.}")
        L.append("\\label{tab:mpi_comp_temps}\n\\end{table}\n")

        L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline")
        L.append("\\textbf{Processos} & \\textbf{Speedup ref.} & \\textbf{Speedup propi} & "
                 "\\textbf{Eficiència ref.} & \\textbf{Eficiència pròpia} \\\\ \\hline")
        for r in con_ref:
            L.append("%d & %s & %s $\\pm$ %s & %s & %s \\\\ \\hline" % (
                r["p"], fc(r["S_ref"]), fc(r["S"]), fc(r["Serr"]), fc(r["E_ref"]), fc(r["E"])))
        L.append("\\end{tabular}")
        L.append("\\caption{Speedup i eficiència MPI propis i de referència, calculats amb el temps "
                 "seqüencial de referència (%s s).}" % fc(ref_seq, 3))
        L.append("\\label{tab:mpi_comp_speedup}\n\\end{table}\n")

    with open(args.out + "_taules.tex", "w") as f:
        f.write("\n".join(L))

    # ---- gràfics
    ps = [r["p"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4))

    ax = axes[0]
    ax.axhline(ref_seq, ls=":", color="gray", label="seqüencial ref.")
    if con_ref:
        ax.plot([r["p"] for r in con_ref], [r["ref"] for r in con_ref], "s--", color="gray", label="Referència")
    ax.errorbar(ps, [r["mean"] for r in rows], yerr=[r["std"] for r in rows], marker="o", capsize=3, label="Propi")
    for r in rows:
        ax.scatter([r["p"]] * len(r["times"]), r["times"], s=10, alpha=0.3, color="C0")
    ax.set_ylabel("Temps (s)")
    ax.set_title("Temps d'execució")

    ax = axes[1]
    if con_ref:
        ax.plot([r["p"] for r in con_ref], [r["S_ref"] for r in con_ref], "s--", color="gray", label="Referència")
    ax.errorbar(ps, [r["S"] for r in rows], yerr=[r["Serr"] for r in rows], marker="o", capsize=3, label="Propi")
    ax.plot(ps, ps, "k:", label="ideal")
    ax.set_ylabel("Speedup (respecte al seqüencial de referència)")
    ax.set_title("Speedup")

    ax = axes[2]
    if con_ref:
        ax.plot([r["p"] for r in con_ref], [r["E_ref"] for r in con_ref], "s--", color="gray", label="Referència")
    ax.plot(ps, [r["E"] for r in rows], "o-", label="Propi")
    ax.axhline(1, ls=":", color="k", label="ideal")
    ax.set_ylabel("Eficiència")
    ax.set_title("Eficiència")

    for ax in axes:
        ax.set_xlabel("Processos MPI")
        ax.set_xticks(ps)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=200)
    fig.savefig(args.out + ".pdf")

    print("\nGenerats: {0}_taules.tex, {0}.csv, {0}.png, {0}.pdf".format(args.out))


if __name__ == "__main__":
    main()
