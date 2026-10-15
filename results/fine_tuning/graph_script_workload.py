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
from matplotlib.ticker import NullFormatter

RE_TIME = re.compile(r"Goodbye!\s*\(([\d.,]+)\s*sec\)")
RE_THREADS = re.compile(r"^\s*Threads:\s*(\d+)")
RE_LABEL = re.compile(r"^\s*LABEL=(.+)$")
RE_LOSS = re.compile(r"Loss:\s*([-+\d.,eE]+)")
RE_HITS = re.compile(r"Total encerts\s*=\s*(\d+)")
RE_SEQ = re.compile(r"stats for '\.?/?exec_seq")
RE_ID = re.compile(r"slurm-(\d+)\.out$")
RE_POL = re.compile(r"^(static|dynamic|guided)[_,:\-\s]*(\d+)?$", re.I)
POL_ORDER = {"static": 0, "dynamic": 1, "guided": 2}


def num(s):
    return float(s.replace(",", "."))


def mean_std(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fc(x, d=2):
    """Número amb coma decimal (per a LaTeX en català)."""
    return f"{x:.{d}f}".replace(".", ",")


def set_log2_x(ax):
    """Escala logarítmica en base 2 a l'eix X, compatible amb matplotlib antic (<3.3) i nou."""
    try:
        ax.set_xscale("log", base=2)
    except (ValueError, TypeError):
        ax.set_xscale("log", basex=2)


def load_map(path):
    m = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m[row["id"].strip()] = (row["threads"].strip(), (row.get("label") or "").strip())
    return m


def parse_text(txt):
    """Retorna una llista de dicts, una per cada execució trobada al text."""
    runs, last = [], None
    threads = label = None
    losses, hits = [], None
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
            losses, hits = [], None
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
            last = dict(label=label or "", threads=threads, time=num(m.group(1)),
                        loss=losses[-1] if losses else None, hits=hits)
            runs.append(last)
            threads = None  # la propera execució ha de portar la seva pròpia línia "Threads:"
            continue
        # el bloc de perf (després del Goodbye) diu si era l'executable seqüencial
        if RE_SEQ.search(line) and last is not None and last["threads"] is None:
            last["threads"] = "seq"
    return runs


def parse(folder, mapping):
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "slurm-*.out"))):
        with open(path, errors="replace") as f:
            found = parse_text(f.read())
        if not found:
            print(f"[avís] {path}: no hi ha la línia 'Goodbye!' (execució fallida?)", file=sys.stderr)
            continue
        jid = RE_ID.search(path).group(1)
        for r in found:
            if jid in mapping:
                r["threads"], lab = mapping[jid]
                if lab:
                    r["label"] = lab
            if r["threads"] is None:
                print(f"[avís] {path}: no sé si és seqüencial ni quants fils té (usa --map)", file=sys.stderr)
                continue
            r["label"] = r["label"] or "default"
            runs.append(r)
    return runs


# --------------------------------------------------------------------------
# Mode 2: polítiques de repartiment de feina
# --------------------------------------------------------------------------
def parse_policy(label):
    m = RE_POL.match(label.strip())
    if not m:
        return None
    return m.group(1).lower(), (int(m.group(2)) if m.group(2) else None)


def welch_z(a, b):
    """z aproximat de la diferència de mitjanes (error estàndard de cada mitjana)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    se = (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5
    return (st.mean(a) - st.mean(b)) / se if se > 0 else 0.0


def policy_report(groups, threads, out):
    cfg = {}  # (política, chunk) -> [temps]
    for (lab, p), ts in groups.items():
        if p != threads:
            continue
        pc = parse_policy(lab)
        if pc is None:
            print(f"[avís] Etiqueta ignorada (no és política[_chunk]): {lab}", file=sys.stderr)
            continue
        pol, chunk = pc
        if pol != "static" and chunk is None:
            chunk = 1  # chunk per defecte d'OpenMP per a dynamic i guided
        cfg.setdefault((pol, chunk), []).extend(ts)
    if not cfg:
        sys.exit(f"No hi ha execucions amb {threads} fils i etiqueta de política (static, dynamic_4, ...).")

    ref_key = ("static", None)
    if ref_key not in cfg:
        ref_key = min(cfg, key=lambda k: st.mean(cfg[k]))
        print(f"[avís] No hi ha 'static' de referència: es compara amb la millor configuració {ref_key}.",
              file=sys.stderr)
    ref = cfg[ref_key]
    t_ref = st.mean(ref)
    best = min(cfg, key=lambda k: st.mean(cfg[k]))
    keys = sorted(cfg, key=lambda k: (POL_ORDER[k[0]], k[1] if k[1] is not None else 0))

    # ---- CSV + resum per pantalla
    print(f"Polítiques amb {threads} fils (referència: {ref_key[0]}"
          f"{'' if ref_key[1] is None else ',' + str(ref_key[1])}, {t_ref:.3f} s)")
    print(f"{'política':<10}{'chunk':>6}{'n':>4}{'temps (s)':>20}{'dif. %':>9}{'z':>7}")
    with open(out + "_politiques.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "chunk", "n", "mean_s", "std_s", "diff_pct", "z"])
        for k in keys:
            m, s = mean_std(cfg[k])
            d = (m - t_ref) / t_ref * 100
            z = 0.0 if k == ref_key else welch_z(cfg[k], ref)
            w.writerow([k[0], k[1], len(cfg[k]), m, s, d, z])
            print(f"{k[0]:<10}{'-' if k[1] is None else k[1]:>6}{len(cfg[k]):>4}"
                  f"{m:>12.3f} ± {s:<6.3f}{d:>9.2f}{z:>7.1f}")

    # ---- taula LaTeX
    lines = []
    for pol in POL_ORDER:
        ks = [k for k in keys if k[0] == pol]
        for i, k in enumerate(ks):
            m, s = mean_std(cfg[k])
            d = (m - t_ref) / t_ref * 100
            sig = "$^{*}$" if (k != ref_key and abs(welch_z(cfg[k], ref)) > 2) else ""
            tcell = f"{fc(m, 3)} $\\pm$ {fc(s, 3)}"
            dcell = ("$-$" if d < -0.005 else "") + fc(abs(d), 2) + sig
            if k == best:
                tcell, dcell = "\\textbf{%s}" % tcell, "\\textbf{%s}" % dcell
            chunk = "---" if k[1] is None else str(k[1])
            first = "\\multirow{%d}{*}{\\textit{%s}}" % (len(ks), pol) if i == 0 else ""
            end = " \\\\ \\hline" if i == len(ks) - 1 else " \\\\ \\cline{2-4}"
            lines.append(f"{first} & {chunk} & {tcell} & {dcell}{end}")
    refname = "\\textit{static}" if ref_key == ("static", None) else "la millor configuració"
    with open(out + "_politiques.tex", "w") as f:
        f.write("% cal \\usepackage{multirow}\n")
        f.write("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|}\n\\hline\n")
        f.write("\\textbf{Política} & \\textbf{Chunk} & \\textbf{Temps (s)} & "
                "\\textbf{Dif. respecte " + refname + " (\\%)} \\\\ \\hline\n")
        f.write("\n".join(lines) + "\n")
        f.write("\\end{tabular}\n")
        f.write("\\par\\vspace{2pt}{\\footnotesize $^{*}$ Diferència significativa respecte a la "
                "referència ($|z|>2$). En negreta, la configuració amb millor temps.}\n")
        f.write("\\caption{Temps d'execució segons la política i la mida de \\textit{chunk} amb "
                "%d fils (mitjana $\\pm$ desviació estàndard).}\n" % threads)
        f.write("\\label{tab:politiques}\n\\end{table}\n")

    # ---- gràfic: temps vs chunk
    chunks = sorted({k[1] for k in keys if k[1] is not None})
    if chunks:
        fig, ax = plt.subplots(figsize=(6, 4))
        for pol in ("dynamic", "guided", "static"):
            ks = [k for k in keys if k[0] == pol and k[1] is not None]
            if not ks:
                continue
            ms = [mean_std(cfg[k]) for k in ks]
            eb = ax.errorbar([k[1] for k in ks], [m for m, _ in ms], yerr=[s for _, s in ms],
                             marker="o", capsize=3, label=pol)
            for k in ks:
                ax.scatter([k[1]] * len(cfg[k]), cfg[k], s=10, alpha=0.3, color=eb[0].get_color())
        if ("static", None) in cfg:
            m, s = mean_std(cfg[("static", None)])
            ax.axhline(m, ls="--", color="gray", label="static (per defecte)")
            ax.axhspan(m - s, m + s, color="gray", alpha=0.15)
        set_log2_x(ax)
        ax.set_xticks(chunks)
        ax.set_xticklabels([str(c) for c in chunks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Mida de chunk")
        ax.set_ylabel("Temps (s)")
        ax.set_title(f"Polítiques de repartiment ({threads} fils)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out + "_politiques.png", dpi=200)
        fig.savefig(out + "_politiques.pdf")
    print(f"\nGenerats: {out}_politiques.tex, {out}_politiques.csv"
          + (f", {out}_politiques.png, {out}_politiques.pdf" if chunks else ""))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta amb els fitxers slurm-*.out")
    ap.add_argument("--map", help="CSV id,threads[,label] per als .out que no s'identifiquen sols")
    ap.add_argument("--out", default="resultat", help="prefix dels fitxers de sortida")
    ap.add_argument("--politiques", type=int, metavar="N",
                    help="mode polítiques de repartiment: taula i gràfic amb N fils")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="tolerància relativa per comprovar la loss final respecte al seqüencial")
    args = ap.parse_args()

    runs = parse(args.carpeta, load_map(args.map) if args.map else {})
    if not runs:
        sys.exit("No s'ha trobat cap execució vàlida.")

    # ---- baseline seqüencial i comprovació de correcció (loss final i encerts)
    seq = [r for r in runs if r["threads"].lower() == "seq"]
    base_loss = [r["loss"] for r in seq if r["loss"] is not None]
    base_hits = [r["hits"] for r in seq if r["hits"] is not None]
    L0 = st.mean(base_loss) if base_loss else None
    H0 = Counter(base_hits).most_common(1)[0][0] if base_hits else None
    for r in runs:
        if L0 is not None and r["loss"] is not None and abs(r["loss"] - L0) > args.tol * max(abs(L0), 1e-12):
            print(f"[avís] Loss final diferent del seqüencial: {r['label']}, {r['threads']} fils: "
                  f"{r['loss']} (seq: {L0:.6f})", file=sys.stderr)
        if H0 is not None and r["hits"] is not None and r["hits"] != H0:
            print(f"[avís] Encerts diferents del seqüencial: {r['label']}, {r['threads']} fils: "
                  f"{r['hits']} (seq: {H0})", file=sys.stderr)

    groups = defaultdict(list)  # (etiqueta, fils) -> [temps]
    for r in runs:
        if r["threads"].lower() != "seq":
            groups[(r["label"], int(r["threads"]))].append(r["time"])

    if args.politiques:
        policy_report(groups, args.politiques, args.out)
        return

    # ---- mode escalat amb els fils
    T0 = S0 = None
    if seq:
        T0, S0 = mean_std([r["time"] for r in seq])
    else:
        ones = [t for (lab, p), ts in groups.items() if p == 1 for t in ts]
        if ones:
            T0, S0 = mean_std(ones)
            print("[avís] No hi ha execucions seqüencials: speedup respecte a 1 fil.", file=sys.stderr)
        else:
            print("[avís] No hi ha baseline: no es calcularan speedup ni eficiència.", file=sys.stderr)

    rows = []
    for (lab, p), ts in sorted(groups.items()):
        m, s = mean_std(ts)
        row = dict(label=lab, p=p, n=len(ts), mean=m, std=s, median=st.median(ts), times=ts,
                   S=None, Serr=None, E=None)
        if T0:
            S = T0 / m
            Serr = S * ((S0 / T0) ** 2 + (s / m) ** 2) ** 0.5
            row.update(S=S, Serr=Serr, E=S / p)
        rows.append(row)

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "threads", "n", "mean_s", "std_s", "median_s", "speedup", "speedup_err", "efficiency"])
        for r in rows:
            w.writerow([r["label"], r["p"], r["n"], r["mean"], r["std"], r["median"], r["S"], r["Serr"], r["E"]])

    if T0:
        print(f"Seqüencial: {T0:.3f} ± {S0:.3f} s ({len(seq)} execucions)")
    print(f"{'etiqueta':<14}{'fils':>5}{'n':>4}{'temps (s)':>20}{'speedup':>10}{'efic.':>8}")
    for r in rows:
        sp = f"{r['S']:.2f}" if r["S"] is not None else "-"
        ef = f"{r['E']:.2f}" if r["E"] is not None else "-"
        print(f"{r['label']:<14}{r['p']:>5}{r['n']:>4}{r['mean']:>12.3f} ± {r['std']:<6.3f}{sp:>10}{ef:>8}")

    labels = sorted({r["label"] for r in rows})
    ths = sorted({r["p"] for r in rows})
    ncol = 3 if T0 else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4), squeeze=False)
    axes = axes[0]

    def by_label(lab):
        return [r for r in rows if r["label"] == lab]

    ax = axes[0]
    for lab in labels:
        rs = by_label(lab)
        eb = ax.errorbar([r["p"] for r in rs], [r["mean"] for r in rs], yerr=[r["std"] for r in rs],
                         marker="o", capsize=3, label=lab)
        for r in rs:
            ax.scatter([r["p"]] * len(r["times"]), r["times"], s=10, alpha=0.3, color=eb[0].get_color())
    if T0:
        ax.axhline(T0, ls="--", color="gray", label="seqüencial")
        ax.axhspan(T0 - S0, T0 + S0, color="gray", alpha=0.15)
    ax.set_ylabel("Temps (s)")
    ax.set_title("Temps d'execució")

    if T0:
        ax = axes[1]
        for lab in labels:
            rs = by_label(lab)
            ax.errorbar([r["p"] for r in rs], [r["S"] for r in rs], yerr=[r["Serr"] for r in rs],
                        marker="o", capsize=3, label=lab)
        ax.plot(ths, ths, "k:", label="ideal")
        ax.set_ylabel("Speedup")
        ax.set_title("Speedup")

        ax = axes[2]
        for lab in labels:
            rs = by_label(lab)
            ax.errorbar([r["p"] for r in rs], [r["E"] for r in rs],
                        yerr=[r["Serr"] / r["p"] for r in rs], marker="o", capsize=3, label=lab)
        ax.axhline(1, ls=":", color="k", label="ideal")
        ax.set_ylabel("Eficiència (speedup / fils)")
        ax.set_title("Eficiència")

    for ax in axes:
        ax.set_xlabel("Fils")
        ax.set_xticks(ths)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=200)
    fig.savefig(args.out + ".pdf")

    with open(args.out + "_taula.tex", "w") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
        f.write("\\textbf{Etiqueta} & \\textbf{Fils} & \\textbf{Temps (s)} & "
                "\\textbf{Speedup} & \\textbf{Eficiència} \\\\ \\hline\n")
        if T0:
            f.write(f"--- & seq & {fc(T0, 3)} $\\pm$ {fc(S0, 3)} & 1 & --- \\\\ \\hline\n")
        for r in rows:
            lab = r["label"].replace("_", "\\_")
            sp = f"{fc(r['S'])} $\\pm$ {fc(r['Serr'])}" if r["S"] is not None else "---"
            ef = fc(r["E"]) if r["E"] is not None else "---"
            f.write(f"{lab} & {r['p']} & {fc(r['mean'], 3)} $\\pm$ {fc(r['std'], 3)} & {sp} & {ef} \\\\ \\hline\n")
        f.write("\\end{tabular}\n\\caption{Mitjana $\\pm$ desviació estàndard de les execucions.}\n"
                "\\label{tab:resultats}\n\\end{table}\n")

    print(f"\nGenerats: {args.out}.png, {args.out}.pdf, {args.out}.csv, {args.out}_taula.tex")


if __name__ == "__main__":
    main()
