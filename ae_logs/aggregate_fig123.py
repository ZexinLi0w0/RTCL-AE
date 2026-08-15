import re, os, csv, sys
AE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(AE)
NTRAIN = {"split_cifar10": 50000, "split_cifar100": 5000, "core50_ni": 119894, "core50_nc": 119894, "core50_nic": 119894}

def parse_master(path):
    d = {}
    if not os.path.exists(path):
        print("WARN: missing master log %s" % path, file=sys.stderr)
        return d
    for line in open(path):
        m = re.search(r"\[RUN (\S+)\] END .* rc=(\d+) dur=(\d+)s", line)
        if m and m.group(2) == "0":
            d[m.group(1)] = (int(m.group(2)), int(m.group(3)))  # last successful wins
    return d

M = parse_master(AE + "/ae_master.log")

def sa(path, vanilla):
    if not os.path.exists(path):
        print("WARN: missing log %s" % path, file=sys.stderr)
        return None
    t = open(path, errors="ignore").read()
    pat = r"Top1_Acc_Stream/eval_phase/test_stream/Task[0-9]+ = ([0-9.]+)" if vanilla else r"Overall Streaming Accuracy: ([0-9.]+)"
    v = re.findall(pat, t)
    return float(v[-1]) if v else None

def saf(p, van):
    s = sa(p, van)
    return round(s, 4) if s is not None else ""

def w(path, cols, rows):
    with open(path, "w", newline="") as f:
        c = csv.writer(f); c.writerow(cols); c.writerows(rows)
    print("WROTE %s rows=%d" % (path, len(rows)))

# ---- Fig 1: GPU utilization traces ----
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
trace_rows, summary_rows = [], []
for mode in ["vanilla", "da", "fp", "ours"]:
    tlog = AE + "/fig1/tegra_%s.log" % mode
    if not os.path.exists(tlog):
        print("WARN: missing tegrastats log %s" % tlog, file=sys.stderr)
        continue
    vals = []
    for line in open(tlog, errors="ignore"):
        g = GR3D.findall(line)
        if g:
            vals.append(int(g[0]))
    for i, v in enumerate(vals):
        trace_rows.append([mode, i, v])
    dur = M["fig1/" + mode][1] if "fig1/" + mode in M else ""
    mean = round(sum(vals) / len(vals), 2) if vals else ""
    summary_rows.append([mode, len(vals), mean, dur])
w(ROOT + "/fig1_gpu_util.csv", ["mode", "sample_idx", "gr3d_pct"], trace_rows)
w(ROOT + "/fig1_gpu_util_summary.csv", ["mode", "samples", "mean_gr3d_pct", "run_wallclock_sec"], summary_rows)

# ---- Fig 2a: alternation methods at fixed 0.1s timeslice ----
rows = []
for b in ["split_cifar10", "core50_nc", "core50_nic"]:
    n = NTRAIN[b]
    for mode in ["da", "fp", "ta", "aa"]:
        name = "%s_%s" % (mode, b); k = "fig2a/" + name
        if k not in M:
            print("WARN: no successful run for %s" % k, file=sys.stderr); continue
        rc, d = M[k]; lp = AE + "/fig2a/" + name + ".log"
        rows.append([mode, b, 0.1, n, d, round(n / d, 1) if d else "", saf(lp, False), rc, "ae_logs/fig2a/" + name + ".log"])
w(ROOT + "/fig2a_alternation_methods.csv",
  ["method", "benchmark", "timeslice_s", "train_samples", "wallclock_sec", "qps", "streaming_accuracy", "return_code", "log_file"], rows)

# ---- Fig 2b: alternation-interval sweep on core50_nic ----
rows = []
n = NTRAIN["core50_nic"]
names = [("vanilla", "-", "vanilla_core50_nic", True)]
for mode in ["da", "fp"]:
    for ts in ["0.001", "0.01", "0.1", "1.0"]:
        names.append((mode, ts, "%s_ts%s_core50_nic" % (mode, ts), False))
for mode, ts, name, van in names:
    k = "fig2b/" + name
    if k not in M:
        print("WARN: no successful run for %s" % k, file=sys.stderr); continue
    rc, d = M[k]; lp = AE + "/fig2b/" + name + ".log"
    rows.append([mode, "core50_nic", ts, n, d, round(n / d, 1) if d else "", saf(lp, van), rc, "ae_logs/fig2b/" + name + ".log"])
w(ROOT + "/fig2b_alternation_interval.csv",
  ["method", "benchmark", "timeslice_s", "train_samples", "wallclock_sec", "qps", "streaming_accuracy", "return_code", "log_file"], rows)

# ---- Fig 3: batch-size sweep (2-minute smoke test) ----
def parse_smoke(path):
    d = {}
    if not os.path.exists(path):
        print("WARN: missing master log %s" % path, file=sys.stderr)
        return d
    for line in open(path):
        m = re.search(r"\[SMOKE (\S+)\] END .* rc=(\d+) verdict=(PASS|FAIL)", line)
        if m:
            d[m.group(1)] = (m.group(3), int(m.group(2)))  # last entry wins
    return d

SM = parse_smoke(AE + "/ae_master.log")
NOTE = "2-minute smoke test only; full sweep omitted for time (see README for the full command)"
rows = []
for b in ["split_cifar10", "core50_nc", "core50_nic"]:
    for mode in ["da", "fp"]:
        for bs in [8, 16, 32, 64, 128, 256]:
            name = "%s_bs%d_%s" % (mode, bs, b); k = "fig3/" + name
            if k not in SM:
                print("WARN: no smoke entry for %s" % k, file=sys.stderr); continue
            status, rc = SM[k]
            rows.append([mode, b, bs, status, rc, "ae_logs/fig3/" + name + ".log", NOTE])
w(ROOT + "/fig3_batch_size.csv",
  ["method", "benchmark", "training_bs", "smoke_status", "return_code", "log_file", "note"], rows)
