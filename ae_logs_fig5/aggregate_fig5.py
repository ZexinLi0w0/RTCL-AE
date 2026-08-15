#!/usr/bin/env python3
"""Aggregate Fig 5 (large-model) sweep on SplitCIFAR10 into fig5_results.csv."""
import re, csv, os
BR="/experiment/zexin/RTCL-AE-fig5"
LD=f"{BR}/ae_logs_fig5"
MASTER=f"{LD}/fig5_master.log"
NTRAIN=50000  # split_cifar10
METHOD={"vanilla":("Avalanche (sequential)","test_avalanche_lib.py","-"),
        "da":("Default Alternation (DA)","main.py","default"),
        "ekya":("Ekya","main.py","ekya"),
        "recl":("RECL","main.py","recl_sched"),
        "fp":("AOCL_basic (fully parallel)","main.py","fully_parallel"),
        "ours":("AOCL (ours)","main.py","adaptocl")}
def last_end(name):
    rc=dur=None
    pat=re.compile(rf"\[RUN {re.escape(name)}\] END .* rc=(\d+) dur=(\d+)s")
    for line in open(MASTER):
        m=pat.search(line)
        if m: rc=int(m.group(1)); dur=int(m.group(2))
    return rc,dur
def sa(log, vanilla):
    if not os.path.exists(log): return ""
    txt=open(log,errors="ignore").read()
    if vanilla:
        ms=re.findall(r"Top1_Acc_Stream/eval_phase/test_stream/Task[0-9]+ = ([0-9.]+)", txt)
    else:
        ms=re.findall(r"Overall Streaming Accuracy: ([0-9.]+)", txt)
    return ms[-1] if ms else ""
names=sorted(set(re.findall(r"\[RUN (\S+)\] START", open(MASTER).read())))
rows=[]
for name in names:
    mkey, model = name.split("_",1)
    disp,entry,mode=METHOD[mkey]
    rc,dur=last_end(name)
    logrel=f"ae_logs_fig5/{name}.log"
    vanilla=(mkey=="vanilla")
    streaming=sa(f"{LD}/{name}.log",vanilla) if rc==0 else ""
    qps=round(NTRAIN/dur,1) if (rc==0 and dur) else ""
    note=""
    if vanilla and rc not in (0,None):
        note="Avalanche baseline unsupported on this branch (test_avalanche_lib.py builds CIFAR ResNets by depth)"
    elif rc not in (0,None):
        note="run failed (see log)"
    rows.append([disp,"split_cifar10",model,entry,mode,"replay",NTRAIN,dur or "",qps,streaming,rc if rc is not None else "",logrel,note])
morder={m:i for i,m in enumerate(["resnet50","resnet101","resnet152","resnet200","mobilenetv1","vit_tiny"])}
korder={k:i for i,k in enumerate(["vanilla","da","ekya","recl","fp","ours"])}
d2k={v[0]:k for k,v in METHOD.items()}
rows.sort(key=lambda r:(morder.get(r[2],99), korder.get(d2k[r[0]],99)))
hdr=["method","benchmark","model","entry_point","scheduler_mode","algorithm","train_samples","wallclock_sec","qps","streaming_accuracy","return_code","log_file","note"]
with open(f"{BR}/fig5_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("WROTE fig5_results.csv rows=",len(rows))
