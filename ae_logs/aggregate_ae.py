import re, os, csv
ROOT="/experiment/zexin/RTCL-AE"; AE=ROOT+"/ae_logs"; F6=ROOT+"/fig6_logs"
NTRAIN={"split_cifar10":50000,"split_cifar100":5000,"core50_ni":119894,"core50_nc":119894,"core50_nic":119894}
def parse_master(path):
    d={}
    if os.path.exists(path):
        for line in open(path):
            m=re.search(r"\[RUN (\S+)\] END .* rc=(\d+) dur=(\d+)s", line)
            if m: d[m.group(1)]=(int(m.group(2)),int(m.group(3)))
    return d
AEM=parse_master(AE+"/ae_master.log"); F6M=parse_master(F6+"/fig6_master.log")
def sa(path,vanilla):
    if not os.path.exists(path): return None
    t=open(path,errors="ignore").read()
    pat=r"Top1_Acc_Stream/eval_phase/test_stream/Task[0-9]+ = ([0-9.]+)" if vanilla else r"Overall Streaming Accuracy: ([0-9.]+)"
    v=re.findall(pat,t); return float(v[-1]) if v else None
def ss_fallback(path):
    # For semseg logs predating the eval_worker fix: mean per-experience pixel
    # accuracy from the LAST "Experience Accuracies for this cycle:" block.
    if not os.path.exists(path): return None
    t=open(path,errors="ignore").read()
    blocks=re.findall(r"Experience Accuracies for this cycle:\n((?:.*- Experience \d+: [0-9.]+\n)+)",t)
    if not blocks: return None
    vals=[float(x) for x in re.findall(r"- Experience \d+: ([0-9.]+)",blocks[-1])]
    return round(sum(vals)/len(vals),4) if vals else None
def zero_zero(path):
    if not os.path.exists(path): return False
    t=open(path,errors="ignore").read()
    return bool(re.search(r"Overall Streaming Accuracy: 0\.0+ \(0/0\)",t))
def saf_note(p,van,ep,bench):
    s=sa(p,van); note=""
    if ep=="main.py" and (bench=="endless_ss" or (s==0.0 and zero_zero(p))):
        fb=ss_fallback(p)
        if fb is not None:
            return fb,"pixel accuracy: mean over experiences of final eval cycle"
    return (round(s,4) if s is not None else ""),note
def w(path,cols,rows):
    with open(path,"w",newline="") as f:
        c=csv.writer(f); c.writerow(cols); c.writerows(rows)
    print("WROTE",path,"rows=",len(rows))

# ---- Tab 4 ablation ----
benches=list(NTRAIN)
rows=[]
def saf(p,van): 
    s=sa(p,van); return round(s,4) if s is not None else ""
for b in benches:
    n=NTRAIN[b]
    if "fp_"+b in F6M:
        rc,d=F6M["fp_"+b]; rows.append(["AOCL_basic",b,"fully_parallel",n,d,round(n/d,1),saf(F6+"/fp_"+b+".log",False),rc,"fig6_logs/fp_"+b+".log",""])
    k="tab4/adaptive_time_"+b
    if k in AEM:
        rc,d=AEM[k]; rows.append(["TA",b,"adaptive_time",n,d,round(n/d,1) if d else "",saf(AE+"/tab4/adaptive_time_"+b+".log",False),rc,"ae_logs/tab4/adaptive_time_"+b+".log",""])
    k="tab4/adaptive_accuracy_"+b
    if k in AEM:
        rc,d=AEM[k]; rows.append(["AA",b,"adaptive_accuracy",n,d,round(n/d,1) if d else "",saf(AE+"/tab4/adaptive_accuracy_"+b+".log",False),rc,"ae_logs/tab4/adaptive_accuracy_"+b+".log",""])
    if "ours_"+b in F6M:
        rc,d=F6M["ours_"+b]; rows.append(["AOCL",b,"adaptocl",n,d,round(n/d,1),saf(F6+"/ours_"+b+".log",False),rc,"fig6_logs/ours_"+b+".log",""])
w(ROOT+"/tab4_ablation.csv",["config","benchmark","scheduler_mode","train_samples","wallclock_sec","qps","streaming_accuracy","return_code","log_file","note"],rows)

# ---- Fig 7 ----
M=[("vanilla","Avalanche (sequential)","test_avalanche_lib.py","-",True),
   ("da","Default Alternation (DA)","main.py","default",False),
   ("ekya","Ekya","main.py","ekya",False),
   ("recl","RECL","main.py","recl_sched",False),
   ("fp","AOCL_basic (fully parallel)","main.py","fully_parallel",False),
   ("ours","AOCL (ours)","main.py","adaptocl",False)]
algos=["replay","gss_greedy","gem","agem"]; N=119894
base={a:AEM["fig7/vanilla_"+a][1] for a in algos if "fig7/vanilla_"+a in AEM}
frows=[]
for a in algos:
    for mk,mn,ep,mode,van in M:
        k="fig7/%s_%s"%(mk,a)
        if k not in AEM: continue
        rc,d=AEM[k]; lp=AE+"/fig7/%s_%s.log"%(mk,a)
        sp=round(base[a]/d,2) if a in base and d else ""
        frows.append([mn,"core50_nc",ep,mode,a,N,d,round(N/d,1) if d else "",sp,saf(lp,van),rc,"ae_logs/fig7/%s_%s.log"%(mk,a),""])
w(ROOT+"/fig7_results.csv",["method","benchmark","entry_point","scheduler_mode","algorithm","train_samples","wallclock_sec","qps","speedup_vs_avalanche","streaming_accuracy","return_code","log_file","note"],frows)

# ---- Tab 3 ----
trows=[]
for task in ["ilc","ss"]:
    for scen in ["Classes","Illumination","Weather"]:
        for mk,mn,ep,mode,van in M:
            name="%s_%s_%s"%(mk,task,scen); k="tab3/"+name
            if k not in AEM: continue
            rc,d=AEM[k]; acc,note=saf_note(AE+"/tab3/"+name+".log",van,ep,"endless_"+task)
            trows.append([mn,"endless_"+task,scen,ep,mode,d,acc,rc,"ae_logs/tab3/"+name+".log",note])
for scen in ["ic","il"]:
    for mk,mn,ep,mode,van in M:
        name="softrobot_%s_%s"%(mk,scen); k="tab3/"+name
        if k not in AEM: continue
        rc,d=AEM[k]; trows.append([mn,"soft_robot",scen,ep,mode,d,saf(AE+"/tab3/"+name+".log",van),rc,"ae_logs/tab3/"+name+".log",""])
w(ROOT+"/tab3_results.csv",["method","task","scenario","entry_point","scheduler_mode","wallclock_sec","streaming_accuracy","return_code","log_file","note"],trows)
