import re, os, csv, math
LD="/experiment/zexin/RTCL-AE/fig6_logs"
methods=[("da","Default Alternation (DA)"),("ekya","Ekya"),("recl","RECL"),
         ("fp","AOCL_basic (fully parallel)"),("ours","AOCL (ours)")]
benches=["split_cifar10","split_cifar100","core50_ni","core50_nc","core50_nic"]
pat=re.compile(r"batch_lat=([0-9.]+)s")
def pct(v,p):
    v=sorted(v); k=(len(v)-1)*p/100.0; f=math.floor(k); c=math.ceil(k)
    return v[int(k)] if f==c else v[f]*(c-k)+v[c]*(k-f)
rows8=[]; rows2=[]
for mk,mn in methods:
    for b in benches:
        p=os.path.join(LD,"%s_%s.log"%(mk,b)); vals=[]
        if os.path.exists(p):
            for line in open(p,errors="ignore"):
                m=pat.search(line)
                if m: vals.append(float(m.group(1))*1000.0)
        n=len(vals)
        if n==0:
            rows8.append([mn,b,0,"","","","","",""]); rows2.append([mn,b,0,"","",""]); continue
        rows8.append([mn,b,n,round(pct(vals,50),3),round(pct(vals,90),3),round(pct(vals,99),3),
                      round(pct(vals,99.9),3),round(sum(vals)/n,3),round(max(vals),3)])
        f16=sum(1 for x in vals if x>16)/n*100; f33=sum(1 for x in vals if x>33)/n*100; f100=sum(1 for x in vals if x>100)/n*100
        rows2.append([mn,b,n,round(f16,3),round(f33,3),round(f100,3)])
with open(os.path.join(LD,"fig8_latency_cdf.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["method","benchmark","samples","p50_ms","p90_ms","p99_ms","p99_9_ms","mean_ms","max_ms"]); w.writerows(rows8)
with open(os.path.join(LD,"tab2_dmr.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["method","benchmark","samples","dmr_16ms_pct","dmr_33ms_pct","dmr_100ms_pct"]); w.writerows(rows2)
print("done")
