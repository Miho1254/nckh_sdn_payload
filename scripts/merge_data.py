import sys, os

scenarios = ["golden_hour", "video_conference", "hardware_degradation", "low_rate_dos"]
first = True

flow_out = 'stats/flow_stats.csv'
port_out = 'stats/port_stats.csv'

if os.path.exists(flow_out): os.remove(flow_out)
if os.path.exists(port_out): os.remove(port_out)

for s in scenarios:
    f = f"stats/results/COLLECT_{s}/flow_stats.csv"
    p = f"stats/results/COLLECT_{s}/port_stats.csv"
    
    if os.path.exists(f):
        if first:
            with open(flow_out, 'w') as out:
                head = open(f).readline().strip()
                out.write(head + ',scenario\n')
            if os.path.exists(p):
                with open(port_out, 'w') as out:
                    head = open(p).readline().strip()
                    out.write(head + ',scenario\n')
            first = False
            
        with open(flow_out, 'a') as out, open(f) as fin:
            lines = fin.readlines()[1:]
            for l in lines:
                out.write(l.strip() + ',' + s + '\n')
                
        if os.path.exists(p):
            with open(port_out, 'a') as out, open(p) as fin:
                lines = fin.readlines()[1:]
                for l in lines:
                    out.write(l.strip() + ',' + s + '\n')
        print(f"  + Merged: {s}")

flow_lines = sum(1 for _ in open(flow_out)) - 1 if os.path.exists(flow_out) else 0
port_lines = sum(1 for _ in open(port_out)) - 1 if os.path.exists(port_out) else 0
print(f"  Flow stats: {flow_lines} rows")
if os.path.exists(port_out):
    print(f"  Port stats: {port_lines} rows")
