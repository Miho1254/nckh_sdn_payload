cores = [200, 333, 466, 600]
aggs = [[70, 130], [270, 330], [470, 530], [670, 730]]
edges = [[70, 130], [270, 330], [470, 530], [670, 730]]
hosts = [
    [40, 80, 120, 160],
    [240, 280, 320, 360],
    [440, 480, 520, 560],
    [640, 680, 720, 760]
]

svg = []
svg.append('<svg viewBox="0 0 800 420" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%;">')
svg.append('<!-- DEFINITIONS -->')
svg.append('<defs>')
svg.append('  <style>')
svg.append('    .link { stroke: #cbd5e1; stroke-width: 2; }')
svg.append('    .link-active { stroke: #3b82f6; stroke-width: 3; stroke-dasharray: 4; animation: dash 1s linear infinite; }')
svg.append('    .node-core { fill: #f8fafc; stroke: #0f172a; stroke-width: 3; }')
svg.append('    .node-agg { fill: #e2e8f0; stroke: #0f172a; stroke-width: 3; }')
svg.append('    .node-edge { fill: #cbd5e1; stroke: #0f172a; stroke-width: 3; }')
svg.append('    .node-host { fill: #ffffff; stroke: #0f172a; stroke-width: 3; }')
svg.append('    .host-text { font-family: Inter, sans-serif; font-size: 10px; font-weight: bold; fill: #0f172a; text-anchor: middle; alignment-baseline: middle; }')
svg.append('    .node-shadow { filter: drop-shadow(3px 3px 0px #0f172a); }')
svg.append('    @keyframes dash { to { stroke-dashoffset: -8; } }')
svg.append('  </style>')
svg.append('</defs>')

# Lines Core -> Agg
svg.append('<!-- Core to Agg -->')
for c_idx, cx in enumerate(cores):
    for p_idx in range(4):
        a_idx = 0 if c_idx < 2 else 1
        ax = aggs[p_idx][a_idx]
        svg.append(f'  <line x1="{cx}" y1="70" x2="{ax}" y2="140" class="link" />')

# Lines Agg -> Edge
svg.append('<!-- Agg to Edge -->')
for p_idx in range(4):
    for a_idx, ax in enumerate(aggs[p_idx]):
        for e_idx, ex in enumerate(edges[p_idx]):
            svg.append(f'  <line x1="{ax}" y1="170" x2="{ex}" y2="240" class="link" />')

# Lines Edge -> Host
svg.append('<!-- Edge to Host -->')
for p_idx in range(4):
    for e_idx, ex in enumerate(edges[p_idx]):
        h_start = 0 if e_idx == 0 else 2
        for h in range(h_start, h_start+2):
            hx = hosts[p_idx][h]
            link_cls = "link"
            if p_idx == 1 and h in [0, 2, 3]:
                link_cls = "link link-active"
            svg.append(f'  <line x1="{ex}" y1="270" x2="{hx}" y2="340" class="{link_cls}" />')

# Nodes Cores
svg.append('<!-- Cores -->')
for cx in cores:
    svg.append(f'  <rect x="{cx-25}" y="40" width="50" height="30" class="node-core node-shadow" rx="4"/>')
    svg.append(f'  <text x="{cx}" y="55" class="host-text">CORE</text>')

# Nodes Agg
svg.append('<!-- Aggregations -->')
for p_idx in range(4):
    for ax in aggs[p_idx]:
        svg.append(f'  <rect x="{ax-20}" y="140" width="40" height="30" class="node-agg node-shadow" rx="4"/>')
        svg.append(f'  <text x="{ax}" y="155" class="host-text">AGG</text>')

# Nodes Edge
svg.append('<!-- Edges -->')
for p_idx in range(4):
    for ex in edges[p_idx]:
        svg.append(f'  <rect x="{ex-20}" y="240" width="40" height="30" class="node-edge node-shadow" rx="4"/>')
        svg.append(f'  <text x="{ex}" y="255" class="host-text">EDG</text>')

# Nodes Host
svg.append('<!-- Hosts -->')
host_count = 1
for p_idx in range(4):
    for idx, hx in enumerate(hosts[p_idx]):
        fill_color = "#ffffff"
        text_color = "#0f172a"
        if host_count == 5:
            fill_color = "#ef4444"
            text_color = "#ffffff"
        elif host_count == 7:
            fill_color = "#eab308"
            text_color = "#000000"
        elif host_count == 8:
            fill_color = "#22c55e"
            text_color = "#ffffff"
        
        svg.append(f'  <rect x="{hx-14}" y="340" width="28" height="28" class="node-host node-shadow" rx="6" style="fill: {fill_color}"/>')
        svg.append(f'  <text x="{hx}" y="354" class="host-text" style="fill: {text_color}">h{host_count}</text>')
        host_count += 1

svg.append('</svg>')

with open("fat_tree.svg", "w") as f:
    f.write("\n".join(svg))
