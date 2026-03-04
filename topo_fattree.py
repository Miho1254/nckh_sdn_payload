"""
Fat-Tree Topology (k=4) cho nghiên cứu cân bằng tải SDN.

Kiến trúc 3 tầng:
  - Core layer:        2 switches  (s1, s2)
  - Aggregation layer: 4 switches  (s3, s4, s5, s6)
  - Edge layer:        4 switches  (s7, s8, s9, s10)
  - Hosts:             16 hosts    (h1 - h16), 4 hosts/edge switch

Ref: "A Transformer-Based Deep Q-Learning" paper — Section V.A & V.E
"""

import subprocess
from functools import partial
from time import sleep

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

# ── Topology ────────────────────────────────────────────────

class FatTree(Topo):
    """Fat-Tree k=4 (gọn nhẹ: 2 core switches)."""

    def build(self):
        # ── Core switches ───────────────────────────────────
        core_switches = []
        for i in range(1, 3):                       # s1, s2
            sw = self.addSwitch(f's{i}', protocols='OpenFlow13', failMode='secure')
            core_switches.append(sw)

        # ── Aggregation switches ────────────────────────────
        agg_switches = []
        for i in range(3, 7):                       # s3, s4, s5, s6
            sw = self.addSwitch(f's{i}', protocols='OpenFlow13', failMode='secure')
            agg_switches.append(sw)

        # ── Edge switches ───────────────────────────────────
        edge_switches = []
        for i in range(7, 11):                      # s7, s8, s9, s10
            sw = self.addSwitch(f's{i}', protocols='OpenFlow13', failMode='secure')
            edge_switches.append(sw)

        # ── Hosts ───────────────────────────────────────────
        # MAC cố định cho mỗi host: 00:00:00:00:00:XX (XX = host number)
        # QUAN TRỌNG: h5, h7, h8 phải khớp với BACKENDS trong controller_stats.py
        
        # CẤU HÌNH TÀI NGUYÊN KHÁC NHAU (Heterogeneous)
        # Vì lỗi Docker Cgroups, ta loại bỏ CPU quota và chỉ dùng Bandwidth (dưới đây) làm nút thắt cổ chai.
        
        hosts = []
        for i in range(1, 17):                      # h1 - h16
            h_name = f'h{i}'
            mac = '00:00:00:00:00:%02x' % i
            h = self.addHost(h_name, mac=mac)
            hosts.append(h)

        # ── Links: Core ↔ Aggregation ───────────────────────
        # Mỗi core switch kết nối tới TẤT CẢ aggregation switches
        for core in core_switches:
            for agg in agg_switches:
                self.addLink(core, agg)

        # ── Links: Aggregation ↔ Edge (theo pod) ────────────
        # Pod 0: s3 ↔ s7    Pod 1: s4 ↔ s8
        # Pod 2: s5 ↔ s9    Pod 3: s6 ↔ s10
        for idx, agg in enumerate(agg_switches):
            self.addLink(agg, edge_switches[idx])

        # ── Links: Edge ↔ Hosts (4 hosts mỗi edge switch) ──
        # Băng thông mô phỏng thực tế: h5 bị nghẽn (10M), h7 bình thường (50M), h8 máy chủ xịn (100M)
        bw_configs = {
            'h5': 10,   # 10 Mbps (mạng yếu)
            'h7': 50,   # 50 Mbps 
            'h8': 100,  # 100 Mbps (siêu mạng)
        }
        
        for idx, edge in enumerate(edge_switches):
            for j in range(4):
                host_index = idx * 4 + j             # 0-based
                h_name = f'h{host_index + 1}'
                
                # Áp dụng giới hạn bandwidth nếu host là server
                if h_name in bw_configs:
                    self.addLink(edge, hosts[host_index], bw=bw_configs[h_name])
                else:
                    self.addLink(edge, hosts[host_index])

# ── Queue Configuration ─────────────────────────────────────

def configure_queues(net):
    """
    Cấu hình QoS queues trên các switches (sử dụng ovs-vsctl).

    Theo paper:
      - Edge switches (s7-s10): min_rate=5Mbps, max_rate=10Mbps
      - Core/Agg switches:      min_rate=1Mbps, max_rate=5Mbps
    """
    edge_switch_names = {f's{i}' for i in range(7, 11)}

    for sw in net.switches:
        name = sw.name
        is_edge = name in edge_switch_names

        min_rate = 5_000_000 if is_edge else 1_000_000
        max_rate = 10_000_000 if is_edge else 5_000_000

        for intf in sw.intfList():
            if intf.name == 'lo':
                continue
            # Queue 0 (default)
            sw.cmd(
                f'ovs-vsctl -- set port {intf.name} qos=@newqos '
                f'-- --id=@newqos create qos type=linux-htb '
                f'other-config:max-rate={max_rate} queues=0=@q0 '
                f'-- --id=@q0 create queue '
                f'other-config:min-rate={min_rate} '
                f'other-config:max-rate={max_rate}'
            )

    info('*** Queue configuration applied\n')


# ── Traffic Generation ───────────────────────────────────────

def generate_traffic(net):
    """
    Tạo lưu lượng hỗn hợp (TCP, UDP, HTTP, DNS, ICMP) giữa các hosts.
    Mỗi host khởi động HTTP server, sau đó traffic được phát sinh chéo.
    """
    hosts = net.hosts

    # Khởi động HTTP server trên mỗi host
    for h in hosts:
        h.cmd('mkdir -p /tmp/www')
        h.cmd('echo "<html><body>Host {}</body></html>" > /tmp/www/index.html'.format(h.name))
        h.cmd('cd /tmp/www && python3 -m http.server 80 &')

    sleep(2)

    info('*** Starting traffic generation\n')

    half = len(hosts) // 2
    for i in range(half):
        src = hosts[i]
        dst = hosts[half + i]
        dst_ip = dst.IP()

        # TCP  (iperf, port 5001, 60s, 500Mbps)
        dst.cmd('iperf -s -p 5001 &')
        src.cmd(f'iperf -c {dst_ip} -p 5001 -t 60 -b 500M &')

        # UDP  (iperf, port 5002, 60s, 500Mbps)
        dst.cmd('iperf -s -u -p 5002 &')
        src.cmd(f'iperf -c {dst_ip} -u -p 5002 -t 60 -b 500M &')

        # HTTP
        src.cmd(f'wget -q -O /dev/null http://{dst_ip}/index.html &')

        # DNS
        src.cmd(f'dig @{dst_ip} www.example.com &')

        # ICMP
        src.cmd(f'ping -c 100 {dst_ip} &')

    info('*** Traffic generation started\n')


# ── Main ─────────────────────────────────────────────────────

if __name__ == '__main__':
    setLogLevel('info')

    # ⚠️  THỨ TỰ CHẠY:
    #   Terminal 1:  ryu-manager controller_stats.py --ofp-tcp-listen-port 6633
    #   Terminal 2:  python3 topo_fattree.py
    #   (Ryu phải chạy TRƯỚC, Mininet kết nối TỚI Ryu)

    topo = FatTree()
    net = Mininet(
        topo=topo,
        controller=None,       # Không tự tạo controller
        switch=OVSSwitch,
        host=Host,
        link=TCLink,
    )

    # Ryu controller tại localhost:6633
    net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633,
    )

    net.start()

    # Đợi switches kết nối controller (timeout 10s thay vì vô hạn)
    info('*** Waiting for switches to connect...\n')
    net.waitConnected(timeout=10)

    # Bật STP để chặn broadcast loop trong Fat-Tree
    info('*** Enabling STP on all switches...\n')
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=true')
    info('*** Waiting 30s for STP convergence...\n')
    sleep(30)

    # Cấu hình QoS queues
    configure_queues(net)

    # Test kết nối cơ bản
    info('*** Running pingAll\n')
    net.pingAll()

    # (Tùy chọn) Phát sinh traffic
    # generate_traffic(net)

    # Mở CLI để tương tác
    CLI(net)

    net.stop()