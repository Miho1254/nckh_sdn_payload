"""
Test NAT LB v3: Thêm kiểm tra OVS flows TRƯỚC VÀ SAU curl.
"""
import time
import subprocess
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.link import TCLink, TCIntf
from mininet.log import setLogLevel, info
from topo_fattree import FatTree

# --- MACRO MONKEY PATCH TẢNG HÌNH CẢNH BÁO CỦA TCLINK ---
original_tc = TCIntf.tc
def patched_tc(self, cmd, *args, **kwargs):
    if isinstance(cmd, str) and not cmd.endswith(' 2>/dev/null'):
        cmd += ' 2>/dev/null'
    return original_tc(self, cmd, *args, **kwargs)
TCIntf.tc = patched_tc
# --------------------------------------------------------

setLogLevel('info')

def test_lb():
    topo = FatTree()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1'),
        switch=OVSSwitch,
        host=Host,
        link=TCLink,
    )

    info('\n*** Starting Ryu Controller...\n')
    ryu = subprocess.Popen(
        ['ryu-manager', 'controller_stats.py', '--verbose'],
        cwd='/work',
        stdout=open('/tmp/ryu_test.log', 'w'),
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    net.start()

    # STP
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=true')
        sw.cmd(f'ovs-vsctl set bridge {sw.name} fail_mode=secure')

    info('*** Waiting 45s for STP convergence...\n')
    for remaining in range(45, 0, -5):
        info(f'  {remaining}s..')
        time.sleep(5)
    info('\n')

    info('*** PingAll...\n')
    net.pingAll()

    from config import BACKENDS, VIP
    backend_names = [b['name'] for b in BACKENDS]
    endpoints = [net.get(n) for n in backend_names]
    h9 = net.get('h9')
    s9 = net.get('s9')

    # HTTP servers
    for host in endpoints:
        host.cmd(f'python3 -m http.server 4000 > /tmp/{host.name}_http.log 2>&1 &')
    time.sleep(1)

    # CHECK: VIP redirect flow on s9 BEFORE curl
    info('\n*** FLOWS on s9 (priority>=50 BEFORE curl)\n')
    flows = s9.cmd(f'ovs-ofctl dump-flows s9 -O OpenFlow13 | grep priority=50')
    info(f'{flows}\n')

    if not flows.strip():
        info('*** WARNING: No VIP redirect flow on s9! Re-installing...\n')
        # Install manually via ovs-ofctl
        s9.cmd(f'ovs-ofctl add-flow s9 "priority=50,ip,nw_dst={VIP},actions=CONTROLLER:65535" -O OpenFlow13')
        s9.cmd(f'ovs-ofctl add-flow s9 "priority=50,arp,arp_tpa={VIP},actions=CONTROLLER:65535" -O OpenFlow13')
        flows2 = s9.cmd('ovs-ofctl dump-flows s9 -O OpenFlow13 | grep priority=50')
        info(f'After re-install: {flows2}\n')

    # tcpdump
    for host in endpoints:
        host.cmd(f'tcpdump -l -i {host.name}-eth0 -n -c 20 > /tmp/{host.name}_pcap.txt 2>&1 &')
    h9.cmd('tcpdump -l -i h9-eth0 -n -c 20 > /tmp/h9_pcap.txt 2>&1 &')
    time.sleep(1)

    # TEST 1: direct curl to first backend
    first_b = BACKENDS[0]
    info(f'\n*** TEST 1: curl direct {first_b["name"]} ({first_b["ip"]}:4000)...\n')
    out1 = h9.cmd(f'curl -s -o /dev/null -w "%{{http_code}}" http://{first_b["ip"]}:4000 --max-time 5')
    info(f'    HTTP Status: {out1}\n')

    # TEST 2: curl VIP
    info(f'\n*** TEST 2: curl VIP ({VIP}:4000)...\n')
    out2 = h9.cmd(f'curl -v http://{VIP}:4000 --max-time 5 2>&1')
    info(f'    Output:\n{out2}\n')

    # CHECK: flows on s9 AFTER curl
    info('\n*** FLOWS on s9 (priority>=50 AFTER curl)\n')
    flows_after = s9.cmd('ovs-ofctl dump-flows s9 -O OpenFlow13 | grep -E "priority=(50|100)"')
    info(f'{flows_after}\n')

    time.sleep(2)
    for host_name in backend_names:
        h = net.get(host_name)
        h.cmd('killall tcpdump 2>/dev/null')
    h9.cmd('killall tcpdump 2>/dev/null')
    time.sleep(1)

    for host_name in [backend_names[0], 'h9']:
        h = net.get(host_name)
        pcap = h.cmd(f'cat /tmp/{host_name}_pcap.txt')
        print(f'\n--- {host_name} PCAP ---')
        print(pcap if pcap.strip() else '(EMPTY)')

    # Ryu log
    print('\n--- RYU LOG (LB related) ---')
    import subprocess as sp
    ryu_log = sp.check_output(['grep', '-E', r'\[LB\]|error|ERROR|Switch s', '/tmp/ryu_test.log']).decode()
    print(ryu_log[:2000])

    net.stop()
    ryu.terminate()

if __name__ == '__main__':
    test_lb()
