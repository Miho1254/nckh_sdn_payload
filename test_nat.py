"""
Test NAT LB v3: Thêm kiểm tra OVS flows TRƯỚC VÀ SAU curl.
"""
import time
import subprocess
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel, info
from topo_fattree import FatTree

setLogLevel('info')

def test_lb():
    topo = FatTree()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1'),
        switch=OVSSwitch,
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

    h5, h7, h8, h9 = [net.get(n) for n in ('h5','h7','h8','h9')]
    s9 = net.get('s9')

    # HTTP servers
    h5.cmd('python3 -m http.server 4000 > /tmp/h5_http.log 2>&1 &')
    h7.cmd('python3 -m http.server 4000 > /tmp/h7_http.log 2>&1 &')
    h8.cmd('python3 -m http.server 4000 > /tmp/h8_http.log 2>&1 &')
    time.sleep(1)

    # CHECK: VIP redirect flow on s9 BEFORE curl
    info('\n*** FLOWS on s9 (priority>=50 BEFORE curl)\n')
    flows = s9.cmd('ovs-ofctl dump-flows s9 -O OpenFlow13 | grep priority=50')
    info(f'{flows}\n')

    if not flows.strip():
        info('*** WARNING: No VIP redirect flow on s9! Re-installing...\n')
        # Install manually via ovs-ofctl
        s9.cmd('ovs-ofctl add-flow s9 "priority=50,ip,nw_dst=10.0.0.100,actions=CONTROLLER:65535" -O OpenFlow13')
        s9.cmd('ovs-ofctl add-flow s9 "priority=50,arp,arp_tpa=10.0.0.100,actions=CONTROLLER:65535" -O OpenFlow13')
        flows2 = s9.cmd('ovs-ofctl dump-flows s9 -O OpenFlow13 | grep priority=50')
        info(f'After re-install: {flows2}\n')

    # tcpdump
    h5.cmd('tcpdump -l -i h5-eth0 -n -c 20 > /tmp/h5_pcap.txt 2>&1 &')
    h9.cmd('tcpdump -l -i h9-eth0 -n -c 20 > /tmp/h9_pcap.txt 2>&1 &')
    time.sleep(1)

    # TEST 1: direct curl
    info('\n*** TEST 1: curl direct h5 (10.0.0.5:4000)...\n')
    out1 = h9.cmd('curl -s -o /dev/null -w "%{http_code}" http://10.0.0.5:4000 --max-time 5')
    info(f'    HTTP Status: {out1}\n')

    # TEST 2: curl VIP
    info('\n*** TEST 2: curl VIP (10.0.0.100:4000)...\n')
    out2 = h9.cmd('curl -v http://10.0.0.100:4000 --max-time 5 2>&1')
    info(f'    Output:\n{out2}\n')

    # CHECK: flows on s9 AFTER curl
    info('\n*** FLOWS on s9 (priority>=50 AFTER curl)\n')
    flows_after = s9.cmd('ovs-ofctl dump-flows s9 -O OpenFlow13 | grep -E "priority=(50|100)"')
    info(f'{flows_after}\n')

    time.sleep(2)
    for host_name in ['h5', 'h7', 'h8']:
        h = net.get(host_name)
        h.cmd('killall tcpdump 2>/dev/null')
    h9.cmd('killall tcpdump 2>/dev/null')
    time.sleep(1)

    for host_name in ['h5', 'h9']:
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
