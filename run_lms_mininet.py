import time
import os
import sys

# Đọc kịch bản mục tiêu (Mặc định nếu quên truyền là flash_crowd.yml)
SCENARIO_FILE = os.environ.get('SCENARIO', 'flash_crowd.yml')

from topo_fattree import FatTree, configure_queues
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

def deploy_lms(net):
    info('\n*** [1/5] Khoi dong PostgreSQL Database tren h6...\n')
    h6 = net.get('h6')
    h6.cmd('pg_dropcluster 14 main --stop || true; rm -rf /var/run/postgresql/* /var/lib/postgresql/14/main /etc/postgresql/14/main; pg_createcluster 14 main; sed -i "s/#listen_addresses = \'localhost\'/listen_addresses = \'*\'/" /etc/postgresql/14/main/postgresql.conf; echo "host all all 0.0.0.0/0 trust" >> /etc/postgresql/14/main/pg_hba.conf; /etc/init.d/postgresql start > /tmp/h6_pg.log 2>&1 && sleep 1 && su - postgres -c "psql -c \\"CREATE USER lms WITH PASSWORD \'lms123\';\\" && psql -c \\"CREATE DATABASE lms OWNER lms;\\""')
    time.sleep(1)

    info('*** [2/5] Seed Du Lieu (Tu h5)...\n')
    h5 = net.get('h5')
    h5.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 npm run seed:massive > /tmp/h5_seed.log 2>&1')
    
    info('*** [3/5] Khoi dong 3 Instances Backend Node.js...\n')
    h5.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 PORT=4000 node server.js > /tmp/h5_api.log 2>&1 &')
    info('    h5 dang chay Backend o port 4000 (Tro ve DB h6)\n')
    
    h7 = net.get('h7')
    h7.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 PORT=4000 node server.js > /tmp/h7_api.log 2>&1 &')
    info('    h7 dang chay Backend o port 4000 (Tro ve DB h6)\n')
    
    h8 = net.get('h8')
    h8.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 PORT=4000 node server.js > /tmp/h8_api.log 2>&1 &')
    info('    h8 dang chay Backend o port 4000 (Tro ve DB h6)\n')
    time.sleep(2)

    info('*** [4/5] Kich hoat Artillery Stress Test toi AI Load Balancer (10.0.0.100)...\n')
    target_apis = ['10.0.0.100:4000']
    
    for i in range(9, 17):
        host = net.get(f'h{i}')
        target = target_apis[i % len(target_apis)]
        info(f'    Khoi chay Labeled Stress Test ({SCENARIO_FILE}) tren h{i} (Muc tieu: {target})...\n')
        host.cmd(f'cd /work/lms/stress-test && SCENARIO="{SCENARIO_FILE}" TARGET="http://{target}" python3 -u run_labeled_test.py > /tmp/h{i}_stress.log 2>&1 &')
        
    info(f'\n*** [5/5] Qua trinh ep traffic theo kich ban {SCENARIO_FILE} da bat dau!\n')
    info('    Script se TU DONG dung khi Artillery hoan tat. Bro chi can ngoi cho.\n')
    info('    (Nhan Ctrl+C bat ky luc nao de vao Mininet CLI thu cong)\n\n')

def wait_for_artillery(net):
    """Doi tat ca tien trinh Artillery tren h9-h16 ket thuc."""
    check_hosts = [net.get(f'h{i}') for i in range(9, 17)]
    start_time = time.time()
    
    info('*** [AUTO-WAIT] Dang giam sat tien trinh Artillery...\n\n')
    
    while True:
        alive_count = 0
        for h in check_hosts:
            result = h.cmd('pgrep -f "artillery run" | wc -l').strip()
            try:
                if int(result) > 0:
                    alive_count += 1
            except ValueError:
                pass
        
        elapsed = int(time.time() - start_time)
        mins = elapsed // 60
        secs = elapsed % 60
        
        if alive_count == 0:
            info(f'\n============================================================\n')
            info(f'  STRESS TEST HOAN TAT! Tong thoi gian: {mins}m {secs}s\n')
            info(f'  Dang tu dong dung Mininet va sao luu ket qua...\n')
            info(f'============================================================\n')
            break
        
        info(f'  [{mins:02d}:{secs:02d}] Artillery dang chay tren {alive_count}/8 hosts...\r')
        sys.stdout.flush()
        time.sleep(15)

if __name__ == '__main__':
    setLogLevel('info')

    topo = FatTree()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
    )

    net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633,
    )

    net.start()
    info('*** Doi cac Switches ket noi Controller (timeout 10s)...\n')
    net.waitConnected(timeout=10)

    info('*** Cau hinh STP va Secure Mode tren switches de chan Loop...\n')
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=true')
        sw.cmd(f'ovs-vsctl set bridge {sw.name} fail_mode=secure')
    
    info('*** Cho 45 giay de STP hoi tu (Fat-Tree can thoi gian dai)...\n')
    for i in range(45):
        if i % 5 == 0: info(f'{45-i}s.. ')
        time.sleep(1)
    info('\n')

    info('*** Vo hieu hoa IPv6 tren toan bo hosts...\n')
    for h in net.hosts:
        h.cmd('sysctl -w net.ipv6.conf.all.disable_ipv6=1')
        h.cmd('sysctl -w net.ipv6.conf.default.disable_ipv6=1')

    info('*** [WARM-UP] "Lam nong" bang MAC toan mang (pingAll)...\n')
    net.pingAll(timeout='0.5')

    # === CÀI VIP REDIRECT FLOW SAU STP CONVERGENCE ===
    # STP có thể xóa flow table khi convergence, nên cài lại ở đây
    VIP = '10.0.0.100'
    info('*** Cai VIP redirect flow (priority=50) tren tat ca switches...\n')
    for sw in net.switches:
        sw.cmd(f'ovs-ofctl add-flow {sw.name} "priority=50,ip,nw_dst={VIP},actions=CONTROLLER:65535" -O OpenFlow13')
        sw.cmd(f'ovs-ofctl add-flow {sw.name} "priority=50,arp,arp_tpa={VIP},actions=CONTROLLER:65535" -O OpenFlow13')
    info('    Done! VIP redirect flows installed on all %d switches.\n' % len(net.switches))

    info('*** Ping kiem tra ket noi toi DB Server (h6) cuoi cung...\n')
    h1 = net.get('h1')
    h6 = net.get('h6')
    h1.cmd(f'arp -s 10.0.0.6 {h6.MAC()}')
    
    connected = False
    for i in range(10):
        res = h1.cmd('ping -c 1 10.0.0.6')
        if '1 received' in res:
            info(f'Mang DA THONG SUOT sau {i} giay!\n')
            connected = True
            break
        info('x')
        time.sleep(1)
    
    if not connected:
        info('Canh bao: Mang van chua on dinh hoan toan, nhung van thu tiep tuc.\n')
    
    deploy_lms(net)
    
    try:
        wait_for_artillery(net)
    except KeyboardInterrupt:
        info('\n*** Ctrl+C! Chuyen sang Mininet CLI thu cong...\n')
        CLI(net)
    
    net.stop()
