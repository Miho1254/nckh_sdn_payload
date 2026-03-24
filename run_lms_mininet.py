import time
import os
import sys

# Đọc kịch bản mục tiêu (Mặc định nếu quên truyền là golden_hour.yml)
SCENARIO_FILE = os.environ.get('SCENARIO', 'golden_hour.yml')

from topo_fattree import FatTree, configure_queues
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.link import TCLink, TCIntf
from mininet.log import setLogLevel, info
from mininet.cli import CLI

# --- MACRO MONKEY PATCH TẢNG HÌNH CẢNH BÁO CỦA TCLINK ---
# Lệnh `tc` của Linux luôn in cảnh báo ra stderr khiến Mininet tự động phát hiện và in "*** Error:"
# Điều này làm console bị bẩn và gây hoang mang. Patch này thêm 2>/dev/null vào mọi lệnh tc.
original_tc = TCIntf.tc
def patched_tc(self, cmd, *args, **kwargs):
    # Nhét cờ chặn stderr trực tiếp vào câu lệnh bash
    if isinstance(cmd, str) and not cmd.endswith(' 2>/dev/null'):
        cmd += ' 2>/dev/null'
    return original_tc(self, cmd, *args, **kwargs)
TCIntf.tc = patched_tc
# --------------------------------------------------------

def deploy_lms(net):
    from config import BACKENDS, DB_NODE, DB_IP
    
    info(f'\n*** [1/5] Khoi dong PostgreSQL Database tren {DB_NODE}...\n')
    db_host = net.get(DB_NODE)
    db_host.cmd('pg_dropcluster 14 main --stop || true; rm -rf /var/run/postgresql/* /var/lib/postgresql/14/main /etc/postgresql/14/main; pg_createcluster 14 main; sed -i "s/#listen_addresses = \'localhost\'/listen_addresses = \'*\'/" /etc/postgresql/14/main/postgresql.conf; echo "host all all 0.0.0.0/0 trust" >> /etc/postgresql/14/main/pg_hba.conf; /etc/init.d/postgresql start > /tmp/db_pg.log 2>&1 && sleep 1 && su - postgres -c "psql -c \\"CREATE USER lms WITH PASSWORD \'lms123\';\\" && psql -c \\"CREATE DATABASE lms OWNER lms;\\""')
    time.sleep(1)

    info(f'*** [2/6] Kiem tra Node.js Dependencies...\n')
    # Cài đặt npm dependencies nếu chưa có (chỉ cần làm 1 lần)
    first_backend = net.get(BACKENDS[0]["name"])
    check_cmd = 'ls /work/lms/backend/node_modules/ > /dev/null 2>&1 || echo "NEED_INSTALL"'
    if "NEED_INSTALL" in first_backend.cmd(check_cmd):
        info('    Running npm install... (this may take 30-60 seconds)\n')
        first_backend.cmd(f'cd /work/lms/backend && npm install --silent 2>&1')
    else:
        info('    node_modules found. Skipping npm install.\n')
    time.sleep(1)
    
    info(f'*** [3/6] Seed Du Lieu (Tu {BACKENDS[0]["name"]})...\n')
    first_backend.cmd(f'cd /work/lms/backend && DB_HOST={DB_IP} npm run seed:massive > /tmp/seed.log 2>&1')
    time.sleep(2)
    
    info(f'*** [4/6] Khoi dong {len(BACKENDS)} Instances Backend Node.js...\n')
    
    for node in BACKENDS:
        name = node["name"]
        host = net.get(name)
        host.cmd(f'cd /work/lms/backend && DB_HOST={DB_IP} PORT=4000 node server.js > /tmp/{name}_api.log 2>&1 &')
        info(f'    {name} dang chay Backend o port 4000 (Tro ve DB {DB_IP})\n')
        
    time.sleep(3)

    from config import BACKENDS, DB_NODE, DB_IP, VIP, TEST_CLIENTS

    info('*** [5/6] Kich hoat Artillery Stress Test toi AI Load Balancer (VIP)...\n')
    target_apis = [f'{VIP}:4000']
    
    for i, client_name in enumerate(TEST_CLIENTS):
        host = net.get(client_name)
        target = target_apis[i % len(target_apis)]
        info(f'    Khoi chay Labeled Stress Test ({SCENARIO_FILE}) tren {client_name} (Muc tieu: {target})...\n')
        host.cmd(f'cd /work/lms/stress-test && SCENARIO="{SCENARIO_FILE}" TARGET="http://{target}" python3 -u run_labeled_test.py > /tmp/{client_name}_stress.log 2>&1 &')
        
    info(f'\n*** [6/6] Qua trinh ep traffic theo kich ban {SCENARIO_FILE} da bat dau!\n')
    info('    Script se TU DONG dung khi Artillery hoan tat. Bro chi can ngoi cho.\n')
    info('    (Nhan Ctrl+C bat ky luc nao de vao Mininet CLI thu cong)\n\n')

def wait_for_artillery(net):
    """Doi tat ca tien trinh Artillery tren clients ket thuc."""
    from config import TEST_CLIENTS
    check_hosts = [net.get(name) for name in TEST_CLIENTS]
    start_time = time.time()

    info('\n*** Doi Artillery Stress Test ket thuc...\n')
    while True:
        all_done = True
        for host in check_hosts:
            # Kiem tra xem tien trinh python3 co con chay khong
            # ps -ef | grep python3 | grep run_labeled_test.py | grep -v grep
            # Neu lenh tra ve rong, tuc la tien trinh da ket thuc
            output = host.cmd('ps -ef | grep python3 | grep run_labeled_test.py | grep -v grep')
            if output.strip():
                all_done = False
                break
        
        if all_done:
            info(f'\n*** Tat ca Artillery Stress Test da ket thuc sau {int(time.time() - start_time)} giay!\n')
            break
        
        info('.')
        time.sleep(5)

if __name__ == '__main__':
    from config import TEST_CLIENTS, DB_NODE, DB_IP, VIP
    setLogLevel('info')

    topo = FatTree()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        host=Host,
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
    info(f'*** Cai VIP redirect flow (priority=50) tren tat ca switches (VIP: {VIP})...\n')
    for sw in net.switches:
        sw.cmd(f'ovs-ofctl add-flow {sw.name} "priority=50,ip,nw_dst={VIP},actions=CONTROLLER:65535" -O OpenFlow13')
        sw.cmd(f'ovs-ofctl add-flow {sw.name} "priority=50,arp,arp_tpa={VIP},actions=CONTROLLER:65535" -O OpenFlow13')
    info('    Done! VIP redirect flows installed on all %d switches.\n' % len(net.switches))

    info(f'*** Ping kiem tra ket noi toi DB Server ({DB_NODE}) cuoi cung...\n')
    h1 = net.get('h1')
    db_node_host = net.get(DB_NODE)
    h1.cmd(f'arp -s {DB_IP} {db_node_host.MAC()}')
    
    connected = False
    for i in range(10):
        res = h1.cmd(f'ping -c 1 {DB_IP}')
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
