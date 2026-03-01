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
    info('\n*** [1/5] Khởi động PostgreSQL Database trên h6...\n')
    h6 = net.get('h6')
    h6.cmd('su - postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main start" > /tmp/h6_pg.log 2>&1 &')
    time.sleep(3) # Chờ DB start

    info('*** [2/5] Seed Dữ Liệu (Từ h5)...\n')
    h5 = net.get('h5')
    h5.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 npm run seed:massive > /tmp/h5_seed.log 2>&1')
    
    info('*** [3/5] Khởi động 3 Instances Backend Node.js...\n')
    # API 1
    h5.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 PORT=4000 node server.js > /tmp/h5_api.log 2>&1 &')
    info('    ↳ h5 đang chạy Backend ở port 4000 (Trỏ về DB h6)\n')
    
    # API 2
    h7 = net.get('h7')
    h7.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 PORT=4000 node server.js > /tmp/h7_api.log 2>&1 &')
    info('    ↳ h7 đang chạy Backend ở port 4000 (Trỏ về DB h6)\n')
    
    # API 3
    h8 = net.get('h8')
    h8.cmd('cd /work/lms/backend && DB_HOST=10.0.0.6 PORT=4000 node server.js > /tmp/h8_api.log 2>&1 &')
    info('    ↳ h8 đang chạy Backend ở port 4000 (Trỏ về DB h6)\n')
    time.sleep(2) # Chờ API start

    info('*** [4/5] Kích hoạt Artillery Stress Test tới AI Load Balancer (10.0.0.100)...\n')
    # Tất cả Artillery đều trỏ thẳng vào Load Balancer Áo (10.0.0.100)
    target_apis = ['10.0.0.100:4000']
    
    for i in range(9, 17):
        host = net.get(f'h{i}')
        # Chia đều target API
        target = target_apis[i % len(target_apis)]
        info(f'    ↳ Khởi chạy Labeled Stress Test ({SCENARIO_FILE}) trên h{i} (Mục tiêu: {target})...\n')
        host.cmd(f'cd /work/lms/stress-test && SCENARIO="{SCENARIO_FILE}" TARGET="http://{target}" python3 run_labeled_test.py > /tmp/h{i}_stress.log 2>&1 &')
        
    info(f'\n*** [5/5] Quá trình ép traffic theo kịch bản {SCENARIO_FILE} đã bắt đầu! Đang vào màn hình Mininet CLI...\n')
    info('    Gợi ý: Thu thập file `stats/flow_stats.csv` khi test hoàn thành.\n\n')

if __name__ == '__main__':
    setLogLevel('info')

    # Lưu ý: Ryu Controller phải được chạy MỞ SẴN ở terminal khác.
    
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
    info('*** Đợi các Switches kết nối Controller (timeout 10s)...\n')
    net.waitConnected(timeout=10)

    info('*** Bật giao thức chống loop STP và chờ 30 giây hội tụ...\n')
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=true')
    time.sleep(30)

    # configure_queues(net)

    info('*** Ping cơ bản trước khi chạy thực tế...\n')
    net.pingAll()

    # Bắt đầu tự động hoá
    deploy_lms(net)

    CLI(net)
    net.stop()
