"""
Ryu Controller cho Fat-Tree SDN — Thu thập thống kê & Chuyển mạch L2.

Hai nhiệm vụ chính:
  1. Learning Switch (L2): Học MAC → port, flood nếu chưa biết.
  2. Stats Collector: Mỗi 10 giây gửi FlowStatsRequest & PortStatsRequest
     tới tất cả switches, ghi kết quả ra CSV.

Chạy:
    ryu-manager controller_stats.py --verbose

Ref: Paper "A Transformer-Based Deep Q-Learning" — Module 3, Section V.E
"""

import csv
import datetime
import os
import time
import threading
import numpy as np

try:
    import torch
    from ai_model.tft_dqn_net import TFT_DQN_Model
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("WARNING: PyTorch or TFT_DQN_Model not found! Running in dumb mode.")

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from ryu.lib import hub
from ryu.lib.packet import arp, ethernet, ether_types, packet
from ryu.ofproto import ofproto_v1_3
from operator import attrgetter


# ── Đường dẫn output CSV ────────────────────────────────────

STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats')
FLOW_STATS_CSV = os.path.join(STATS_DIR, 'flow_stats.csv')
PORT_STATS_CSV = os.path.join(STATS_DIR, 'port_stats.csv')

POLL_INTERVAL = 10  # giây


# ── Định nghĩa Backend API & Virtual IP ──────────────────────
VIP = "10.0.0.100"
VMAC = "00:00:00:00:01:00" # MAC ảo cho Load Balancer

BACKENDS = [
    {"ip": "10.0.0.5", "mac": "00:00:00:00:00:05", "port": 5, "weight": 1},
    {"ip": "10.0.0.7", "mac": "00:00:00:00:00:07", "port": 7, "weight": 2},
    {"ip": "10.0.0.8", "mac": "00:00:00:00:00:08", "port": 8, "weight": 3}
]

# Thuật toán LB mặc định (RR, WRR, AI, COLLECT)
ALGO = os.environ.get('LB_ALGO', 'RR').upper()

# ═══════════════════════════════════════════════════════════
#  CONTROLLER CHÍNH
# ═══════════════════════════════════════════════════════════

class FatTreeController(app_manager.RyuApp):
    """Ryu OpenFlow 1.3: L2 Switch + Stats Collector + AI Load Balancer."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mac_to_port = {}
        self.datapaths = {}

        self._init_csv_files()
        
        # Biến điều phối Round Robin
        self.rr_index = 0
        
        # Biến điều phối Weighted Round Robin
        self.wrr_sequence = []
        for i, b in enumerate(BACKENDS):
            self.wrr_sequence.extend([i] * b.get('weight', 1))
        self.wrr_index = 0

        # Biến chọn Backend hiện tại (Dùng cho AI hoặc Sync)
        self.current_best_backend_idx = 0 
        self.ai_agent = None
        
        # Real-time state tracking for AI
        self.stats_lock = threading.Lock()
        self.dp_current_bytes = {dpid: 0 for dpid in range(7, 11)}
        self.dp_current_packets = {dpid: 0 for dpid in range(7, 11)}
        
        self.total_bytes_prev = 0
        self.total_packets_prev = 0
        self.last_update_time = time.time()
        
        # Per-server load tracking (from s8 port stats)
        self.server_tx_bytes = {2: 0, 4: 0, 5: 0}  # port_no -> tx_bytes (s8)
        self.server_tx_prev = {2: 0, 4: 0, 5: 0}
        self.norm_load_h5 = 0.0
        self.norm_load_h7 = 0.0
        self.norm_load_h8 = 0.0
        
        # Normalized inputs for AI
        self.norm_byte_rate = 0.5
        self.norm_packet_rate = 0.5

        # Init AI Model (Chỉ init nếu mode là AI)
        if ALGO == 'AI':
            self._init_ai_model()

        self.logger.info(f"🚀 RIU Load Balancer started with Strategy: {ALGO}")

        # Threads
        self.monitor_thread = hub.spawn(self._monitor_loop)
        if ALGO == 'AI' and self.ai_agent is not None:
            self.ai_thread = hub.spawn(self._ai_inference_loop)

    @set_ev_cls(ofp_event.EventOFPErrorMsg, MAIN_DISPATCHER)
    def error_msg_handler(self, ev):
        """Bắt lỗi OpenFlow từ switch."""
        msg = ev.msg
        self.logger.error('[OFP_ERROR] type=0x%02x code=0x%02x on s%s',
                          msg.type, msg.code, msg.datapath.id)

    # ─────────────────────────────────────────────────────────
    #  PHẦN 1: LEARNING SWITCH (Cảnh sát giao thông)
    # ─────────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Khi switch kết nối: cài table-miss flow entry (gửi tới controller)."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Table-miss: match tất cả → gửi tới controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, priority=0, match=match, actions=actions)

        # VIP redirect: BẤT KỲ gói IP nào đi tới VIP → gửi lên controller
        # Priority=50 > L2 learning (1) nhưng < NAT flow (100)
        # Đảm bảo traffic tới VIP luôn đi qua LB logic, không bị L2 forward nhầm
        vip_match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=VIP)
        vip_actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, priority=50, match=vip_match, actions=vip_actions)

        # ARP cho VIP cũng cần redirect lên controller
        arp_vip_match = parser.OFPMatch(eth_type=0x0806, arp_tpa=VIP)
        self._add_flow(datapath, priority=50, match=arp_vip_match, actions=vip_actions)

        self.logger.info('Switch s%s connected — table-miss + VIP redirect installed', datapath.id)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Xử lý packet chưa có flow: Chọn Backend, cài NAT + Xử lý ARP VIP."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        if eth.dst[:8] == '01:80:c2': # Chặn STP (Spanning Tree) để OVS tự xử lý
            return
        if eth.ethertype == ether_types.ETH_TYPE_IPV6: # Loại bỏ nhiễu IPv6
            return

        # Log traffic thực tế để debug
        self.logger.info("[PKT_IN] Switch: s%s | src:%s -> dst:%s | Port:%s", 
                         dpid, eth.src, eth.dst, in_port)

        # -------------------------------------------------------------
        # XỬ LÝ ARP: "Kẻ hủy diệt" Load Balancer (Đóng vai Virtual MAC)
        # -------------------------------------------------------------
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            # Nếu ai đó hỏi "Ai là 10.0.0.100?"
            if arp_pkt.dst_ip == VIP and arp_pkt.opcode == arp.ARP_REQUEST:
                self._send_arp_reply(datapath, in_port, eth.src, arp_pkt.src_ip)
                return
            
        # -------------------------------------------------------------
        # L2 LEARNING CƠ BẢN
        # -------------------------------------------------------------
        src_mac = eth.src
        dst_mac = eth.dst
        self.mac_to_port.setdefault(dpid, {})
        # Không learn VMAC vào bảng MAC (VMAC là ảo, do controller tạo ra)
        if src_mac != VMAC:
            self.mac_to_port[dpid][src_mac] = in_port

        # -------------------------------------------------------------
        # ĐIỀU HƯỚNG TỚI VIRTUAL IP (LOAD BALANCING)
        # -------------------------------------------------------------
        from ryu.lib.packet import ipv4, tcp, udp
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        if ipv4_pkt and ipv4_pkt.dst == VIP:
            # --- CHỌN BACKEND THEO THUẬT TOÁN ---
            if ALGO == 'AI':
                idx = self.current_best_backend_idx
            elif ALGO == 'WRR':
                idx = self.wrr_sequence[self.wrr_index]
                self.wrr_index = (self.wrr_index + 1) % len(self.wrr_sequence)
            else:  # RR hoặc COLLECT
                idx = self.rr_index
                self.rr_index = (self.rr_index + 1) % len(BACKENDS)

            best_backend = BACKENDS[idx]
            self.logger.info(
                "[LB] algo=%s → backend=%s | src=%s",
                ALGO, best_backend['ip'], ipv4_pkt.src
            )

            client_mac = pkt.get_protocol(ethernet.ethernet).src
            backend_mac = best_backend['mac']
            
            # Lookup port for backend
            if backend_mac in self.mac_to_port.get(datapath.id, {}):
                out_port_in = self.mac_to_port[datapath.id][backend_mac]
            else:
                out_port_in = ofproto.OFPP_FLOOD
                
            # Lookup port for client
            if client_mac in self.mac_to_port.get(datapath.id, {}):
                out_port_out = self.mac_to_port[datapath.id][client_mac]
            else:
                out_port_out = ofproto.OFPP_FLOOD

            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)

            if tcp_pkt:
                match_in = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=0x0800,
                    ip_proto=6,
                    ipv4_src=ipv4_pkt.src,
                    ipv4_dst=VIP,
                    tcp_src=tcp_pkt.src_port,
                )
                match_out = parser.OFPMatch(
                    eth_type=0x0800,
                    ip_proto=6,
                    ipv4_src=best_backend['ip'],
                    ipv4_dst=ipv4_pkt.src,
                    tcp_dst=tcp_pkt.src_port,
                )
            elif udp_pkt:
                match_in = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=0x0800,
                    ip_proto=17,
                    ipv4_src=ipv4_pkt.src,
                    ipv4_dst=VIP,
                    udp_src=udp_pkt.src_port,
                )
                match_out = parser.OFPMatch(
                    eth_type=0x0800,
                    ip_proto=17,
                    ipv4_src=best_backend['ip'],
                    ipv4_dst=ipv4_pkt.src,
                    udp_dst=udp_pkt.src_port,
                )
            else:
                match_in = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=0x0800,
                    ipv4_src=ipv4_pkt.src,
                    ipv4_dst=VIP,
                )
                match_out = parser.OFPMatch(
                    eth_type=0x0800,
                    ipv4_src=best_backend['ip'],
                    ipv4_dst=ipv4_pkt.src,
                )

            actions_in = [
                parser.OFPActionSetField(eth_dst=backend_mac),
                parser.OFPActionSetField(ipv4_dst=best_backend['ip']),
                parser.OFPActionOutput(out_port_in)
            ]

            actions_out = [
                parser.OFPActionSetField(eth_src=VMAC),
                parser.OFPActionSetField(ipv4_src=VIP),
                parser.OFPActionOutput(out_port_out)
            ]

            # QUAN TRỌNG: Cài match_out (return path) TRƯỚC match_in
            # để SYN-ACK trả về từ backend LUÔN được rewrite src→VIP
            # trước khi PacketOut gửi gói tin đầu tiên đi.
            # FIX: idle_timeout=0 để flow NAT tồn tại suốt quá trình thí nghiệm
            # (trước đây idle_timeout=30 khiến flow bị xóa trước khi stats collection)
            self._add_flow(datapath, priority=100, match=match_out,
                           actions=actions_out, idle_timeout=0)
            self._add_flow(datapath, priority=100, match=match_in,
                           actions=actions_in, idle_timeout=0)

            # Barrier: chờ OVS xác nhận flow đã install xong
            datapath.send_msg(parser.OFPBarrierRequest(datapath))

            # Forward gói tin đầu tiên ngay lập tức
            data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
            out = parser.OFPPacketOut(
                datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
                actions=actions_in, data=data)
            datapath.send_msg(out)
            return


        # -------------------------------------------------------------
        # FORWARDING BÌNH THƯỜNG (Các gói không dính tới VIP)
        # -------------------------------------------------------------

        # QUAN TRỌNG: Không learn VMAC vào bảng MAC → tránh tạo flow L2
        # cho traffic tới VIP (những gói này phải LUÔN đi qua controller)
        if dst_mac == VMAC:
            # Gói tới VMAC mà chưa match VIP check ở trên → drop silently
            return

        out_port = self.mac_to_port[dpid].get(dst_mac, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            self._add_flow(datapath, priority=1, match=match, actions=actions,
                           idle_timeout=60, hard_timeout=180)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def _send_arp_reply(self, datapath, in_port, dst_mac, dst_ip):
        """Trả lời ARP: Ai là 10.0.0.100? Dạ tôi đây (VMAC)"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_ARP,
            dst=dst_mac,
            src=VMAC
        ))
        pkt.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=VMAC,
            src_ip=VIP,
            dst_mac=dst_mac,
            dst_ip=dst_ip
        ))
        pkt.serialize()

        actions = [parser.OFPActionOutput(in_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=pkt.data
        )
        datapath.send_msg(out)

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        """Helper: Cài một flow entry lên switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.info('[FLOW_MOD] s%s priority=%s match=%s idle=%s',
                         datapath.id, priority, match, idle_timeout)

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    # ─────────────────────────────────────────────────────────
    #  PHẦN 2: STATS COLLECTOR (Thám tử thu thập dữ liệu)
    # ─────────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """Theo dõi switch connect/disconnect để biết hỏi thăm ai."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Datapath s%s registered for monitoring', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Datapath s%s unregistered', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor_loop(self):
        """Green thread: cứ mỗi 10 giây gửi request stats tới tất cả switches."""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _request_stats(self, datapath):
        """Gửi FlowStatsRequest và PortStatsRequest tới một switch."""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # Flow stats
        flow_req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(flow_req)

        # Port stats
        port_req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(port_req)

    # ── Flow Stats Reply ────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Nhận flow stats → ghi vào CSV."""
        timestamp = datetime.datetime.now().isoformat()
        dpid = ev.msg.datapath.id
        body = ev.msg.body

        # Aggregate current active NAT flows (s7-s10)
        if 7 <= dpid <= 10:
            dp_bytes = 0
            dp_packets = 0
            for stat in body:
                if stat.priority == 100: # NAT flows
                    dp_bytes += stat.byte_count
                    dp_packets += stat.packet_count
            with self.stats_lock:
                self.dp_current_bytes[dpid] = dp_bytes
                self.dp_current_packets[dpid] = dp_packets

        with open(FLOW_STATS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for stat in sorted(body, key=lambda s: (s.table_id, s.priority)):
                # Trích xuất match fields
                in_port = stat.match.get('in_port', '')
                eth_src = stat.match.get('eth_src', '')
                eth_dst = stat.match.get('eth_dst', '')
                ipv4_src = stat.match.get('ipv4_src', '')
                ipv4_dst = stat.match.get('ipv4_dst', '')

                writer.writerow([
                    timestamp,
                    dpid,
                    stat.table_id,
                    stat.priority,
                    in_port,
                    eth_src,
                    eth_dst,
                    ipv4_src,
                    ipv4_dst,
                    stat.packet_count,
                    stat.byte_count,
                    stat.duration_sec,
                    stat.duration_nsec,
                    self._get_current_label(),
                ])

        self.logger.debug('Flow stats collected from s%s: %d entries', dpid, len(body))

    # ── Port Stats Reply ────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Nhận port stats → ghi vào CSV."""
        timestamp = datetime.datetime.now().isoformat()
        dpid = ev.msg.datapath.id
        body = ev.msg.body

        with open(PORT_STATS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for stat in sorted(body, key=attrgetter('port_no')):
                writer.writerow([
                    timestamp,
                    dpid,
                    stat.port_no,
                    stat.rx_packets,
                    stat.rx_bytes,
                    stat.rx_errors,
                    stat.rx_dropped,
                    stat.tx_packets,
                    stat.tx_bytes,
                    stat.tx_errors,
                    stat.tx_dropped,
                    stat.collisions,
                    stat.duration_sec,
                    stat.duration_nsec,
                    self._get_current_label(),
                ])

        self.logger.debug('Port stats collected from s%s: %d ports', dpid, len(body))

        # Track per-server tx_bytes from s8 for AI inference
        if dpid == 8:
            with self.stats_lock:
                for stat in body:
                    if stat.port_no in self.server_tx_bytes:
                        self.server_tx_bytes[stat.port_no] = stat.tx_bytes

    # ─────────────────────────────────────────────────────────
    #  CSV INITIALIZATION
    # ─────────────────────────────────────────────────────────

    def _init_csv_files(self):
        """Tạo thư mục stats/ và ghi header cho CSV nếu file chưa tồn tại."""
        os.makedirs(STATS_DIR, exist_ok=True)

        if not os.path.exists(FLOW_STATS_CSV):
            with open(FLOW_STATS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'datapath_id',
                    'table_id',
                    'priority',
                    'in_port',
                    'eth_src',
                    'eth_dst',
                    'ipv4_src',
                    'ipv4_dst',
                    'packet_count',
                    'byte_count',
                    'duration_sec',
                    'duration_nsec',
                    'label',
                ])

        if not os.path.exists(PORT_STATS_CSV):
            with open(PORT_STATS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'datapath_id',
                    'port_no',
                    'rx_packets',
                    'rx_bytes',
                    'rx_errors',
                    'rx_dropped',
                    'tx_packets',
                    'tx_bytes',
                    'tx_errors',
                    'tx_dropped',
                    'collisions',
                    'duration_sec',
                    'duration_nsec',
                    'label',
                ])

        self.logger.info('CSV output: %s, %s', FLOW_STATS_CSV, PORT_STATS_CSV)
        self._update_label_file('NORMAL')  # Reset label to NORMAL on start

    def _get_current_label(self):
        """Đọc label hiện tại từ file shared."""
        label_file = os.path.join(STATS_DIR, 'current_label.txt')
        try:
            with open(label_file, 'r') as f:
                return f.read().strip() or 'NORMAL'
        except:
            return 'NORMAL'

    def _update_label_file(self, label):
        """Ghi label vào file (cho script runner dùng hoặc init)."""
        label_file = os.path.join(STATS_DIR, 'current_label.txt')
        try:
            with open(label_file, 'w') as f:
                f.write(label)
        except Exception as e:
            self.logger.error("Failed to write label file: %s", e)

    # ─────────────────────────────────────────────────────────
    #  PHẦN 3: AI INFERENCE MODULE (Não Bộ)
    # ─────────────────────────────────────────────────────────

    def _init_ai_model(self):
        """Khởi tạo mô hình TFT-DQN từ Checkpoint."""
        if not AI_AVAILABLE:
            return
            
        model_path = os.path.join(os.path.dirname(__file__), 'ai_model', 'checkpoints', 'tft_dqn_master.pth')
        if not os.path.exists(model_path):
            self.logger.warning(f"AI Model not found at {model_path}! Will fallback to Round Robin.")
            return

        try:
            # Hyperparams PHẢI KHỚP với train.py: hidden_size=32, num_actions=3
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            import sys
            sys.path.append(os.path.dirname(__file__))
            from ai_model.tft_dqn_net import TFT_DQN_Model
            
            self.ai_agent = TFT_DQN_Model(
                input_size=5, 
                seq_len=5, 
                hidden_size=32, 
                num_actions=3
            ).to(self.device)
            
            self.ai_agent.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.ai_agent.eval()
            self.logger.info(f"✅ AI Model TFT-DQN V3 (Boltzmann) loaded on {self.device}. (5-feature state)")
            
            # Boltzmann temperature for inference (khớp với training V3)
            self.ai_temperature = 0.5
            
            # Ring Buffer 5 timesteps x 5 features
            self.state_buffer = [[0.5, 0.5, 0.33, 0.33, 0.33] for _ in range(5)]
            
        except Exception as e:
            self.logger.error(f"Failed to load AI model: {e}")
            self.ai_agent = None


    def _ai_inference_loop(self):
        """Green Thread suy luận: Đọc dữ liệu mô phỏng, gọi Pytorch."""
        # Khởi tạo file CSV log inference
        inference_csv = os.path.join(STATS_DIR, 'inference_log.csv')
        with open(inference_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'inference_ms', 'action', 'q_values', 'switched'])
        
        while True:
            start_time = time.time()
            
            # Calculate real deltas
            with self.stats_lock:
                duration = time.time() - self.last_update_time
                if duration > 1.0:
                    active_total_bytes = sum(self.dp_current_bytes.values())
                    active_total_packets = sum(self.dp_current_packets.values())
                    
                    byte_diff = max(0, active_total_bytes - self.total_bytes_prev)
                    packet_diff = max(0, active_total_packets - self.total_packets_prev)
                    
                    self.total_bytes_prev = active_total_bytes
                    self.total_packets_prev = active_total_packets
                    self.last_update_time = time.time()
                    
                    self.norm_byte_rate = min(0.95, (byte_diff * 8 / duration) / 1e8)
                    self.norm_packet_rate = min(0.95, (packet_diff / duration) / 1e4)
                    
                    # Per-server load deltas
                    for port in [2, 4, 5]:
                        delta = max(0, self.server_tx_bytes[port] - self.server_tx_prev[port])
                        self.server_tx_prev[port] = self.server_tx_bytes[port]
                        norm_load = min(0.95, (delta * 8 / duration) / 1e8)
                        if port == 2:
                            self.norm_load_h5 = norm_load
                        elif port == 4:
                            self.norm_load_h7 = norm_load
                        elif port == 5:
                            self.norm_load_h8 = norm_load

            current_byte_rate = self.norm_byte_rate
            current_packet_rate = self.norm_packet_rate
            
            self.state_buffer.pop(0)
            self.state_buffer.append([
                current_byte_rate, 
                current_packet_rate,
                self.norm_load_h5,
                self.norm_load_h7,
                self.norm_load_h8
            ])
            
            state_tensor = torch.tensor([self.state_buffer], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                q_values, _ = self.ai_agent(state_tensor)
                
                # Boltzmann (Softmax) Selection — khớp với training V3
                # Khi Q-values gần nhau → phân bố đều → cân bằng tải
                # Khi Q-values khác biệt → ưu tiên server tốt nhất
                q_np = q_values.cpu().numpy()[0]
                q_scaled = q_np / max(self.ai_temperature, 0.01)
                q_scaled = q_scaled - np.max(q_scaled)  # Stability
                exp_q = np.exp(q_scaled)
                probs = exp_q / exp_q.sum()
                action = np.random.choice(3, p=probs)
                
            old_backend = self.current_best_backend_idx
            self.current_best_backend_idx = action
            
            inference_latency = (time.time() - start_time) * 1000
            switched = (action != old_backend)
            
            # Ghi vào CSV
            with open(inference_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    f"{inference_latency:.3f}",
                    action,
                    q_values.cpu().numpy().tolist(),
                    int(switched)
                ])
            
            if switched:
                best_ip = BACKENDS[action]['ip']
                self.logger.info(
                    f"\n=======================================================\n"
                    f"[AI_LOG] TFT-DQN V3 Boltzmann Decision\n"
                    f"-> Switching backend to {best_ip} (Action: {action})\n"
                    f"-> Q-values: {q_np.tolist()}\n"
                    f"-> Probabilities: [{probs[0]:.1%}, {probs[1]:.1%}, {probs[2]:.1%}]\n"
                    f"-> Inference Latency: {inference_latency:.2f} ms\n"
                    f"=======================================================\n"
                )
            
            hub.sleep(5.0)

