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
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("WARNING: PyTorch not found! Running in dumb mode.")

try:
    from ai_model.tft_ac_net import TFT_ActorCritic_Model
    AC_AVAILABLE = True
except ImportError:
    AC_AVAILABLE = False

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

from config import BACKENDS, VIP, VMAC, POLL_INTERVAL

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
        self.ai_probs = [1.0 / len(BACKENDS) for _ in BACKENDS]  # Phân phối xác suất mặc định đều
        self.ai_agent = None
        
        # Real-time state tracking for AI
        self.stats_lock = threading.Lock()
        self.dp_current_bytes = {dpid: 0 for dpid in range(7, 11)}
        self.dp_current_packets = {dpid: 0 for dpid in range(7, 11)}
        
        self.total_bytes_prev = 0
        self.total_packets_prev = 0
        self.last_update_time = time.time()
        
        # Per-server load tracking mapping (port_no, dpid) -> tx_bytes
        # Dùng dictionary key là tupple (dpid, port) để nhận dạng độc lập
        self.server_tx_bytes = {(b['dpid'], b['port']): 0 for b in BACKENDS}
        self.server_tx_prev = {(b['dpid'], b['port']): 0 for b in BACKENDS}
        self.norm_loads = [0.0] * len(BACKENDS)
        
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
                # Chốt: Dùng xác suất trả về từ AI để chọn server ngẫu nhiên, y hệt như WRR
                # nhưng có tính "động". Nếu AI thấy h8 ngon, nó sẽ buff prob lên cao.
                idx = np.random.choice(len(BACKENDS), p=self.ai_probs)
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

        # Aggregate current active NAT flows (Edge switches)
        from config import EDGE_DPIDS
        if dpid in EDGE_DPIDS:
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

        # Track per-server tx_bytes flexibly from any switches they are attached to
        with self.stats_lock:
            for stat in body:
                key = (dpid, stat.port_no)
                if key in self.server_tx_bytes:
                    self.server_tx_bytes[key] = stat.tx_bytes

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
        """Auto-detect TFT-AC (CQL) hoặc TFT-DQN checkpoint."""
        if not AI_AVAILABLE:
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = None  # 'ac' or 'dqn'
        num_actions = len(BACKENDS)

        # ── Thử load TFT-AC (CQL) model trước ──
        custom_path = os.environ.get('AI_CHECKPOINT_PATH')
        ac_path = os.path.join(os.path.dirname(__file__), 'ai_model', 'checkpoints', 'tft_ac_best.pth')
        
        paths_to_try = [custom_path] if custom_path else [ac_path]

        for path in paths_to_try:
            if path and AC_AVAILABLE and os.path.exists(path):
                try:
                    ckpt = torch.load(path, map_location=self.device, weights_only=False)
                    # Auto-detect input_size from checkpoint
                    state_dict = ckpt.get('model_state_dict', ckpt)
                    # VSN weight_grn fc1 weight shape tells us input_size
                    vsn_key = 'vsn.weight_grn.fc1.weight'
                    if vsn_key in state_dict:
                        input_size = state_dict[vsn_key].shape[1]
                    else:
                        input_size = 42  # default v3

                    hidden_key = 'lstm.weight_ih_l0'
                    hidden_size = state_dict[hidden_key].shape[0] // 4 if hidden_key in state_dict else 64

                    self.ai_agent = TFT_ActorCritic_Model(
                        input_size=input_size, seq_len=5,
                        hidden_size=hidden_size, num_actions=num_actions
                    ).to(self.device)
                    self.ai_agent.load_state_dict(state_dict)
                    self.ai_agent.eval()
                    self.model_type = 'ac'
                    self.model_input_size = input_size
                    self.state_buffer = [[0.0] * input_size for _ in range(5)]
                    self.logger.info(f"TFT-AC (CQL) model loaded: {input_size} features, {hidden_size} hidden")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load TFT-AC from {path}: {e}")

        # No legacy TFT-DQN support – only TFT‑CQL (Actor‑Critic) is used.
        self.logger.warning("Legacy TFT-DQN model not supported. AI disabled.")
        return

    def _ai_inference_loop(self):
        """Green Thread: suy luận AI (hỗ trợ cả TFT-AC và TFT-DQN)."""
        inference_csv = os.path.join(STATS_DIR, 'inference_log.csv')
        with open(inference_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'inference_ms', 'action', 'model_type',
                             'probs', 'safety_override', 'switched'])

        while True:
            start_time = time.time()

            # ── Collect real-time state ──
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

                    from config import SCALING_BYTE_RATE, SCALING_PKT_RATE, SCALING_LOAD
                    self.norm_byte_rate = min(1.0, (byte_diff * 8 / duration) / SCALING_BYTE_RATE)
                    self.norm_packet_rate = min(1.0, (packet_diff / duration) / SCALING_PKT_RATE)

                    for i, b in enumerate(BACKENDS):
                        key = (b['dpid'], b['port'])
                        delta = max(0, self.server_tx_bytes.get(key, 0) - self.server_tx_prev.get(key, 0))
                        self.server_tx_prev[key] = self.server_tx_bytes.get(key, 0)
                        self.norm_loads[i] = min(1.0, (delta * 8 / duration) / SCALING_LOAD)

            # ── Build state vector ──
            self.state_buffer.pop(0)
            if self.model_type == 'ac':
                # V3 features: pad to input_size if needed
                feature_vec = [self.norm_byte_rate, self.norm_packet_rate] + self.norm_loads
                # Pad remaining features with 0 (utilization, headroom etc. computed offline)
                while len(feature_vec) < self.model_input_size:
                    feature_vec.append(0.0)
                self.state_buffer.append(feature_vec[:self.model_input_size])
            elif self.model_input_size == 2:
                self.state_buffer.append([self.norm_byte_rate, self.norm_packet_rate])
            else:
                self.state_buffer.append(
                    [self.norm_byte_rate, self.norm_packet_rate] + self.norm_loads)

            state_tensor = torch.tensor([self.state_buffer], dtype=torch.float32).to(self.device)
            safety_override = False

            with torch.no_grad():
                if self.model_type == 'ac':
                    # TFT-AC: Serving Policy Selection
                    # For load balancing, policy distribution IS the routing ratio, but test argmax too
                    # [0.06, 0.31, 0.63] = "send 6% to h5, 31% to h7, 63% to h8"
                    probs = self.ai_agent.get_policy(state_tensor).cpu().numpy()[0]
                    probs = np.clip(probs, 1e-6, None)
                    probs = probs / probs.sum()
                    
                    ai_serving_rule = os.environ.get('AI_SERVING_RULE', 'sampled').lower()
                    if ai_serving_rule == 'argmax':
                        action = int(np.argmax(probs))
                    else:
                        action = int(np.random.choice(len(probs), p=probs))

                    # Safety mask: kiểm tra utilization server được chọn
                    from config import SAFETY_THRESHOLD, CAPACITIES, SCALING_LOAD
                    chosen_load = self.norm_loads[action] if action < len(self.norm_loads) else 0
                    chosen_util = (chosen_load * SCALING_LOAD) / (CAPACITIES[action] * 1e6) if CAPACITIES[action] > 0 else 0
                    if chosen_util > SAFETY_THRESHOLD:
                        # Fallback: chọn server có headroom lớn nhất
                        headrooms = []
                        for i in range(len(BACKENDS)):
                            load_i = self.norm_loads[i] if i < len(self.norm_loads) else 0
                            util_i = (load_i * SCALING_LOAD) / (CAPACITIES[i] * 1e6) if CAPACITIES[i] > 0 else 1.0
                            headrooms.append(1.0 - util_i)
                        action = int(np.argmax(headrooms))
                        safety_override = True

                else:
                    # No AI model loaded; fallback to weighted round robin.
                    action = self.wrr_sequence[self.wrr_index]
                    self.wrr_index = (self.wrr_index + 1) % len(self.wrr_sequence)
                    probs = [0.0] * len(BACKENDS)  # Default probs for fallback

            self.ai_probs = probs
            old_backend = self.current_best_backend_idx
            self.current_best_backend_idx = action

            inference_latency = (time.time() - start_time) * 1000
            switched = (action != old_backend)

            with open(inference_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    f"{inference_latency:.3f}",
                    action,
                    self.model_type or 'unknown',
                    [f"{p:.4f}" for p in probs],
                    int(safety_override),
                    int(switched)
                ])

            if switched:
                model_label = 'TFT-AC (CQL)' if self.model_type == 'ac' else 'Unknown'
                override_str = ' [SAFETY OVERRIDE]' if safety_override else ''
                prob_str = ", ".join([f"{p:.1%}" for p in probs])
                self.logger.info(
                    f"\n=======================================================\n"
                    f"[AI_LOG] {model_label} Decision{override_str}\n"
                    f"-> Backend: {BACKENDS[action]['ip']} (Action: {action})\n"
                    f"-> Policy: [{prob_str}]\n"
                    f"-> Latency: {inference_latency:.2f} ms\n"
                    f"=======================================================\n"
                )

            hub.sleep(5.0)

