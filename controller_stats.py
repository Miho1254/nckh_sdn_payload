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


# ═══════════════════════════════════════════════════════════
#  CONTROLLER CHÍNH
# ═══════════════════════════════════════════════════════════

class FatTreeController(app_manager.RyuApp):
    """Ryu OpenFlow 1.3 controller: L2 learning switch + stats collector."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MAC address table: { dpid: { mac: port } }
        self.mac_to_port = {}

        # Theo dõi các datapath đang kết nối
        self.datapaths = {}

        # Khởi tạo thư mục & CSV headers
        self._init_csv_files()

        # Tạo green thread thu thập stats
        self.monitor_thread = hub.spawn(self._monitor_loop)

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
        self.logger.info('Switch s%s connected — table-miss flow installed', datapath.id)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Xử lý packet chưa có flow rule: học MAC, quyết định flood/forward."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Bỏ qua LLDP
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        self.mac_to_port.setdefault(dpid, {})

        # Học: source MAC → in_port
        self.mac_to_port[dpid][src_mac] = in_port

        # Quyết định output port
        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Nếu đã biết đích → cài flow rule để switch xử lý trực tiếp
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            # idle_timeout=60: xóa flow nếu 60s không match
            # hard_timeout=180: xóa flow sau 180s bất kể
            self._add_flow(datapath, priority=1, match=match, actions=actions,
                           idle_timeout=60, hard_timeout=180)

        # Gửi packet ra
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        """Helper: Cài một flow entry lên switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

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

        with open(FLOW_STATS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for stat in sorted(body, key=lambda s: (s.table_id, s.priority)):
                # Trích xuất match fields
                in_port = stat.match.get('in_port', '')
                eth_src = stat.match.get('eth_src', '')
                eth_dst = stat.match.get('eth_dst', '')

                writer.writerow([
                    timestamp,
                    dpid,
                    stat.table_id,
                    stat.priority,
                    in_port,
                    eth_src,
                    eth_dst,
                    stat.packet_count,
                    stat.byte_count,
                    stat.duration_sec,
                    stat.duration_nsec,
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
                ])

        self.logger.debug('Port stats collected from s%s: %d ports', dpid, len(body))

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
                    'packet_count',
                    'byte_count',
                    'duration_sec',
                    'duration_nsec',
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
                ])

        self.logger.info('CSV output: %s, %s', FLOW_STATS_CSV, PORT_STATS_CSV)
