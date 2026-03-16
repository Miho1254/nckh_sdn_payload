import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

# ----------------------------------------------------------------------
# 1. ĐỊNH NGHĨA GIAO DIỆN (PROTOCOL) TRỪU TƯỢNG
# ----------------------------------------------------------------------

class BaseLoadBalancer(ABC):
    """
    Lớp cơ sở trừu tượng (Abstract Base Class) định nghĩa giao diện
    chung cho tất cả các thuật toán cân bằng tải.
    """
    
    @abstractmethod
    def select_server(self) -> Optional[str]:
        """
        Phương thức trừu tượng để chọn một server.
        
        Returns:
            Optional[str]: Tên của server được chọn, hoặc None nếu không có server nào.
        """
        pass

# ----------------------------------------------------------------------
# 2. TRIỂN KHAI CÁC THUẬT TOÁN
# ----------------------------------------------------------------------

class RoundRobin(BaseLoadBalancer):
    """
    Triển khai thuật toán cân bằng tải Round Robin (RR) an toàn.
    [cite_start][cite: 109-111, 117]
    """
    def __init__(self, servers: Optional[List[str]]):
        """
        Khởi tạo với một danh sách các server.
        
        Args:
            servers (Optional[List[str]]): Danh sách các server, ví dụ: ['S1', 'S2', 'S3']
        """
        self.servers: List[str] = servers if servers else []
        self.current_index: int = 0

    def select_server(self) -> Optional[str]:
        """
        Chọn server tiếp theo theo thứ tự vòng tròn.
        
        Returns:
            Optional[str]: Tên server, hoặc None nếu danh sách rỗng.
        """
        if not self.servers:
            return None
        
        server: str = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server

# Định nghĩa kiểu cho trạng thái server nội bộ của WRR
# (Tên server, Trọng số gốc, Trọng số hiện tại)
ServerStats = List[Union[str, int]]

class SmoothWeightedRoundRobin(BaseLoadBalancer):
    """
    Triển khai Smooth WRR (tương tự thuật toán của Nginx).
    Tránh "burst" request bằng cách phân phối xen kẽ.
    [cite_start][cite: 126-130]
    """
    def __init__(self, servers_with_weights: Optional[Dict[str, int]]):
        """
        Khởi tạo với dict {server: weight}.
        
        Args:
            servers_with_weights (Optional[Dict[str, int]]): 
                Ví dụ: {'S1': 3, 'S2': 2, 'S3': 1}.
                An toàn nếu truyền vào {} hoặc None.
        """
        self.servers: List[ServerStats] = []
        self.total_weight: int = 0
        
        if servers_with_weights:
            for server, weight in servers_with_weights.items():
                if weight > 0:
                    # [tên server, trọng số gốc, trọng số hiện tại]
                    self.servers.append([server, weight, 0])
                    self.total_weight += weight

    def select_server(self) -> Optional[str]:
        """
        Chọn server một cách "mượt" (smooth).
        
        Returns:
            Optional[str]: Tên server, hoặc None nếu danh sách rỗng.
        """
        if not self.servers:
            return None

        best_server: Optional[ServerStats] = None
        
        for server_stats in self.servers:
            # 1. Tăng trọng số hiện tại (current_weight)
            server_stats[2] = int(server_stats[2]) + int(server_stats[1]) # current_weight += original_weight

            # 2. Tìm server có current_weight cao nhất
            if best_server is None or server_stats[2] > best_server[2]:
                best_server = server_stats

        # 3. Giảm current_weight của server được chọn
        # best_server không thể là None ở đây nếu self.servers không rỗng
        best_server[2] = int(best_server[2]) - self.total_weight
        
        return str(best_server[0]) # Trả về TÊN server (phần tử [0])

# ----------------------------------------------------------------------
# 3. HÀM MÔ PHỎNG (SIMULATION)
# ----------------------------------------------------------------------

# Định nghĩa kiểu cho thông số server
ServerSpec = Dict[str, int]

def simulate(balancer: BaseLoadBalancer, 
             server_specs: Dict[str, ServerSpec], 
             requests: List[int]) -> None:
    # server_state lưu thời điểm (tick) mà server sẽ rảnh
    server_state: Dict[str, int] = {server: 0 for server in server_specs}
    
    current_tick: int = 0
    total_wait_time: int = 0
    
    print(f"\n--- Bắt đầu mô phỏng với {balancer.__class__.__name__} ---")
    
    for i, req_cost in enumerate(requests):
        current_tick += 1 # Giả sử mỗi tick có 1 request mới
        
        # 1. Balancer chọn server (Sử dụng giao diện trừu tượng)
        server_name: Optional[str] = balancer.select_server()
        
        # 2. Kiểm tra "chống đạn" (robustness check) nếu balancer không trả về server
        if server_name is None:
            print(f"[Tick {current_tick:02d}] Req {i+1} (cost {req_cost}) -> KHÔNG CÓ SERVER! Request bị HỦY.")
            continue 
        
        # 3. Kiểm tra xem server được chọn có trong danh sách thông số không
        if server_name not in server_specs:
            print(f"[Tick {current_tick:02d}] Req {i+1} (cost {req_cost}) -> Server '{server_name}' không có trong specs! HỦY.")
            continue

        # 4. Tính toán thời gian
        server_free_at: int = server_state[server_name]
        server_speed: int = server_specs[server_name]['processing_time']
        
        actual_processing_time: int = req_cost * server_speed
        
        start_time: int = max(current_tick, server_free_at)
        finish_time: int = start_time + actual_processing_time
        
        # 5. Cập nhật trạng thái server
        server_state[server_name] = finish_time
        
        # 6. Tính độ trễ (thời gian chờ)
        wait_time: int = start_time - current_tick
        total_wait_time += wait_time
        
        print(f"[Tick {current_tick:02d}] Req {i+1} (cost {req_cost}) -> {server_name}. "
              f"Server rảnh lúc {server_free_at:02d}. "
              f"Chờ {wait_time:02d} ticks. "
              f"Xử lý từ {start_time:02d} -> {finish_time:02d}.")
              
    print(f"--- Kết thúc: Tổng thời gian chờ = {total_wait_time} ticks ---")

# ----------------------------------------------------------------------
# 4. THỰC THI CÁC KỊCH BẢN
# ----------------------------------------------------------------------

# --- KỊCH BẢN 1: MÁY CHỦ KHÔNG ĐỒNG NHẤT ---
print("==============================================")
print("KỊCH BẢN 1: Máy chủ không đồng nhất (RR sập)")
print("==============================================")
servers_heterogeneous: Dict[str, ServerSpec] = {
    'S1': {'processing_time': 1},  # Nhanh
    'S2': {'processing_time': 1},  # Nhanh
    'S3': {'processing_time': 5}   # Chậm gấp 5 lần
}
requests_list: List[int] = [1] * 12

# --- Chạy RR ---
rr_balancer: BaseLoadBalancer = RoundRobin(list(servers_heterogeneous.keys()))
simulate(rr_balancer, servers_heterogeneous, requests_list)

# --- Chạy WRR (với cấu hình trọng số ĐÚNG) ---
weights_wrr: Dict[str, int] = {'S1': 5, 'S2': 5, 'S3': 1}
wrr_balancer: BaseLoadBalancer = SmoothWeightedRoundRobin(weights_wrr)
simulate(wrr_balancer, servers_heterogeneous, requests_list)


# --- KỊCH BẢN 2: TẮC NGHẼN ĐỘNG ---
print("\n==============================================")
print("KỊCH BẢN 2: Tắc nghẽn Động (Cả hai đều 'mù')")
print("==============================================")
servers_equal: Dict[str, ServerSpec] = {
    'S1': {'processing_time': 1},
    'S2': {'processing_time': 1},
    'S3': {'processing_time': 1}
}
requests_list_dynamic: List[int] = [1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1]

# --- Chạy WRR (với trọng số 1:1:1, tức là giống RR) ---
weights_equal: Dict[str, int] = {'S1': 1, 'S2': 1, 'S3': 1}
wrr_equal_balancer: BaseLoadBalancer = SmoothWeightedRoundRobin(weights_equal)
simulate(wrr_equal_balancer, servers_equal, requests_list_dynamic)

# --- KỊCH BẢN 3: KIỂM TRA LỖI "NONE" ---
print("\n==============================================")
print("KỊCH BẢN 3: Kiểm tra lỗi (Balancer rỗng)")
print("==============================================")
servers_empty_specs: Dict[str, ServerSpec] = {} # Không có thông số server
requests_empty_list: List[int] = [1] * 3

# Khởi tạo balancer với danh sách rỗng
empty_balancer: BaseLoadBalancer = SmoothWeightedRoundRobin({})
simulate(empty_balancer, servers_empty_specs, requests_empty_list)
