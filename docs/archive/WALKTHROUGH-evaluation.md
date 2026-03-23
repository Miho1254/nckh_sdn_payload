# Walkthrough: Nghiệm thu Hệ thống Load Balancer SDN - AI (TFT-DQN)

Hệ thống đã được thiết lập thành công theo mô hình **Developer-Friendly**. Dưới đây là các bước để bro chứng kiến "Bộ não" AI thực hiện điều phối mạng trong thời gian thực.

## 1. Kiểm chứng AI Inference (Sức mạnh của trí tuệ nhân tạo)
Mở một Terminal mới (ngoài Mininet) và chui vào soi Log của Ryu Controller. Bro sẽ thấy AI đang dự đoán Pattern của traffic và chọn Backend tối ưu nhất.

```bash
docker exec -it nckh-sdn-mininet tail -f /tmp/ryu_ai.log
```
*Lưu ý: Nếu bro thấy log báo "Switching backend to 10.0.0.x", đó chính là lúc AI vừa "bẻ lái" luồng mạng để tránh nghẽn.*

## 2. Kiểm tra tải thực tế trên các Backend
Trong màn hình `mininet>`, bro có thể kiểm tra xem 3 con Backend (h5, h7, h8) có đang thực sự nhận tải không:

```bash
# Xem log của Backend trên h5
mininet> h5 tail -f /tmp/h5_api.log
```

## 3. Theo dõi bảng Flow Rules trên Switch
Để xem cách AI "nhồi" luật vào Switch để điều hướng traffic:

```bash
# Xem các dòng Flow trên Switch s7 (Edge Switch kết nối với User)
docker exec nckh-sdn-mininet ovs-ofctl -O OpenFlow13 dump-flows s7
```

## 4. Thu thập số liệu sau khi Test
Sau khi bro gõ `exit` trong Mininet để kết thúc bài test, hãy kiểm tra thư mục `stats/` trên máy Host:
- `stats/flow_stats.csv`: Chỉ số lưu lượng chi tiết.
- `stats/port_stats.csv`: Tình trạng các cổng switch.

**Chúc mừng bro đã hoàn thành triển khai hạ tầng nghiên cứu SDN siêu cấp!** 🚀🏆
