#!/bin/bash

# End Game Benchmark Script
# Chạy đánh giá Live Mininet trên các thuật toán cơ bản và các biến thể AI
# Yêu cầu: sudo

SCENARIOS=("golden_hour" "video_conference" "hardware_degradation" "low_rate_dos")

# Danh sách các model/checkpoint cần đánh giá
MODELS=(
    "RR RR none none"
    "WRR WRR none none"
    "CQL_BEST_SAMPLED AI /work/ai_model/checkpoints/tft_ac_best.pth sampled"
    "CQL_BEST_ARGMAX AI /work/ai_model/checkpoints/tft_ac_best.pth argmax"
)

# Thư mục lưu kết quả benchmark
RES_DIR="stats/benchmark_final"
mkdir -p "$RES_DIR"

echo "=========================================================="
echo "  BẮT ĐẦU END GAME LIVE BENCHMARK (4 SCENARIOS)"
echo "=========================================================="
echo ""

# --- Quét trước tiến độ ---
TOTAL_TASKS=$(( ${#MODELS[@]} * ${#SCENARIOS[@]} ))
COMPLETED=0
MISSING_LIST=""

for model in "${MODELS[@]}"; do
    eval arr=($model)
    NAME="${arr[0]}"
    for scene in "${SCENARIOS[@]}"; do
        OUT_FILE="$RES_DIR/${NAME}_${scene}_metrics.json"
        if [ -f "$OUT_FILE" ]; then
            COMPLETED=$((COMPLETED+1))
        else
            MISSING_LIST="$MISSING_LIST\n   - $NAME ($scene)"
        fi
    done
done

REMAINING=$((TOTAL_TASKS - COMPLETED))

echo ">> TÌNH TRẠNG HIỆN TẠI:"
if [ "$COMPLETED" -eq 0 ]; then
    echo "   - Chưa có bài test nào được hoàn thành (0 / $TOTAL_TASKS)."
else
    echo "   - Đã hoàn thành : $COMPLETED / $TOTAL_TASKS bài test"
    echo "   - Còn thiếu     : $REMAINING bài test"
    if [ "$REMAINING" -gt 0 ]; then
        echo -e "   - Danh sách các bài chưa chạy:$MISSING_LIST"
    else
        echo "   - Tuyệt vời! Bạn đã hoàn thành toàn bộ Benchmark!"
    fi
fi
echo ""

echo -ne "Bạn muốn chạy tiếp (Resume - R) hay bắt đầu trận mới (New - N)? [R/n]: "
read choice
if [[ "$choice" == "n" || "$choice" == "N" ]]; then
    echo ">> Xóa data cũ, bắt đầu lại từ đầu..."
    rm -f "$RES_DIR"/*
else
    echo ">> Giữ lại data cũ, chỉ chạy các kịch bản còn thiếu (Resume mode)..."
fi
echo ""

for model in "${MODELS[@]}"; do
    eval arr=($model)
    NAME="${arr[0]}"
    ALGO="${arr[1]}"
    CHK="${arr[2]}"
    RULE="${arr[3]}"
    
    echo ""
    echo ">> Đang test Model/Config: $NAME"
    echo "   ALGO=$ALGO"
    
    if [ "$CHK" != "none" ]; then
        echo "   AI_CHECKPOINT_PATH=$CHK"
        echo "   AI_SERVING_RULE=$RULE"
        # Export cho controller
        export AI_CHECKPOINT_PATH=$CHK
        export AI_SERVING_RULE=$RULE
    else
        unset AI_CHECKPOINT_PATH
        unset AI_SERVING_RULE
    fi
    
    # Evaluate 4 scenario bằng script có sẵn
    # Eval script đã dọn kết quả vào stats/results/EVAL_$scenario
    for scene in "${SCENARIOS[@]}"; do
        echo "   - Đánh giá kịch bản: $scene"
        
        OUT_FILE="$RES_DIR/${NAME}_${scene}_metrics.json"
        if [ -f "$OUT_FILE" ]; then
            echo "     [ĐÃ CHẠY] Kịch bản này đã có kết quả! Bỏ qua (Resume mode)."
            continue
        fi
        
        export SCENARIO=$scene
        export EVAL_MODE=1
        export LB_ALGO=$ALGO
        
        # Mượn script đánh giá cũ (Truyền đủ ALGO và SCENARIO để không bị kẹt ở lệnh read)
        ./scripts/evaluate_sdn.sh "$ALGO" "$scene" > /dev/null
        
        # Cleanup mininet
        sudo mn -c > /dev/null 2>&1
        
        # Copy kết quả sang thư mục phân biệt dể dễ so sánh
        cp stats/results/${ALGO}_${scene}/evaluation_metrics.json "$OUT_FILE"
        
        echo "     [Xong $scene]"
    done
done

echo ""
echo "=========================================================="
echo "  BENCHMARK HOÀN TẤT. Xem kết quả tại: $RES_DIR"
echo "=========================================================="
