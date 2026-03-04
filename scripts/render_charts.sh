#!/bin/bash
# render_charts.sh — Render biểu đồ so sánh thuật toán
# 
# Sử dụng:
#   ./scripts/render_charts.sh                     # Tất cả scenario, dark theme
#   ./scripts/render_charts.sh --presentation      # 5 chart thuyết trình (tiếng Việt, nền trắng)
#   ./scripts/render_charts.sh --theme light       # 9 chart đầy đủ, nền trắng (in ấn)

set -e
cd "$(dirname "$0")/.."

SCENARIOS=("flash_crowd" "predictable_ramping" "targeted_congestion" "gradual_shift")
RESULTS_DIR="stats/results"

# Xác định folder output theo flag
TARGET_DIR="$RESULTS_DIR/charts"
for arg in "$@"; do
    if [ "$arg" = "--presentation" ]; then
        TARGET_DIR="$RESULTS_DIR/charts_presentation"
    fi
done
# Check --theme light
PREV=""
for arg in "$@"; do
    if [ "$PREV" = "--theme" ] && [ "$arg" = "light" ]; then
        TARGET_DIR="$RESULTS_DIR/charts_light"
    fi
    PREV="$arg"
done

# Xóa sạch folder cũ 1 LẦN DUY NHẤT
if [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"
    echo "  [CLEAN] Đã xóa: $TARGET_DIR"
fi
mkdir -p "$TARGET_DIR"

echo ""
echo "============================================================"
echo "  NCKH SDN — Chart Renderer"
echo "  Scenarios: ${SCENARIOS[*]}"
echo "  Output: $TARGET_DIR"
echo "  Args: $@"
echo "============================================================"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    echo ">>> Rendering: $scenario $@"
    python3 ai_model/generate_comparison_charts.py --scenario "$scenario" "$@"
    echo ""
done

echo "============================================================"
echo "  HOÀN TẤT! Tất cả biểu đồ tại: $TARGET_DIR/"
ls -1 "$TARGET_DIR"/*.png 2>/dev/null | wc -l | xargs -I{} echo "  Tổng: {} files"
echo "============================================================"
