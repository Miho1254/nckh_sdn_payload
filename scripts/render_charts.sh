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

echo ""
echo "============================================================"
echo "  NCKH SDN — Chart Renderer"
echo "  Scenarios: ${SCENARIOS[*]}"
echo "  Args: $@"
echo "============================================================"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    echo ">>> Rendering: $scenario $@"
    python3 ai_model/generate_comparison_charts.py --scenario "$scenario" "$@"
    echo ""
done

echo "============================================================"
echo "  HOÀN TẤT! Tất cả biểu đồ đã được render."
echo "============================================================"
