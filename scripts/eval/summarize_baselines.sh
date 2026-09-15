#!/usr/bin/env bash
# =============================================================================
# Summarize baseline evaluation results — comparison table.
#
# Usage:
#   bash scripts/eval/summarize_baselines.sh
# =============================================================================
set -euo pipefail

VOPD_ROOT="$(cd "$(dirname "$0")/../.." && pwd)/third_party/Vision-OPD/eval"

# Benchmark → JSON file mapping (same as prepare_data.py / run_eval.sh)
declare -A BENCH_JSON=(
    [vstar]=vstar.json
    [zoombench]=zoombench.json
    [hrbench-4k]=hr_bench_4k.json
    [hrbench-8k]=hr_bench_8k.json
    [mme-realworld]=MME_RealWorld.json
    [mme-realworld-cn]=MME_RealWorld_CN.json
    [mme-realworld-lite]=MME_RealWorld_Lite.json
    [mmstar]=mmstar.json
    [pope]=POPE.json
    [pope_adv]=POPE_adv.json
    [pope_pop]=POPE_pop.json
    [pope_random]=POPE_random.json
    [cv-bench]=cv_bench.json
    [mmvp]=mmvp.json
    [visualprobe]=visualprobe.json
)

BENCH_ALL=(
    vstar zoombench hrbench-4k hrbench-8k
    mme-realworld mme-realworld-cn mme-realworld-lite
    mmstar
    pope pope_adv pope_pop pope_random
    cv-bench mmvp visualprobe
)

echo ""
echo "=============================================================================="
echo "  Vision-OPD Baseline Accuracy Summary"
echo "=============================================================================="
printf "  %-28s  %14s  %14s  %8s\n" "Benchmark" "Vision-OPD-4B" "Qwen3.5-4B" "Δ"
echo "------------------------------------------------------------------------------"

for bench in "${BENCH_ALL[@]}"; do
    bench_json="${BENCH_JSON[$bench]}"
    val_vo="N/A"
    val_b="N/A"
    delta=""

    for model in "Vision-OPD-4B_seed42" "Qwen3.5-4B_seed42"; do
        judge_json="${VOPD_ROOT}/judge/${bench}/${model}_answer.jsonl"
        if [ ! -f "${judge_json}" ]; then
            continue
        fi

        raw=$(python3 "${VOPD_ROOT}/cal_acc.py" \
            --benchmark "${bench}" \
            --judge_json "${judge_json}" \
            --benchmark_json "${VOPD_ROOT}/${bench_json}" 2>/dev/null) || true

        # Parse accuracy from the last line (format varies by benchmark)
        acc_line=$(echo "${raw}" | tail -1)
        # Try "Acc: X/Y = Z%" or "accuracy=X%" or "average: Z%" or "cv-bench: Z%"
        acc=$(echo "${acc_line}" | grep -oP '=\s*\K[0-9]+\.[0-9]+%' | head -1 \
            || echo "${acc_line}" | grep -oP 'accuracy=\K[0-9]+\.[0-9]+%' | head -1 \
            || echo "${acc_line}" | grep -oP 'average:\s*\K[0-9]+\.[0-9]+%' | head -1 \
            || echo "${acc_line}" | grep -oP ':\s*\K[0-9]+\.[0-9]+%' | head -1)
        acc="${acc:-?}"

        if [ "${model}" = "Vision-OPD-4B_seed42" ]; then
            val_vo="${acc}"
        else
            val_b="${acc}"
        fi
    done

    # Compute delta if both are numeric
    vo_num=$(echo "${val_vo}" | tr -d '%')
    b_num=$(echo "${val_b}" | tr -d '%')
    if [[ "${vo_num}" =~ ^[0-9.]+$ ]] && [[ "${b_num}" =~ ^[0-9.]+$ ]]; then
        delta=$(python3 -c "print(f'{float(${vo_num}) - float(${b_num}):+.1f}pp')" 2>/dev/null || echo "?")
    else
        delta="?"
    fi

    printf "  %-28s  %14s  %14s  %8s\n" "${bench}" "${val_vo}" "${val_b}" "${delta}"
done

echo "------------------------------------------------------------------------------"
echo "  Δ = Vision-OPD-4B − Qwen3.5-4B  (positive = OPD improves over base)"
echo "=============================================================================="
