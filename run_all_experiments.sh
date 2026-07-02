#!/bin/bash
# ================================================
# MenoSCA-FBTS One-Click Run All Experiments
# ================================================
# Execution Order:
# 1. SOTA Comparison Experiment
# 2. Ablation Study
# 3. Three-Group Experiment (Sleep-EDF)
# 4. Three-Group Experiment (ISRUC)
# 5. Three-Group Experiment (DREAMS)
# 6. Generate Representative Hypnograms
# ================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/experiment_results"
LOG_FILE="${LOG_DIR}/batch_run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

run_experiment() {
    local name="$1"
    local cmd="$2"

    log "================================================"
    log "[START] ${name}"
    log "Command: python ${cmd}"

    local start_time=$(date +%s)

    python ${cmd} 2>&1 | tee -a "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ ${exit_code} -eq 0 ]; then
        log "[DONE] ${name} (Exit Code: ${exit_code}, Duration: ${duration}s)"
    else
        log "[WARNING] ${name} (Exit Code: ${exit_code}, Duration: ${duration}s)"
    fi
    log ""
}

main() {
    log "================================================"
    log "MenoSCA-FBTS One-Click Run Script"
    log "================================================"
    log "Start Time: $(date)"
    log "Log File: ${LOG_FILE}"
    log "================================================"

    cd "${SCRIPT_DIR}"

    run_experiment "SOTA Comparison Experiment" "experiment_sota_comparison.py"
    run_experiment "Ablation Study" "experiment_ablation_study.py"
    run_experiment "Three-Group Experiment (Sleep-EDF)" "experiment_three_groups_paper.py"
    run_experiment "Three-Group Experiment (ISRUC)" "experiment_three_groups_paper.py --dataset isruc"
    run_experiment "Three-Group Experiment (DREAMS)" "experiment_three_groups_paper.py --dataset dreams"
    run_experiment "Generate Representative Hypnograms" "generate_representative_hypnograms.py"
    run_experiment "Comprehensive Analysis Study" "analysis_comprehensive.py"

    log "================================================"
    log "All Experiments Completed!"
    log "End Time: $(date)"
    log "Log File: ${LOG_FILE}"
    log "================================================"
}

main "$@"
