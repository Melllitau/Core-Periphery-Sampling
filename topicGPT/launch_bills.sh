set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$0")")"
mkdir -p topicGPT/data/output/bills logs

LOG=logs/bills.log
echo "[$(date)] Lancando pipeline Bills..." | tee -a "$LOG"

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "  Usando conda env: $CONDA_DEFAULT_ENV"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "  Usando venv: $VIRTUAL_ENV"
else
    echo "  Usando Python do sistema: $(which python3)"
fi

MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-16}

wait_for_memory() {
    local min_gb=$1
    if [[ "$(uname)" == "Darwin" ]]; then
        return  # macOS não tem /proc/meminfo
    fi
    local min_kb=$((min_gb * 1024 * 1024))
    echo "  Aguardando RAM livre (minimo ${min_gb}GB)..."
    while true; do
        avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        if [ "$avail_kb" -ge "$min_kb" ]; then
            echo "  RAM disponivel: $((avail_kb/1024/1024))GB — OK"
            return
        fi
        echo "  RAM disponivel: $((avail_kb/1024/1024))GB — aguardando..."
        sleep 5
    done
}

wait_for_gpu() {
    local gpu_id=${1:-0}
    if ! command -v nvidia-smi &>/dev/null; then
        return  # sem nvidia-smi (ex: Mac com MPS)
    fi
    echo "  Aguardando VRAM da GPU $gpu_id..."
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
        if [ -z "$used" ] || [ "$used" -lt 500 ]; then
            echo "  GPU $gpu_id: ${used:-0} MiB — OK"
            return
        fi
        echo "  GPU $gpu_id: ${used} MiB — aguardando..."
        sleep 5
    done
}

for PHASE in generation refinement assignment; do
    echo "============================================================"
    echo "[$(date)] FASE: $PHASE"
    echo "============================================================"

    while true; do
        python3 -u topicGPT/run_bills.py --phase "$PHASE" \
            >> "$LOG" 2>&1 &
        PID=$!
        echo "  PID=$PID"

        wait $PID
        EXIT=$?

        echo "[$(date)] Fase $PHASE (exit=$EXIT)"

        if [ $EXIT -eq 0 ]; then
            echo "[$(date)] Fase $PHASE concluida com sucesso!"
            break
        fi

        echo "[$(date)] ERRO na fase $PHASE — aguardando cleanup..."
        wait_for_gpu 0
        wait_for_memory $MIN_FREE_RAM_GB
        echo "[$(date)] Recursos liberados. Reiniciando fase $PHASE..."
    done

    echo ""
done

echo "============================================================"
echo "[$(date)] PIPELINE BILLS COMPLETO!"
echo "============================================================"
