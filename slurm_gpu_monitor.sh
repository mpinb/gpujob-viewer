#!/bin/bash
#SBATCH --job-name=gpu_monitor
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=gpu_monitor_%j.out
#SBATCH --error=gpu_monitor_%j.err

# Load required modules (adjust based on your system)
# module load python/3.8
# module load matplotlib
# module load pandas

# Set default values
USERNAME=""
PARTITION="GPU"
INTERVAL=5
DURATION=""
OUTPUT_DIR="gpu_monitoring_$(date +%Y%m%d_%H%M%S)"
CSV_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            USERNAME="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--csv-file)
            CSV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -u, --username     Username to monitor (required)"
            echo "  -p, --partition    Partition to monitor (default: GPU)"
            echo "  -i, --interval     Monitoring interval in seconds (default: 5)"
            echo "  -d, --duration     Monitoring duration in seconds (optional)"
            echo "  -o, --output-dir   Output directory for results"
            echo "  -c, --csv-file     CSV file to save raw data"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if username is provided
if [[ -z "$USERNAME" ]]; then
    echo "Error: Username is required. Use -u or --username option."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set up Python environment (adjust path as needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Build the python command
PYTHON_CMD="python3 slurm_gpu_monitor.py --username $USERNAME --partition $PARTITION --interval $INTERVAL --output-dir $OUTPUT_DIR"

if [[ -n "$DURATION" ]]; then
    PYTHON_CMD="$PYTHON_CMD --duration $DURATION"
fi

if [[ -n "$CSV_FILE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --csv-file $CSV_FILE"
fi

echo "Starting GPU monitoring for user: $USERNAME"
echo "Partition: $PARTITION"
echo "Interval: $INTERVAL seconds"
echo "Output directory: $OUTPUT_DIR"
echo "Command: $PYTHON_CMD"
echo "----------------------------------------"

# Run the monitoring script
$PYTHON_CMD

echo "----------------------------------------"
echo "GPU monitoring completed."
echo "Results saved to: $OUTPUT_DIR"

# Optional: compress results
if command -v tar &> /dev/null; then
    tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"
    echo "Results compressed to: ${OUTPUT_DIR}.tar.gz"
fi