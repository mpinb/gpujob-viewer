# SLURM GPU Monitoring Tool (gpujob-viewer)

SLURM GPU job performance monitoring tool.

This tool monitors GPU utilization for SLURM array jobs, identifying which specific GPU each job is using and collecting performance statistics.

## Features

- **Job-GPU Mapping**: Identifies which GPU each SLURM job is using by matching job processes with GPU processes
- **Real-time Monitoring**: Continuously monitors GPU utilization, memory usage, and performance metrics
- **Parallel Monitoring**: Can monitor multiple jobs simultaneously using threading
- **Statistical Analysis**: Generates summary statistics and efficiency reports
- **Visualization**: Creates plots showing GPU utilization over time
- **SLURM Integration**: Can be run as a SLURM job itself for scalability

## Requirements

```bash
# Python packages
pip install pandas matplotlib numpy

# System requirements
- SSH access to compute nodes
- nvidia-smi available on GPU nodes
- SLURM utilities (squeue, scontrol)
```

## Usage

### 1. Direct Python Script Usage

```bash
# Basic usage - monitor user '$user' jobs indefinitely
python3 slurm_gpu_monitor.py --username $USER

# Monitor for specific duration (300 seconds)
python3 slurm_gpu_monitor.py --username $USER --duration 300

# Specify monitoring interval and output directory
python3 slurm_gpu_monitor.py --username $USER --interval 10 --output-dir my_results

# Save raw data to CSV
python3 slurm_gpu_monitor.py --username $USER --csv-file raw_data.csv
```

### 2. SLURM Batch Job Usage

```bash
# Make the batch script executable
chmod +x slurm_gpu_monitor.sh

# Submit monitoring job
sbatch slurm_gpu_monitor.sh --username $USER

# Monitor for specific duration
sbatch slurm_gpu_monitor.sh --username $USER --duration 1800

# With custom settings
sbatch slurm_gpu_monitor.sh --username $USER --interval 10 --output-dir results_$(date +%Y%m%d)
```

### 3. Interactive Usage Example

```bash
# Start monitoring
python3 slurm_gpu_monitor.py --username $USER --interval 5

# The script will:
# 1. Find all running GPU jobs for the user
# 2. For each job, identify which GPU it's using
# 3. Monitor GPU utilization continuously
# 4. Save data and generate reports when stopped (Ctrl+C)
```

## Output Files

The tool generates several output files:

### Generated Files:
- `job_summary.csv`: Summary statistics for each monitored job
- `gpu_monitoring_plots.png`: Combined visualization of all jobs
- `job_<ID>_monitoring.png`: Individual plots for each job
- `report.txt`: Detailed text report with statistics
- `raw_data.csv`: Raw monitoring data (if --csv-file specified)

### Sample Output Structure:
```
gpu_monitoring_output/
├── job_summary.csv
├── gpu_monitoring_plots.png
├── job_245126_monitoring.png
├── job_245127_monitoring.png
├── report.txt
└── raw_data.csv (optional)
```

## How It Works

1. **Job Discovery**: Uses `squeue` to find running GPU jobs for the specified user
2. **GPU Identification**: 
   - Gets job processes using `ps` on each compute node
   - Gets GPU processes using `nvidia-smi --query-compute-apps`
   - Matches job PIDs with GPU processes to identify which GPU each job uses
3. **Monitoring**: Continuously queries GPU utilization using `nvidia-smi` for the identified GPU
4. **Data Collection**: Stores timestamped utilization data for each job
5. **Analysis**: Generates statistics and visualizations when monitoring completes

## Sample Job Summary Output

```csv
job_id,node,gpu_index,duration_minutes,avg_gpu_utilization,max_gpu_utilization,avg_memory_utilization,max_memory_utilization,avg_memory_used_gb,max_memory_used_gb
245126,somagpu084,1,45.2,78.5,95.0,65.3,82.1,12.4,15.8
245127,somagpu084,2,45.1,82.1,98.2,71.2,89.4,13.7,16.2
245128,somagpu084,3,45.0,15.2,45.8,12.4,28.7,2.8,6.4
```

## Key Features Explained

### GPU Process Identification
The script uses a sophisticated approach to identify which GPU each job is using:

```python
# 1. Get all processes for the job
job_processes = get_job_processes(node, job_id)

# 2. Get GPU processes and their GPU assignments
gpu_processes = get_gpu_processes(node)  # nvidia-smi --query-compute-apps

# 3. Find intersection - which job processes are using GPUs
for process in job_processes:
    if process.pid in gpu_processes:
        gpu_index = gpu_processes[process.pid]['gpu_index']
        # Monitor this specific GPU
```

### Parallel Monitoring
The tool can monitor multiple jobs simultaneously using threading:

```python
# Each job gets its own monitoring thread
with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
    for job in jobs:
        executor.submit(monitor_job, job)
```

### Statistical Analysis
Generates comprehensive statistics:
- Average and maximum GPU utilization
- Memory usage patterns
- Job efficiency metrics
- Duration and performance summaries

## Advanced Usage

### 1. Monitoring Specific Jobs
If you want to monitor only specific job IDs, you can modify the script:

```python
# Add job filtering in get_running_jobs()
target_jobs = ['245126', '245127', '245128']
jobs = [job for job in jobs if job['job_id'] in target_jobs]
```

### 2. Custom Monitoring Intervals
Different monitoring intervals for different scenarios:

```bash
# High-frequency monitoring for short jobs
python3 slurm_gpu_monitor.py --username $USER --interval 1

# Low-frequency monitoring for long jobs
python3 slurm_gpu_monitor.py --username $USER --interval 30
```

### 3. Integration with Job Arrays
For large job arrays, submit the monitoring as a separate job:

```bash
# Submit your job array
sbatch --array=1-100 your_gpu_job.sh

# Submit monitoring job
sbatch slurm_gpu_monitor.sh --username $USER --duration 3600
```

### 4. Real-time Monitoring Dashboard
For real-time monitoring, you can extend the script to output live data:

```python
# Add real-time output in monitor_job()
print(f"Job {job_id}: GPU {gpu_util}%, Memory {mem_util}%")
```

## Troubleshooting

### Common Issues

1. **SSH Connection Issues**
   ```bash
   # Test SSH access to compute nodes
   ssh somagpu084 'nvidia-smi'
   ```

2. **Permission Issues**
   ```bash
   # Ensure you can access job information
   scontrol show job <job_id>
   ```

3. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install pandas matplotlib numpy
   ```

4. **GPU Not Detected**
   - The job might not be using GPU yet
   - Check if the job is actually running GPU code
   - Verify nvidia-smi works on the compute node

### Debug Mode
Add debug output to troubleshoot issues:

```python
# In the script, add debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Monitoring Overhead**: Each SSH call has overhead; balance interval vs. accuracy
- **Network Load**: Multiple concurrent SSH connections to same node
- **Storage**: Raw data can grow large for long monitoring periods

## Integration with SLURM

The tool integrates well with SLURM workflows:

```bash
# Submit your GPU jobs
sbatch --array=1-50 gpu_job.sh

# Monitor them
sbatch --dependency=afterok:$SLURM_JOB_ID slurm_gpu_monitor.sh --username $USER

# Or run monitoring in parallel
sbatch slurm_gpu_monitor.sh --username $USER --duration 1800 &
```

## Example Workflow

1. **Submit your GPU job array**:
   ```bash
   sbatch --array=1-100 sleap_inference.sh
   ```

2. **Start monitoring**:
   ```bash
   python3 slurm_gpu_monitor.py --username $USER --interval 5
   ```

3. **Let it run** until jobs complete or stop with Ctrl+C

4. **Analyze results**:
   - Check `job_summary.csv` for efficiency metrics
   - Review plots in `gpu_monitoring_plots.png`
   - Read detailed report in `report.txt`

5. **Optimize based on findings**:
   - Jobs with low GPU utilization might need optimization
   - Memory usage patterns can inform resource requests
   - Duration data helps with time limit settings

This tool provides comprehensive insights into GPU utilization patterns, helping optimize resource usage and identify performance bottlenecks in SLURM GPU job arrays.