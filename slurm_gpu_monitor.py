
#!/usr/bin/env python3

import subprocess
import time
import argparse
import os
import signal
import sys
from datetime import datetime
from collections import defaultdict
import queue
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Adding a logging library for better debugging and error handling
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Importing necessary libraries for SLURM job monitoring
from slurm_job_process_mapper import SlurmJobProcessMapper

# Importing plotting libraries for real-time visualization
# Using matplotlib for static plots and bokeh for interactive plots

# Importing bokeh and set flag if it is available
try:
    from bokeh.plotting import figure
    from bokeh.layouts import column, row
    from bokeh.models import HoverTool, ColumnDataSource, Div
    from bokeh.palettes import Category10
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.io import output_file, save
    BOKEH_AVAILABLE = True
except ImportError:
    logging.warning("Bokeh is not available for live GPU monitoring.")
    BOKEH_AVAILABLE = False

# Importing matplotlib and set flag if it is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib is not available for plotting.")
    MATPLOTLIB_AVAILABLE = False

# Define the main class for monitoring SLURM GPU jobs
class SlurmGPUMonitor:
    def __init__(self, username, partition="GPU", monitoring_interval=5):
        self.username = username
        self.partition = partition
        self.monitoring_interval = monitoring_interval
        self.job_data = defaultdict(list)
        self.stop_monitoring = False
        self.live_data = defaultdict(list)  # For real-time Bokeh updates
        self.bokeh_sources = {}  # Store ColumnDataSource objects
        self.plotting_backend = 'bokeh'  # Default plotting backend
        self.data_lock = threading.Lock()  # Lock for thread-safe data access
        self.shutdown_event = threading.Event()  # Event for coordinated shutdown
        self.bokeh_server = None
        self.executor = None
        # the slurm job process mapper instance
        self.job_process_mapper = SlurmJobProcessMapper()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # logging setup
        logging.info(f"Initialized SlurmGPUMonitor for user: {self.username}, partition: {self.partition}, monitoring interval: {self.monitoring_interval} seconds")
    
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.stop_monitoring = True
        self.shutdown_event.set()
        
        # Stop Bokeh server if running
        if self.bokeh_server:
            logging.info("Stopping Bokeh server...")
            try:
                self.bokeh_server.stop()
            except Exception as e:
                logging.error(f"Error stopping Bokeh server: {e}")
        
        # If we're in the middle of monitoring, let it clean up
        if hasattr(self, '_monitoring_in_progress') and self._monitoring_in_progress:
            logging.info("Waiting for monitoring threads to complete...")
            return
        
        # Otherwise exit immediately
        logging.info("Shutdown complete.")
        sys.exit(0)
    
    
    def get_running_jobs(self):
        """Get running GPU jobs for the specified user"""
        try:
            cmd = f"squeue -u {self.username} -o '%A %T %N %P %j' --noheader"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            logging.info(f"Running command to get running jobs: {cmd}")
            logging.info(f"Command output: {result.stdout.strip()}")
            
            if result.returncode != 0:
                logging.error(f"Error getting job info: {result.stderr}")
                return []
                
            jobs = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        job_id, status, node, partition_found = parts[:4]
                        job_name = ' '.join(parts[4:]) if len(parts) > 4 else 'Unknown'
                        
                        # Debug output
                        logging.debug(f"Processing job {job_id}, status: '{status}', node: '{node}', partition: '{partition_found}'")
                        
                        # Check for both 'R' and 'RUNNING' status
                        if (status in ['R', 'RUNNING']) and partition_found == self.partition:
                            jobs.append({
                                'job_id': job_id,
                                'status': status,
                                'node': node,
                                'partition': partition_found,
                                'job_name': job_name
                            })
                            logging.debug(f"Added job {job_id} to monitoring list")
                        else:
                            logging.debug(f"Skipped job {job_id} - status: '{status}', partition: '{partition_found}' (looking for '{self.partition}')")
                            
            logging.debug(f"Total jobs found for monitoring: {len(jobs)}")
            return jobs
            
        except Exception as e:
            logging.error(f"Error getting running jobs: {e}")
            return []
    
    
    # def get_job_processes(self, node, job_id):
    #     """Get processes running on a node for a specific job"""
    #     try:
    #         # SSH options to handle host key verification
    #         ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
            
    #         # Get processes for the job
    #         cmd = f"ssh {ssh_opts} {node} 'ps -eo pid,ppid,cmd --no-headers | grep -v grep'"
    #         result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
    #         logging.info(f"Running command to get job processes on {node}: {cmd}")
    #         logging.info(f"Command output: {result.stdout.strip()}")
            
    #         if result.returncode != 0:
    #             return []
                
    #         # Also get SLURM job info to identify job processes
    #         slurm_cmd = f"ssh {ssh_opts} {node} 'scontrol show job {job_id}'"
    #         slurm_result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)
            
    #         logging.info(f"Running command to get SLURM job info on {node}: {slurm_cmd}")
    #         logging.info(f"Command output: {slurm_result.stdout.strip()}")
            
    #         return result.stdout.strip().split('\n')
    #     except Exception as e:
    #         logging.error(f"Error getting job processes for {node}: {e}")
    #         return []
    
    
    def get_gpu_processes(self, node):
        """Get GPU processes and their GPU assignments"""
        try:
            # SSH options to handle host key verification
            ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
            
            cmd = f"ssh {ssh_opts} {node} 'nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            
            logging.info(f"Running command to get GPU processes on {node}: {cmd}")
            logging.info(f"Command output: {result.stdout.strip()}")
            
            if result.returncode != 0:
                return {}
                
            gpu_processes = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, gpu_uuid, used_mem = parts[:3]
                        gpu_processes[pid.strip()] = {
                            'gpu_uuid': gpu_uuid.strip(),
                            'used_memory': used_mem.strip()
                        }
            return gpu_processes
        except Exception as e:
            logging.error(f"Error getting GPU processes for {node}: {e}")
            return {}
    
    
    def get_gpu_info(self, node):
        """Get GPU information and mapping"""
        try:
            # SSH options to handle host key verification
            ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

            cmd = f"ssh {ssh_opts} {node} 'nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            
            logging.info(f"Running command to get GPU info on {node}: {cmd}")
            logging.info(f"Command output: {result.stdout.strip()}")
            
            if result.returncode != 0:
                return {}
                
            gpu_info = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        index, uuid, name = parts[:3]
                        gpu_info[uuid.strip()] = {
                            'index': index.strip(),
                            'name': name.strip()
                        }
            return gpu_info
        except Exception as e:
            logging.error(f"Error getting GPU info for {node}: {e}")
            return {}
    
    
    def get_gpu_utilization(self, node, gpu_index=None):
        """Get GPU utilization for specific GPU or all GPUs"""
        try:
            # SSH options to handle host key verification
            ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
            
            if gpu_index is not None:
                cmd = f"ssh {ssh_opts} {node} 'nvidia-smi --id={gpu_index} --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader'"
            else:
                cmd = f"ssh {ssh_opts} {node} 'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader'"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            logging.info(f"Running command to get GPU utilization on {node}: {cmd}")
            logging.info(f"Command output: {result.stdout.strip()}")
            
            
            if result.returncode != 0:
                return []
                
            utilization_data = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        timestamp, name, gpu_util, mem_util, mem_total, mem_used = parts[:6]
                        utilization_data.append({
                            'timestamp': timestamp.strip(),
                            'name': name.strip(),
                            'gpu_utilization': gpu_util.strip().replace(' %', ''),
                            'memory_utilization': mem_util.strip().replace(' %', ''),
                            'memory_total': mem_total.strip().replace(' MiB', ''),
                            'memory_used': mem_used.strip().replace(' MiB', '')
                        })
            return utilization_data
        except Exception as e:
            logging.error(f"Error getting GPU utilization for {node}: {e}")
            return []
    
    
    def identify_job_gpu(self, node, job_id):
        """Identify which GPU is being used by a specific job"""
        try:
            # Get job processes
            # job_processes = self.get_job_processes(node, job_id)
            job_processes = self.job_process_mapper.get_job_processes(node, job_id, user_id=self.username)
            
            # Get GPU processes
            gpu_processes = self.get_gpu_processes(node)
            
            # Get GPU info
            gpu_info = self.get_gpu_info(node)
            
            # Find intersection - processes that are both job processes and GPU processes
            job_gpu_mapping = {}
            
            for process_line in job_processes:
                if process_line.strip():
                    pid = process_line.split()[0]
                    if pid in gpu_processes:
                        gpu_uuid = gpu_processes[pid]['gpu_uuid']
                        if gpu_uuid in gpu_info:
                            gpu_index = gpu_info[gpu_uuid]['index']
                            job_gpu_mapping[job_id] = {
                                'gpu_index': gpu_index,
                                'gpu_uuid': gpu_uuid,
                                'gpu_name': gpu_info[gpu_uuid]['name'],
                                'pid': pid,
                                'used_memory': gpu_processes[pid]['used_memory']
                            }
                            logging.info(f"Identified job {job_id} on node {node} using GPU {gpu_index} (UUID: {gpu_uuid}, Name: {gpu_info[gpu_uuid]['name']})")
                            break
            
            return job_gpu_mapping
        except Exception as e:
            logging.warning(f"Error identifying job GPU for {node}, job {job_id}: {e}")
            return {}
    
    
    def update_live_data(self, job_id, data_point):
        """Update live data for real-time Bokeh plotting"""
        if self.plotting_backend == 'bokeh':
            with self.data_lock:
                self.live_data[job_id].append(data_point)
                
                # Update Bokeh data sources if they exist
                if job_id in self.bokeh_sources:
                    try:
                        job_data = self.live_data[job_id]
                        df = pd.DataFrame(job_data)
                        
                        # Update GPU utilization source
                        if 'gpu_util_source' in self.bokeh_sources[job_id] and not df.empty:
                            new_data = {
                                'x': df['monitoring_time'].tolist(),
                                'y': pd.to_numeric(df['gpu_utilization'], errors='coerce').fillna(0).tolist(),
                                'job_id': [job_id] * len(df)
                            }
                            self.bokeh_sources[job_id]['gpu_util_source'].data = new_data
                        
                        # Update memory utilization source
                        if 'mem_util_source' in self.bokeh_sources[job_id] and not df.empty:
                            new_data = {
                                'x': df['monitoring_time'].tolist(),
                                'y': pd.to_numeric(df['memory_utilization'], errors='coerce').fillna(0).tolist(),
                                'job_id': [job_id] * len(df)
                            }
                            self.bokeh_sources[job_id]['mem_util_source'].data = new_data
                    except Exception as e:
                        logging.warning(f"Error updating live data for job {job_id}: {e}")


    def monitor_job(self, job_info, results_queue):
        """Monitor a single job's GPU utilization with real-time updates"""
        job_id = job_info['job_id']
        node = job_info['node']
        
        logging.info(f"Starting monitoring for job {job_id} on node {node}")
        
        # Identify which GPU this job is using
        job_gpu_mapping = self.identify_job_gpu(node, job_id)
        
        if not job_gpu_mapping:
            logging.info(f"Could not identify GPU for job {job_id} on node {node}")
            # Monitor all GPUs as fallback
            gpu_index = None
        else:
            gpu_index = job_gpu_mapping[job_id]['gpu_index']
            logging.info(f"Job {job_id} is using GPU {gpu_index} on node {node}")
        
        job_data = []
        start_time = time.time()
        
        while not self.stop_monitoring and not self.shutdown_event.is_set():
            try:
                # Get GPU utilization
                utilization_data = self.get_gpu_utilization(node, gpu_index)
                
                for gpu_data in utilization_data:
                    data_point = {
                        'job_id': job_id,
                        'node': node,
                        'gpu_index': gpu_index,
                        'timestamp': datetime.now().isoformat(),
                        'monitoring_time': time.time() - start_time,
                        **gpu_data
                    }
                    job_data.append(data_point)
                    
                    # Update live data for real-time Bokeh plotting
                    self.update_live_data(job_id, data_point)
                
                # Check for shutdown with timeout
                if self.shutdown_event.wait(timeout=self.monitoring_interval):
                    break
                
            except Exception as e:
                logging.error(f"Error monitoring job {job_id}: {e}")
                if self.shutdown_event.wait(timeout=self.monitoring_interval):
                    break
        
        results_queue.put({job_id: job_data})
        logging.info(f"Finished monitoring job {job_id}")


    def create_bokeh_app(self, doc):
        """Create Bokeh application for real-time monitoring"""
        # Title
        title = Div(text=f"<h1>Real-time GPU Monitoring for {self.username}</h1>")
        
        # Get initial job list
        jobs = self.get_running_jobs()
        
        if not jobs:
            doc.add_root(column(title, Div(text="No running jobs found")))
            return
        
        plots = []
        
        for job in jobs:
            job_id = job['job_id']
            
            # Initialize data sources
            self.bokeh_sources[job_id] = {
                'gpu_util_source': ColumnDataSource(data={'x': [], 'y': [], 'job_id': []}),
                'mem_util_source': ColumnDataSource(data={'x': [], 'y': [], 'job_id': []})
            }
            
            # GPU Utilization plot
            p1 = figure(title=f"Job {job_id} - GPU Utilization", 
                       x_axis_label="Time (seconds)", 
                       y_axis_label="GPU Utilization (%)",
                       width=600, height=300)
            
            p1.line('x', 'y', source=self.bokeh_sources[job_id]['gpu_util_source'], 
                   line_width=2, color="blue")
            p1.add_tools(HoverTool(tooltips=[("Job ID", "@job_id"), ("Time", "@x s"), ("GPU Util", "@y%")]))
            
            # Memory Utilization plot
            p2 = figure(title=f"Job {job_id} - Memory Utilization", 
                       x_axis_label="Time (seconds)", 
                       y_axis_label="Memory Utilization (%)",
                       width=600, height=300)
            
            p2.line('x', 'y', source=self.bokeh_sources[job_id]['mem_util_source'], 
                   line_width=2, color="red")
            p2.add_tools(HoverTool(tooltips=[("Job ID", "@job_id"), ("Time", "@x s"), ("Memory Util", "@y%")]))
            
            # Add plots to layout
            plots.append(row(p1, p2))
        
        # Create layout
        layout = column(title, *plots)
        doc.add_root(layout)
        
        # Add periodic callback to refresh job list
        def update_job_list():
            if not self.shutdown_event.is_set():
                current_jobs = self.get_running_jobs()
                if len(current_jobs) != len(jobs):
                    # Job list changed, need to recreate layout
                    doc.clear()
                    self.create_bokeh_app(doc)
        
        doc.add_periodic_callback(update_job_list, 30000)  # Check every 30 seconds
    

    def monitor_all_jobs(self, duration=None):
        """Monitor all running jobs with real-time updates"""
        self._monitoring_in_progress = True
        
        try:
            jobs = self.get_running_jobs()
            
            if not jobs:
                logging.info(f"No running GPU jobs found for user {self.username}")
                return {}
            
            logging.info(f"Found {len(jobs)} running GPU jobs for user {self.username}")

            # Initialize results queue for collecting job data
            results_queue = queue.Queue()
            
            # Start monitoring threads
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                self.executor = executor
                futures = []
                
                for job in jobs:
                    future = executor.submit(self.monitor_job, job, results_queue)
                    futures.append(future)
                
                # Monitor for specified duration or until interrupted
                start_time = time.time()
                
                try:
                    if duration:
                        # Wait for duration or shutdown event
                        if self.shutdown_event.wait(timeout=duration):
                            logging.info("Shutdown requested during timed monitoring")
                    else:
                        logging.info("Monitoring jobs... Press Ctrl+C to stop")
                        # Keep monitoring until interrupted or no jobs remain
                        while not self.shutdown_event.is_set():
                            time.sleep(1)
                            # Check if any jobs are still running
                            current_jobs = self.get_running_jobs()
                            if not current_jobs:
                                logging.info("All jobs completed")
                                break
                            
                except KeyboardInterrupt:
                    logging.info("\nKeyboardInterrupt received in monitor_all_jobs")
                    self.stop_monitoring = True
                    self.shutdown_event.set()
                
                # Stop monitoring
                logging.info("Stopping job monitoring...")
                self.stop_monitoring = True
                self.shutdown_event.set()
                
                # Wait for all threads to complete with timeout
                logging.info("Waiting for monitoring threads to complete...")
                completed_count = 0
                for future in as_completed(futures, timeout=10):
                    try:
                        future.result()
                        completed_count += 1
                    except Exception as e:
                        logging.error(f"Error in monitoring thread: {e}")
                        completed_count += 1
                
                logging.info(f"Completed {completed_count}/{len(futures)} monitoring threads")
            
            # Collect results
            all_job_data = {}
            while not results_queue.empty():
                job_data = results_queue.get()
                all_job_data.update(job_data)
            
            return all_job_data
            
        finally:
            self._monitoring_in_progress = False

    
    def save_data(self, job_data, filename):
        """Save monitoring data to CSV"""
        if not job_data:
            logging.info("No data to save")
            return
        
        # Flatten data for CSV
        all_data = []
        for job_id, data_points in job_data.items():
            all_data.extend(data_points)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(filename, index=False)
            logging.info(f"Data saved to {filename}")
    
    def generate_report(self, job_data, output_dir="gpu_monitoring_report"):
        """Generate analysis report and plots"""
        if not job_data:
            logging.info("No data to analyze")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        all_data = []
        for job_id, data_points in job_data.items():
            all_data.extend(data_points)
        
        if not all_data:
            logging.warning("No data points to analyze")
            return
            
        df = pd.DataFrame(all_data)
        
        # Convert numeric columns
        numeric_columns = ['gpu_utilization', 'memory_utilization', 'memory_total', 'memory_used', 'monitoring_time']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Generate summary statistics
        summary_stats = {}
        for job_id in df['job_id'].unique():
            job_df = df[df['job_id'] == job_id]
            stats = {
                'job_id': job_id,
                'node': job_df['node'].iloc[0] if not job_df.empty else 'Unknown',
                'gpu_index': job_df['gpu_index'].iloc[0] if not job_df.empty else 'Unknown',
                'duration_minutes': job_df['monitoring_time'].max() / 60 if 'monitoring_time' in job_df.columns else 0,
                'avg_gpu_utilization': job_df['gpu_utilization'].mean() if 'gpu_utilization' in job_df.columns else 0,
                'max_gpu_utilization': job_df['gpu_utilization'].max() if 'gpu_utilization' in job_df.columns else 0,
                'avg_memory_utilization': job_df['memory_utilization'].mean() if 'memory_utilization' in job_df.columns else 0,
                'max_memory_utilization': job_df['memory_utilization'].max() if 'memory_utilization' in job_df.columns else 0,
                'avg_memory_used_gb': job_df['memory_used'].mean() / 1024 if 'memory_used' in job_df.columns else 0,
                'max_memory_used_gb': job_df['memory_used'].max() / 1024 if 'memory_used' in job_df.columns else 0,
            }
            summary_stats[job_id] = stats
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats.values())
        summary_df.to_csv(f"{output_dir}/job_summary.csv", index=False)
        
        # Generate plots
        self.create_plots(df, output_dir)
        
        # Generate text report
        with open(f"{output_dir}/report.txt", 'w') as f:
            f.write(f"GPU Monitoring Report for user: {self.username}\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n\n")
            f.write(f"Total jobs monitored: {len(summary_stats)}\n\n")
            
            for job_id, stats in summary_stats.items():
                f.write(f"Job ID: {job_id}\n")
                f.write(f"  Node: {stats['node']}\n")
                f.write(f"  GPU Index: {stats['gpu_index']}\n")
                f.write(f"  Duration: {stats['duration_minutes']:.2f} minutes\n")
                f.write(f"  Average GPU Utilization: {stats['avg_gpu_utilization']:.2f}%\n")
                f.write(f"  Maximum GPU Utilization: {stats['max_gpu_utilization']:.2f}%\n")
                f.write(f"  Average Memory Utilization: {stats['avg_memory_utilization']:.2f}%\n")
                f.write(f"  Maximum Memory Utilization: {stats['max_memory_utilization']:.2f}%\n")
                f.write(f"  Average Memory Used: {stats['avg_memory_used_gb']:.2f} GB\n")
                f.write(f"  Maximum Memory Used: {stats['max_memory_used_gb']:.2f} GB\n\n")
        
        logging.info(f"Report generated in {output_dir}/")

    def create_plots(self, df, output_dir):
        """Create visualization plots using selected backend"""
        if self.plotting_backend == 'matplotlib':
            self.create_matplotlib_plots(df, output_dir)
        elif self.plotting_backend == 'bokeh':
            self.create_bokeh_plots(df, output_dir)
        else:
            logging.warning("No plotting backend available")
    
    def create_matplotlib_plots(self, df, output_dir):
        """Create visualization plots using matplotlib"""
        # GPU Utilization over time for each job
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'GPU Monitoring Results for {self.username}', fontsize=16)
        
        # Plot 1: GPU Utilization over time
        ax1 = axes[0, 0]
        for job_id in df['job_id'].unique():
            job_df = df[df['job_id'] == job_id]
            if 'monitoring_time' in job_df.columns and 'gpu_utilization' in job_df.columns:
                ax1.plot(job_df['monitoring_time'] / 60, job_df['gpu_utilization'], 
                        label=f'Job {job_id}', alpha=0.7)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('GPU Utilization Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory Utilization over time
        ax2 = axes[0, 1]
        for job_id in df['job_id'].unique():
            job_df = df[df['job_id'] == job_id]
            if 'monitoring_time' in job_df.columns and 'memory_utilization' in job_df.columns:
                ax2.plot(job_df['monitoring_time'] / 60, job_df['memory_utilization'], 
                        label=f'Job {job_id}', alpha=0.7)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Utilization (%)')
        ax2.set_title('Memory Utilization Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average GPU Utilization by job
        ax3 = axes[1, 0]
        job_avg_gpu = df.groupby('job_id')['gpu_utilization'].mean()
        ax3.bar(range(len(job_avg_gpu)), job_avg_gpu.values)
        ax3.set_xlabel('Job ID')
        ax3.set_ylabel('Average GPU Utilization (%)')
        ax3.set_title('Average GPU Utilization by Job')
        ax3.set_xticks(range(len(job_avg_gpu)))
        ax3.set_xticklabels(job_avg_gpu.index, rotation=45)
        
        # Plot 4: Average Memory Usage by job
        ax4 = axes[1, 1]
        job_avg_mem = df.groupby('job_id')['memory_used'].mean() / 1024  # Convert to GB
        ax4.bar(range(len(job_avg_mem)), job_avg_mem.values)
        ax4.set_xlabel('Job ID')
        ax4.set_ylabel('Average Memory Usage (GB)')
        ax4.set_title('Average Memory Usage by Job')
        ax4.set_xticks(range(len(job_avg_mem)))
        ax4.set_xticklabels(job_avg_mem.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gpu_monitoring_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual job plots
        for job_id in df['job_id'].unique():
            job_df = df[df['job_id'] == job_id]
            if len(job_df) > 1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle(f'Job {job_id} GPU Monitoring', fontsize=14)
                
                # GPU Utilization
                if 'monitoring_time' in job_df.columns and 'gpu_utilization' in job_df.columns:
                    ax1.plot(job_df['monitoring_time'] / 60, job_df['gpu_utilization'], 'b-', alpha=0.7)
                    ax1.set_ylabel('GPU Utilization (%)')
                    ax1.set_title('GPU Utilization')
                    ax1.grid(True, alpha=0.3)
                
                # Memory Usage
                if 'monitoring_time' in job_df.columns and 'memory_used' in job_df.columns:
                    ax2.plot(job_df['monitoring_time'] / 60, job_df['memory_used'] / 1024, 'r-', alpha=0.7)
                    ax2.set_xlabel('Time (minutes)')
                    ax2.set_ylabel('Memory Usage (GB)')
                    ax2.set_title('Memory Usage')
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/job_{job_id}_monitoring.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def create_bokeh_plots(self, df, output_dir):
        """Create visualization plots using bokeh"""
        # Set up color palette
        colors = Category10[max(3, min(10, len(df['job_id'].unique())))]
        
        # Create main dashboard
        output_file(f"{output_dir}/gpu_monitoring_dashboard.html")
        
        # Plot 1: GPU Utilization over time
        p1 = figure(title="GPU Utilization Over Time", 
                   x_axis_label="Time (minutes)", 
                   y_axis_label="GPU Utilization (%)",
                   width=600, height=400)
        
        for i, job_id in enumerate(df['job_id'].unique()):
            job_df = df[df['job_id'] == job_id]
            if 'monitoring_time' in job_df.columns and 'gpu_utilization' in job_df.columns:
                source = ColumnDataSource(data=dict(
                    x=job_df['monitoring_time'] / 60,
                    y=job_df['gpu_utilization'],
                    job_id=[job_id] * len(job_df)
                ))
                p1.line('x', 'y', source=source, legend_label=f'Job {job_id}', 
                       line_width=2, color=colors[i % len(colors)])
        
        p1.add_tools(HoverTool(tooltips=[("Job ID", "@job_id"), ("Time", "@x min"), ("GPU Util", "@y%")]))
        p1.legend.location = "top_left"
        
        # Plot 2: Memory Utilization over time
        p2 = figure(title="Memory Utilization Over Time", 
                   x_axis_label="Time (minutes)", 
                   y_axis_label="Memory Utilization (%)",
                   width=600, height=400)
        
        for i, job_id in enumerate(df['job_id'].unique()):
            job_df = df[df['job_id'] == job_id]
            if 'monitoring_time' in job_df.columns and 'memory_utilization' in job_df.columns:
                source = ColumnDataSource(data=dict(
                    x=job_df['monitoring_time'] / 60,
                    y=job_df['memory_utilization'],
                    job_id=[job_id] * len(job_df)
                ))
                p2.line('x', 'y', source=source, legend_label=f'Job {job_id}', 
                       line_width=2, color=colors[i % len(colors)])
        
        p2.add_tools(HoverTool(tooltips=[("Job ID", "@job_id"), ("Time", "@x min"), ("Memory Util", "@y%")]))
        p2.legend.location = "top_left"
        
        # Plot 3: Average GPU Utilization by job
        job_avg_gpu = df.groupby('job_id')['gpu_utilization'].mean()
        p3 = figure(title="Average GPU Utilization by Job", 
                   x_axis_label="Job ID", 
                   y_axis_label="Average GPU Utilization (%)",
                   x_range=job_avg_gpu.index.astype(str),
                   width=600, height=400)
        
        source = ColumnDataSource(data=dict(
            x=job_avg_gpu.index.astype(str),
            y=job_avg_gpu.values,
            job_id=job_avg_gpu.index.astype(str)
        ))
        p3.vbar(x='x', top='y', source=source, width=0.8, color="navy", alpha=0.7)
        p3.add_tools(HoverTool(tooltips=[("Job ID", "@job_id"), ("Avg GPU Util", "@y%")]))
        
        # Plot 4: Average Memory Usage by job
        job_avg_mem = df.groupby('job_id')['memory_used'].mean() / 1024  # Convert to GB
        p4 = figure(title="Average Memory Usage by Job", 
                   x_axis_label="Job ID", 
                   y_axis_label="Average Memory Usage (GB)",
                   x_range=job_avg_mem.index.astype(str),
                   width=600, height=400)
        
        source = ColumnDataSource(data=dict(
            x=job_avg_mem.index.astype(str),
            y=job_avg_mem.values,
            job_id=job_avg_mem.index.astype(str)
        ))
        p4.vbar(x='x', top='y', source=source, width=0.8, color="red", alpha=0.7)
        p4.add_tools(HoverTool(tooltips=[("Job ID", "@job_id"), ("Avg Memory", "@y GB")]))
        
        # Create layout
        layout = column(row(p1, p2), row(p3, p4))
        save(layout)
        
        # Create individual job plots
        for job_id in df['job_id'].unique():
            job_df = df[df['job_id'] == job_id]
            if len(job_df) > 1:
                output_file(f"{output_dir}/job_{job_id}_monitoring.html")
                
                # GPU Utilization
                p1 = figure(title=f"Job {job_id} GPU Utilization", 
                           x_axis_label="Time (minutes)", 
                           y_axis_label="GPU Utilization (%)",
                           width=800, height=300)
                
                if 'monitoring_time' in job_df.columns and 'gpu_utilization' in job_df.columns:
                    source = ColumnDataSource(data=dict(
                        x=job_df['monitoring_time'] / 60,
                        y=job_df['gpu_utilization'],
                        timestamp=job_df['timestamp']
                    ))
                    p1.line('x', 'y', source=source, line_width=2, color="blue")
                    p1.add_tools(HoverTool(tooltips=[("Time", "@timestamp"), ("GPU Util", "@y%")]))
                
                # Memory Usage
                p2 = figure(title=f"Job {job_id} Memory Usage", 
                           x_axis_label="Time (minutes)", 
                           y_axis_label="Memory Usage (GB)",
                           width=800, height=300)
                
                if 'monitoring_time' in job_df.columns and 'memory_used' in job_df.columns:
                    source = ColumnDataSource(data=dict(
                        x=job_df['monitoring_time'] / 60,
                        y=job_df['memory_used'] / 1024,
                        timestamp=job_df['timestamp']
                    ))
                    p2.line('x', 'y', source=source, line_width=2, color="red")
                    p2.add_tools(HoverTool(tooltips=[("Time", "@timestamp"), ("Memory", "@y GB")]))
                
                # Save individual job layout
                layout = column(p1, p2)
                save(layout)
        
        logging.info(f"Bokeh plots saved to {output_dir}/")


    def start_bokeh_server(self, port=5006, output_dir="gpu_monitoring_output"):
        """Start Bokeh server for real-time monitoring"""
        if not BOKEH_AVAILABLE:
            logging.warning("Bokeh not available. Cannot start real-time server.")
            return None
        
        # Create Bokeh application
        app = Application(FunctionHandler(self.create_bokeh_app))
        
        # Start server
        server = Server({'/': app}, port=port, allow_websocket_origin=[f"localhost:{port}"], output_dir=output_dir)
        server.start()
        
        logging.info(f"Bokeh server started on http://localhost:{port}")
        logging.info("Real-time monitoring dashboard is now available!")
        server.io_loop.start()
        # Return server instance for potential stopping later
        self.bokeh_server = server
        
        return server
    
    def stop_bokeh_server(self):
        """Stop the Bokeh server if running"""
        if hasattr(self, 'bokeh_server') and self.bokeh_server:
            self.bokeh_server.stop()
            logging.info("Bokeh server stopped")
        return None


def main():
    parser = argparse.ArgumentParser(description='Monitor SLURM GPU jobs')
    parser.add_argument('--username', '-u', required=True, help='Username to monitor')
    parser.add_argument('--partition', '-p', default='GPU', help='Partition to monitor (default: GPU)')
    parser.add_argument('--interval', '-i', type=int, default=5, help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--duration', '-d', type=int, help='Monitoring duration in seconds (optional)')
    parser.add_argument('--output-dir', '-o', default='gpu_monitoring_output', help='Output directory for results')
    parser.add_argument('--csv-file', '-c', help='CSV file to save raw data')
    parser.add_argument('--backend', '-b', choices=['matplotlib', 'bokeh'], default='bokeh',
                        help='Plotting backend to use (default: bokeh)')
    parser.add_argument('--bokeh_port', '-bp', type=int, default=5006, help='Port for Bokeh server (default: 5006)')
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = SlurmGPUMonitor(args.username, args.partition, args.interval)
    
    # Monitor jobs
    logging.info(f"Starting GPU monitoring for user: {args.username}")
    if args.backend == 'bokeh':
        monitor.plotting_backend = 'bokeh'
    else:
        monitor.plotting_backend = 'matplotlib'
        
    if args.backend == 'bokeh':
        # Set up Bokeh server
        output_file(f"{args.output_dir}/gpu_monitoring_dashboard.html")
        monitor.start_bokeh_server(args.bokeh_port, args.output_dir)
    else:
        logging.info(f"Using {args.backend} for plotting. Results will be saved in {args.output_dir}/")
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    job_data = monitor.monitor_all_jobs(args.duration)
    
    # If Bokeh server is running, stop it
    monitor.stop_bokeh_server()
    
    # Save results
    if job_data:
        # Save CSV if requested
        if args.csv_file:
            monitor.save_data(job_data, args.csv_file)
        
        # Generate report
        monitor.generate_report(job_data, args.output_dir)
        logging.info(f"Monitoring complete. Results saved to {args.output_dir}/")
    else:
        logging.warning("No data collected")


if __name__ == "__main__":
    main()