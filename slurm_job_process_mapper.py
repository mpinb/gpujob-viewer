import subprocess
import logging
import re
import os
from typing import List, Dict, Optional, Union

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SlurmJobProcessMapper:
    def __init__(self):
        self.ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    def get_job_processes(self, node: str, job_id: Union[str, int], user_id: Optional[str] = None) -> Dict:
        """
        Get processes running on a node for a specific SLURM job or job array.
        
        Args:
            node: Compute node hostname
            job_id: SLURM job ID (can be single job like "12345" or array job like "12345_1")
            user_id: Optional user ID to filter processes
            
        Returns:
            Dict containing job info and associated processes
        """
        try:
            # Handle job arrays - extract base job ID and array task ID
            base_job_id, array_task_id = self._parse_job_id(str(job_id))
            
            # Get SLURM job information first
            job_info = self._get_slurm_job_info(node, job_id)
            if not job_info:
                return {"error": f"Could not retrieve SLURM job info for {job_id}"}
            
            # Method 1: Use cgroups (most reliable for modern SLURM)
            processes_cgroup = self._get_processes_from_cgroup(node, base_job_id, array_task_id)
            logging.debug(f"Processes from cgroup for job {job_id}: {processes_cgroup}")
            
            # Method 2: Use SLURM_JOB_ID environment variable
            processes_env = self._get_processes_from_env(node, base_job_id, array_task_id)
            logging.debug(f"Processes from environment for job {job_id}: {processes_env}")
            
            # Method 3: Use process tree from job step PIDs
            processes_tree = self._get_processes_from_tree(node, job_info)
            logging.debug(f"Processes from process tree for job {job_id}: {processes_tree}")
            
            # Combine and deduplicate results
            all_processes = self._merge_process_lists(processes_cgroup, processes_env, processes_tree)
            logging.debug(f"Combined processes for job {job_id}: {all_processes}")
            
            # Filter by user if specified
            if user_id:
                all_processes = self._filter_processes_by_user(node, all_processes, user_id)
                logging.debug(f"Filtered processes for job {job_id} by user {user_id}: {all_processes}")
            
            return {
                "job_id": job_id,
                "base_job_id": base_job_id,
                "array_task_id": array_task_id,
                "node": node,
                "job_info": job_info,
                "processes": all_processes,
                "process_count": len(all_processes)
            }
            
        except Exception as e:
            logging.error(f"Error getting job processes for {job_id} on {node}: {e}")
            return {"error": str(e)}

    def _parse_job_id(self, job_id: str) -> tuple:
        """Parse job ID to handle job arrays."""
        if '_' in job_id:
            parts = job_id.split('_')
            return parts[0], parts[1]
        return job_id, None

    def _get_slurm_job_info(self, node: str, job_id: str) -> Dict:
        """Get SLURM job information."""
        cmd = f"scontrol show job {job_id}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        logging.info(f"Getting SLURM job info: {cmd}")
        
        if result.returncode != 0:
            logging.error(f"Failed to get SLURM job info: {result.stderr}")
            return {}
        
        # Parse scontrol output
        job_info = {}
        for line in result.stdout.split('\n'):
            if '=' in line:
                pairs = re.findall(r'(\w+)=([^\s]+)', line)
                for key, value in pairs:
                    job_info[key] = value
        
        return job_info

    def _get_processes_from_cgroup(self, node: str, base_job_id: str, array_task_id: Optional[str]) -> List[Dict]:
        """Get processes using cgroups (most reliable method)."""
        processes = []
        
        # Construct cgroup path - format varies by SLURM version
        if array_task_id:
            cgroup_patterns = [
                f"/sys/fs/cgroup/memory/slurm/uid_*/job_{base_job_id}/step_{array_task_id}/cgroup.procs",
                f"/sys/fs/cgroup/slurm/uid_*/job_{base_job_id}/step_{array_task_id}/cgroup.procs",
                f"/sys/fs/cgroup/systemd/slurm.slice/slurm-{base_job_id}_{array_task_id}.scope/cgroup.procs"
            ]
        else:
            cgroup_patterns = [
                f"/sys/fs/cgroup/memory/slurm/uid_*/job_{base_job_id}/cgroup.procs",
                f"/sys/fs/cgroup/slurm/uid_*/job_{base_job_id}/cgroup.procs",
                f"/sys/fs/cgroup/systemd/slurm.slice/slurm-{base_job_id}.scope/cgroup.procs"
            ]
        
        for pattern in cgroup_patterns:
            cmd = f"ssh {self.ssh_opts} {node} 'find {os.path.dirname(pattern)} -name \"{os.path.basename(pattern)}\" 2>/dev/null | head -1 | xargs cat 2>/dev/null'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        process_info = self._get_process_info(node, pid.strip())
                        if process_info:
                            process_info['source'] = 'cgroup'
                            processes.append(process_info)
                break
        
        return processes

    def _get_processes_from_env(self, node: str, base_job_id: str, array_task_id: Optional[str]) -> List[Dict]:
        """Get processes by searching for SLURM_JOB_ID environment variable."""
        processes = []
        
        # Search for processes with SLURM_JOB_ID in environment
        job_id_to_search = f"{base_job_id}_{array_task_id}" if array_task_id else base_job_id
        cmd = f"ssh {self.ssh_opts} {node} 'for pid in $(pgrep -f .; do if grep -l \"SLURM_JOB_ID={job_id_to_search}\" /proc/$pid/environ 2>/dev/null; then echo $pid; fi; done'"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    process_info = self._get_process_info(node, pid.strip())
                    if process_info:
                        process_info['source'] = 'environment'
                        processes.append(process_info)
        
        return processes

    def _get_processes_from_tree(self, node: str, job_info: Dict) -> List[Dict]:
        """Get processes from job step process tree."""
        processes = []
        
        # Look for BatchScript or other step PIDs in job info
        step_pids = []
        for key, value in job_info.items():
            if 'pid' in key.lower() and value.isdigit():
                step_pids.append(value)
        
        # Get process tree for each step PID
        for pid in step_pids:
            cmd = f"ssh {self.ssh_opts} {node} 'pstree -p {pid} 2>/dev/null | grep -o \"([0-9]*)\" | tr -d \"()\"'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                tree_pids = result.stdout.strip().split('\n')
                for tree_pid in tree_pids:
                    if tree_pid.strip():
                        process_info = self._get_process_info(node, tree_pid.strip())
                        if process_info:
                            process_info['source'] = 'process_tree'
                            processes.append(process_info)
        
        return processes

    def _get_process_info(self, node: str, pid: str) -> Optional[Dict]:
        """Get detailed information about a specific process."""
        cmd = f"ssh {self.ssh_opts} {node} 'ps -p {pid} -o pid,ppid,uid,user,cmd --no-headers 2>/dev/null'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            fields = result.stdout.strip().split(None, 4)
            if len(fields) >= 5:
                return {
                    'pid': fields[0],
                    'ppid': fields[1],
                    'uid': fields[2],
                    'user': fields[3],
                    'cmd': fields[4]
                }
        
        return None

    def _merge_process_lists(self, *process_lists) -> List[Dict]:
        """Merge and deduplicate process lists."""
        seen_pids = set()
        merged = []
        
        for process_list in process_lists:
            for process in process_list:
                pid = process.get('pid')
                if pid and pid not in seen_pids:
                    seen_pids.add(pid)
                    merged.append(process)
        
        return merged

    def _filter_processes_by_user(self, node: str, processes: List[Dict], user_id: str) -> List[Dict]:
        """Filter processes by user ID."""
        return [p for p in processes if p.get('user') == user_id or p.get('uid') == user_id]

    def get_all_slurm_jobs_processes(self, node: str, user_id: Optional[str] = None) -> Dict:
        """Get processes for all SLURM jobs running on a node."""
        try:
            # Get all running jobs on the node
            cmd = f"squeue -h -w {node} -o \"%i %u\""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"error": "Failed to get running jobs"}
            
            jobs_processes = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id, job_user = parts[0], parts[1]
                        
                        # Skip if user filter is specified and doesn't match
                        if user_id and job_user != user_id:
                            continue
                            
                        job_processes = self.get_job_processes(node, job_id, user_id)
                        jobs_processes[job_id] = job_processes
            
            return jobs_processes
            
        except Exception as e:
            logging.error(f"Error getting all job processes on {node}: {e}")
            return {"error": str(e)}


# Usage example:
if __name__ == "__main__":
    mapper = SlurmJobProcessMapper()
    
    # Example 1: Single job
    result = mapper.get_job_processes("compute-node-01", "12345")
    print(f"Processes for job 12345: {result}")
    
    # Example 2: Job array
    result = mapper.get_job_processes("compute-node-01", "12345_1")
    print(f"Processes for job array 12345_1: {result}")
    
    # Example 3: Filter by user
    result = mapper.get_job_processes("compute-node-01", "12345", user_id="username")
    print(f"Processes for job 12345 (user filtered): {result}")
    
    # Example 4: All jobs on a node
    result = mapper.get_all_slurm_jobs_processes("compute-node-01")
    print(f"All job processes: {result}")