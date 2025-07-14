#!/usr/bin/env python3

import subprocess
import sys

def test_squeue_parsing(username, partition):
    """Test squeue parsing with debug output"""
    
    print(f"Testing squeue for user: {username}, partition: {partition}")
    print("=" * 50)
    
    # Test 1: Basic squeue command
    print("1. Testing basic squeue command:")
    cmd1 = f"squeue -u {username}"
    result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
    print(f"Command: {cmd1}")
    print(f"Return code: {result1.returncode}")
    print("Output:")
    print(result1.stdout)
    print("-" * 30)
    
    # Test 2: With format specification
    print("2. Testing with format specification:")
    cmd2 = f"squeue -u {username} -o '%A %T %N %P %j' --noheader"
    result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    print(f"Command: {cmd2}")
    print(f"Return code: {result2.returncode}")
    print("Output:")
    print(result2.stdout)
    print("-" * 30)
    
    # Test 3: Parse the output
    print("3. Parsing the output:")
    if result2.returncode == 0:
        jobs = []
        for line in result2.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                print(f"Line: '{line}'")
                print(f"Parts: {parts}")
                print(f"Number of parts: {len(parts)}")
                
                if len(parts) >= 4:
                    job_id, status, node, partition_found = parts[:4]
                    job_name = parts[4] if len(parts) > 4 else "Unknown"
                    
                    print(f"  job_id: '{job_id}'")
                    print(f"  status: '{status}'")
                    print(f"  node: '{node}'")
                    print(f"  partition_found: '{partition_found}'")
                    print(f"  job_name: '{job_name}'")
                    print(f"  Looking for partition: '{partition}'")
                    print(f"  Status == 'R': {status == 'R'}")
                    print(f"  Partition match: {partition_found == partition}")
                    
                    if status == 'R' and partition_found == partition:
                        jobs.append({
                            'job_id': job_id,
                            'status': status,
                            'node': node,
                            'partition': partition_found,
                            'job_name': job_name
                        })
                        print(f"  -> JOB ADDED TO LIST")
                    else:
                        print(f"  -> Job skipped")
                print("-" * 20)
        
        print(f"Total jobs found: {len(jobs)}")
        for job in jobs:
            print(f"  {job}")
    else:
        print(f"Command failed with error: {result2.stderr}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python debug_squeue.py <username> <partition>")
        print("Example: python debug_squeue.py $USER GPU-long")
        sys.exit(1)
    
    username = sys.argv[1]
    partition = sys.argv[2]
    test_squeue_parsing(username, partition)