#!/bin/bash

# How often we consider a duplicate (in minutes)
TTL_MIN=60

# Directory where we store timestamps of triggered jobs
STATE="$HOME/.gpu_array_monitor_state"
mkdir -p "$STATE"

# Only care about jobs from this user
USER_NAME=$(whoami)

# sacct prints accounting info; --parsable2 guarantees stable parsing
sacct -n -P --format=JobID,JobName,State,Gres,AllocTRES,Partition,ArrayTaskID,ArrayJobID,Submit,start \
      -u "$USER_NAME" \
      --state=R,PD \
      | while IFS='|' read -r jobid jobname state gres partition taskid arrayjobid submit start
do
    # Skip non-array jobs or trivial 1-element arrays
    if [[ -z "$arrayjobid" ]] || [[ "$arrayjobid" == "0" ]]; then
        continue
    fi

    # sacct reports multiple tasks separately; we only want once per array job
    # So only trigger for the "array job parent", e.g. 123456
    # The parent appears with jobid like 123456 and taskid empty or "0"
    parent_jobid="$arrayjobid"

    # Detect array size using sacct (one call outside the loop is possible, but OK here)
    array_size=$(sacct -n -P -j "$parent_jobid" --format=ArrayTaskID | grep -E '^[0-9]+$' | wc -l)

    if (( array_size < 10 )); then
        continue
    fi

    # Check GPU usage (by matching --gres=gpu or new --gpus= switch)
    gpu_requested=false

    # If the GRES field contains gpu
    [[ "$gres" == *"gpu"* ]] && gpu_requested=true

    # If AllocTRES field contains gpu
    [[ "$alloctres" == *"gpu="* ]] && gpu_requested=true

    if ! $gpu_requested ; then
        continue
    fi

    # Check state file for duplicate triggers
    stamp_file="$STATE/${parent_jobid}.stamp"
    if [[ -f "$stamp_file" ]]; then
        last=$(stat -c %Y "$stamp_file")
        now=$(date +%s)
        if (( now - last < TTL_MIN*60 )); then
            continue   # Recent trigger; skip
        fi
    fi

    # Mark as triggered
    touch "$stamp_file"

    # Trigger the slurm_gpu_monitor job
    /path/to/my_program --job "$parent_jobid"
done
