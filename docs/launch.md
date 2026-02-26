# How to Launch Text Lab

Text Lab runs as an interactive app on the University's High Performance Computing cluster (UBELIX) via the Open OnDemand portal.

## Step-by-Step Guide

1.  Log in to the **[UBELIX Open OnDemand Portal](https://ondemand.hpc.unibe.ch)**.
2.  Navigate to **Interactive Apps** or **Data Science Lab Services** and select **Text Lab**.
3.  You will see a configuration form. Configure the resource requirements as follows:

### Configuration Parameters

| Parameter | Recommended Setting | Description |
| :--- | :--- | :--- |
**Account** | `gratis` | Selects the type of account to launch Text Lab. |
| **SLURM Partition** | `gpu` | Selects the partition with graphical processing units. |
| **QoS (Quality of Service)** | `job_gpu_preemptable` | Use `job_gpu_preemptable` for quick tasks with a chance of being disconnected. Use `job_gpu` if you have a specific allocation. |
| **Job Time** | `1` to `4` hours | Estimate how long you need the app. If the timer runs out, the session closes. |
| **GPU Type** | `rtx4090` | The **RTX4090** is powerful and sufficient for most Text Lab tasks (Transcription/OCR). Use A100/H100 only for very large LLM workloads. Certain LLMs will not run with RTX4090 |
| **Number of GPUs** | `1` | One GPU is sufficient for standard usage. |
| **WCKey** | *Optional* | Leave blank unless you have a specific project accounting key. |

### Advanced Mode
Checking **Advanced Mode** allows you to specify a "Reservation" if you have code for for a workshop or course. Otherwise, leave this unchecked.

### Starting the Session
1.  Click **Launch**.
2.  Wait for the job to start (the status will change from "Queued" to "Running").
3.  Click **Connect to Text Lab** to open the interface in your browser.

### More info on the configuration parameters 
Please check the UBELIX HPC documentation for more info on launching jobs: https://hpc-unibe-ch.github.io/
