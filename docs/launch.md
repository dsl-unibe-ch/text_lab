# How to Launch Text Lab

Text Lab runs as an interactive app on the University's High Performance Computing cluster (UBELIX) via the Open OnDemand portal.

## Step-by-Step Guide

1.  Log in to the **UBELIX Open OnDemand Portal**.
2.  Navigate to **Interactive Apps** and select **Text Lab**.
3.  You will see a configuration form. Configure the resource requirements as follows:

### Configuration Parameters

| Parameter | Recommended Setting | Description |
| :--- | :--- | :--- |
| **SLURM Partition** | `gpu` | Selects the partition with graphical processing units. |
| **QoS (Quality of Service)** | `job_gratis` | Use `job_gratis` for standard tasks. Use `job_gpu` if you have a specific allocation. |
| **Job Time** | `1` to `4` hours | Estimate how long you need the app. If the timer runs out, the session closes. |
| **GPU Type** | `rtx4090` | The **RTX4090** is powerful and sufficient for most Text Lab tasks (Transcription/OCR). Use A100/H100 only for very large LLM workloads. |
| **Number of GPUs** | `1` | One GPU is sufficient for standard usage. |
| **WCKey** | *Optional* | Leave blank unless you have a specific project accounting key. |

### Advanced Mode
Checking **Advanced Mode** allows you to specify a "Reservation" if the DSL team has set aside specific resources for a workshop or course. Otherwise, leave this unchecked.

### Starting the Session
1.  Click **Launch**.
2.  Wait for the job to start (the status will change from "Queued" to "Running").
3.  Click **Connect to Text Lab** to open the interface in your browser.
