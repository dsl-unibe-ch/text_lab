# How to Launch Text Lab

Text Lab runs as an interactive app on the University of Bern's High Performance Computing cluster (UBELIX) via the Open OnDemand portal.

**IMPORTANT:**  
**You must activate your UBELIX account before you can use Text Lab.**  
See the **[UBELIX account activation guide](https://hpc-unibe-ch.github.io/firststeps/accessUBELIX/)** for details.

## Step-by-Step Guide

1. Log in to the **[UBELIX Open OnDemand Portal](https://ondemand.hpc.unibe.ch)**.
2. Navigate to **Interactive Apps** or **Data Science Lab Services** and select **Text Lab**.
3. You will see a configuration form.

## Basic Configuration

By default, only the essential options are shown:

| Parameter | Recommended Setting | Description |
| :--- | :--- | :--- |
| **Account** | `gratis` or your available account | Select the account under which the session should run. |
| **Job Time (hours)** | `1` to `4` | Total time you plan to use Text Lab. If the session reaches the selected limit, it will stop automatically. |
| **Advanced Slurm Options** | Unchecked for most users | Leave unchecked for standard usage. Enable it only if you need to manually choose GPU and other Slurm settings. |

## Advanced Slurm Options

If you check **Advanced Slurm Options**, additional parameters will appear.

| Parameter | Recommended Setting | Description |
| :--- | :--- | :--- |
| **SLURM Partition** | `gpu` | Standard GPU partition for most users. Use `gpu-invest` only if your account and project require it. |
| **Quality of Service (QoS)** | Depends on your account | Choose a QoS that matches your account and partition. For many standard users, `job_gratis` is the correct choice. |
| **GPU Type** | `rtx4090` for most tasks | Suitable for most transcription and OCR tasks. Use `A100`, `H100`, or `H200` only if you need more GPU memory or want to run larger LLMs. |
| **Number of GPU(s) requested** | `1` | One GPU is enough for most use cases. Request more than one only for large LLM workloads. |
| **wckey** | Optional | Provide a valid project identifier for accounting if needed. This is usually optional when using the `gratis` account. |
| **Reservation** | Optional | Only fill this in if you have been given an active Slurm reservation name, for example for a workshop or course. |

## Notes on QoS and GPU Selection

- `job_gratis` is typically the default choice for users launching Text Lab with a gratis account.
- `job_gpu` or `job_gpu_preemptable` may be relevant if you are using a specific project allocation.
- `A100` or `H100` are recommended only for larger LLM workloads and usually together with `job_gpu_preemptable`, depending on your account setup.
- If you are unsure, start with the default settings and only enable advanced options when necessary.

## Starting the Session

1. Click **Launch**.
2. Wait for the job to start. The status will change from **Queued** to **Running**.
3. Click **Connect to Text Lab** to open the interface in your browser.

## More Information

For more information about Slurm parameters and job configuration on UBELIX, see the **[UBELIX documentation](https://hpc-unibe-ch.github.io/)**.