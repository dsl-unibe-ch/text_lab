# Text Lab 

This repository contains the code for the **Text Lab ** – an in-development suite of NLP and text processing tools. Currently, the app includes:

- **Transcribe:** A tool for processing audio input and converting it to text.
- **Chat:** An interactive chat interface powered by LLMs.

More NLP/text processing functionalities will be added over time.

---

## Overview

The Text Lab Interactive App is designed to be an **interactive app**, meaning that:
- Users interact with the app via a web interface.
- The app provides real-time responses, such as streaming outputs.
- All tasks (transcription, chatting, etc.) occur within a unified, user-friendly interface.

> **Note:** This app is still under active development and may have some issues or incomplete features.

---

## Launching the App

When launching the app, you can specify several parameters to tailor its execution:

### Job Time (hours)
- **Job Time (hours):** Specify the maximum duration for which the app will be active. After this time, the app will automatically stop. This parameter helps ensure fair allocation of compute resources.

### GPU Type
- **GPU Type:** You can request a specific GPU type for running your job.
- **Preemptable Option in Quality of Service (QoS):** For example, if you request an **A100 GPU**, note that it is available as a **preemptable** resource.  
  - A **preemptable** GPU resource can be reclaimed by the system if needed, which might result in interruptions.
  - This option allows efficient sharing of high-end GPUs among many users.

For more detailed guidelines on job submission and GPU options, please refer to the [HPC Documentation](https://hpc-unibe-ch.github.io/).

---

## How to Launch the App

To launch the app, you typically submit a job with parameters such as:
- **Job Time (hours):** How long the app should remain active.
- **GPU Selection:** Which GPU to use.
- **SLURM Partition:** GPU or GPU-invest 
- **Number of GPU(s) requested:** If you plan to use the chat option with a very larg model, then you might need to request more than 1 GPU.please refer to the [HPC Documentation](https://hpc-unibe-ch.github.io/).

The app runs only for the specified duration, ensuring effective resource usage on the HPC system.

The app is accessible via [OnDemand HPC](https://ondemand.hpc.unibe.ch/) and is available **only within the Unibe internal network**.

---

## Support & Further Information

For support with the app or related NLP services, please contact:  
**dsl.support@unibe.ch**

For additional details on job submissions and HPC resource usage, check the [HPC Documentation](https://hpc-unibe-ch.github.io/).

---

