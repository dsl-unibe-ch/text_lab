# Batch job templates (planned)

This folder will hold Slurm job templates for running Text Lab features on
large data sets without the interactive app, for example transcribing a
folder of recordings as a job array.

Each template will call the `textlab` command (`src/textlab/cli.py`) inside
the Apptainer image, which in turn calls the same backend code as the app.
Nothing is implemented yet: the refactor first separates each feature's
backend from the Streamlit UI so these commands can be added without
duplicating logic. The planned commands are sketched in the `cli.py` module
of each feature package under `src/textlab/features/`.
