import wandb
import os

wandb.login(key = "d2151653b41de4857b8dc8e5091d811777a475ae")
run = wandb.init(project="Mod-HyDE-Project",job_type="upload-model",entity='astarag2843')
dirs = ["qwensmall-full_parameter"]

for dir in dirs:
    print(f"Directory: {dir}")
    artifact = wandb.Artifact(name=dir,type="model")
    artifact.add_dir(local_path=dir)
    run.log_artifact(artifact)
    # run.log_artifact(dir)