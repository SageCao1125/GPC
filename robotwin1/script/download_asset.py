from huggingface_hub import snapshot_download
import os

snapshot_download(repo_id="ZanxinChen/RoboTwin_asset", 
                  allow_patterns=['aloha_urdf.zip', 'main_models.zip'],
                  local_dir='.', 
                  repo_type="dataset",
                  resume_download=True)