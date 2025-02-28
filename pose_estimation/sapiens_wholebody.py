import os
import tempfile
import shutil
import subprocess
import numpy as np
from pathlib import Path


def detect(image_path: str, output_path: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
        temp_input_dir = Path(temp_input_dir)
        temp_output_dir = Path(temp_output_dir)
        image_filename = Path(image_path).name
        temp_image_path = temp_input_dir / image_filename
        shutil.copy2(image_path, temp_image_path)
        
        script_path = os.path.join(os.getcwd(), 'pipelines/1_pose_estimation.sh')
        cmd = [script_path, temp_input_dir, temp_output_dir]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pose estimation failed: {e.stderr.decode()}")
        
        result_path = temp_output_dir / image_filename.replace('.png', '.json')
        assert result_path.exists(), f"Pose estimation result file not found: {result_path}"
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(result_path, output_path)
