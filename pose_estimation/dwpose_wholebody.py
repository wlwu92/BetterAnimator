from PIL import Image
import onnxruntime as ort
import numpy as np

from third_party.MimicMotion.mimicmotion.dwpose.onnxpose import inference_pose
from third_party.MimicMotion.mimicmotion.dwpose.onnxdet import inference_detector


class DwposeWholebody:
    def __init__(self, model_det, model_pose, device='cpu'):
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        provider_options = None if device == 'cpu' else [{'device_id': 0}]

        self.session_det = ort.InferenceSession(
            path_or_bytes=model_det, providers=providers,  provider_options=provider_options
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=model_pose, providers=providers, provider_options=provider_options
        )

    def detect(self, image: str | Image.Image) -> dict:
        if isinstance(image, str):
            image = Image.open(image)
        det_result = inference_detector(self.session_det, np.array(image))
        keypoints, scores = inference_pose(self.session_pose, det_result, np.array(image))
        return np.concatenate((keypoints, scores[..., None]), axis=-1)