import os
import yaml
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import decord

class VideoMAEHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = []
        self.weights_path = None
        self.verb_list_path = None

    def load_model(self, weights_path, verb_list_path):
        """
        Load VideoMAE V2 model and custom labels.
        """
        self.weights_path = weights_path
        self.verb_list_path = verb_list_path

        # 1. Load labels from data.yaml (similar to YOLO format or simple list)
        with open(verb_list_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'names' in data:
                # YOLO format: {names: {0: 'verb1', 1: 'verb2'}}
                self.labels = [data['names'][i] for i in sorted(data['names'].keys())]
            elif isinstance(data, list):
                self.labels = data
            else:
                # Fallback: line-by-line TXT or simple YAML list
                self.labels = [str(v) for v in data] if data else []

        num_labels = len(self.labels)

        # 2. Initialize VideoMAE V2 model
        # Using the base version as a starting point. 
        # Note: trust_remote_code=True is often required for VideoMAE V2 from OpenGVLab
        model_id = "OpenGVLab/video-mae-v2-base-kinetics" 
        
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            
            # 3. Load custom weights if provided
            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                # If the weights are wrapped in 'model' or 'state_dict' key
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Check for classification head mismatch and handle it
                # Transformers VideoMAE classification head is usually model.classifier
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            return True, f"Loaded model with {num_labels} labels."
        except Exception as e:
            return False, str(e)

    def predict(self, video_path, start_frame, end_frame, top_k=5):
        """
        Predict action labels for a given segment.
        """
        if self.model is None or self.processor is None:
            return None, "Model not loaded."

        try:
            # 1. Sample frames using decord
            vr = decord.VideoReader(video_path, num_threads=1)
            num_frames_available = len(vr)
            
            start = max(0, int(start_frame))
            end = min(num_frames_available - 1, int(end_frame))
            
            # VideoMAE typically expects 16 frames
            num_samples = 16
            if end - start < num_samples:
                # If segment is too short, just take everything and duplicate or pad
                indices = np.linspace(start, end, num_samples).astype(int)
            else:
                indices = np.linspace(start, end, num_samples).astype(int)
            
            video = vr.get_batch(indices).asnumpy() # (T, H, W, C)
            
            # 2. Preprocess
            # VideoMAEImageProcessor expects a list of frames or a 4D array (T, C, H, W)
            # Transform from (T, H, W, C) to (T, C, H, W)
            video = [Image.fromarray(frame) for frame in video]
            inputs = self.processor(list(video), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 3. Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get Top-K
                top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.labels)))
                
                results = []
                for i in range(top_probs.size(1)):
                    idx = top_indices[0, i].item()
                    results.append({
                        "label": self.labels[idx] if idx < len(self.labels) else f"Unknown_{idx}",
                        "score": top_probs[0, i].item()
                    })
                
                return results, None
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, str(e)
