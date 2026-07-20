import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.generation.utils import CWEDetector
import torch
import time
from threading import Thread
from typing import List, Optional, Dict, Any

class LRModel:
    def __init__(self, model_path, device, torch_type, log_dir: str = "cwe_logs", max_injections_per_rule: int = 1):
        self.torch_type = torch_type
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_type
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.cwe_detector = CWEDetector(
            cwe_rules_path="dataprocess/cwe_rules_combined_safety_knowledge_optimized.json",
            tokenizer=self.tokenizer,
            log_dir=log_dir,
            max_injections_per_rule=max_injections_per_rule
        )
        self.log_dir = log_dir

    def inference(
        self,
        messages,
        gen_kwargs: Dict[str, Any],
        use_cwe_detection: bool = True,
        cwe_check_interval: int = 5,
        cwe_injection_mode: str = "immediate",
        cwe_reconstruction_model_type: str = "auto",
    ):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_special_tokens=True,
            skip_prompt=True
        )

        generation_kwargs = {
            **inputs,
            **gen_kwargs,
            "streamer": streamer,
        }
        
        if use_cwe_detection:
            generation_kwargs["cwe_detector"] = self.cwe_detector
            generation_kwargs["cwe_check_interval"] = cwe_check_interval
            generation_kwargs["cwe_injection_mode"] = cwe_injection_mode
            generation_kwargs["cwe_reconstruction_model_type"] = cwe_reconstruction_model_type

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for next_token in streamer:
            yield next_token
