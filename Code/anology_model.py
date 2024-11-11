from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer ,AutoProcessor, CLIPVisionModel
class multi_modal_model(nn.Module):
    def __init__(self):
