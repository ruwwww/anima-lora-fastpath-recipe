from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from library.anima_session import AnimaModelSession
from networks.lora_anima import LoRAModule, LoRANetwork


def test_session_lifecycle_and_reset():
    # Setup dummy base model
    linear = nn.Linear(32, 64, bias=False)
    dummy_dit = nn.Module()
    dummy_dit.blocks = nn.ModuleList([linear])
    
    # Create LoRA
    lora = LoRAModule("test_linear", linear, lora_dim=4, alpha=4.0)
    lora.apply_to()
    
    network = LoRANetwork(
        text_encoders=[],
        unet=dummy_dit,
        multiplier=1.0,
        lora_dim=4,
        alpha=4.0,
        verbose=False,
    )
    network.unet_loras = [lora]
    
    key = {
        "model": "dummy",
        "rank": 4,
        "dtype": "bf16",
    }
    
    # Initialize session
    session = AnimaModelSession(
        session_key=key,
        text_encoders=[],
        vae=None,
        unet=dummy_dit,
        network=network,
    )
    AnimaModelSession.set_active_session(session)
    
    assert AnimaModelSession.get_active_session() is session
    assert AnimaModelSession.is_compatible({"model": "dummy", "rank": 4, "dtype": "bf16"}) is True
    assert AnimaModelSession.is_compatible({"model": "other", "rank": 4, "dtype": "bf16"}) is False
    
    # Test weight mutation & in-place reset
    lora.lora_up.weight.data.fill_(5.0)
    assert float(lora.lora_up.weight[0, 0]) == 5.0
    
    session.reset_for_new_job()
    assert float(lora.lora_up.weight[0, 0]) == 0.0
    assert session.job_count == 1
    
    # Close session
    AnimaModelSession.close_active_session()
    assert AnimaModelSession.get_active_session() is None
