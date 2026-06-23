import numpy as np
import torch

from s2s_omni.codec_data import align_wav_to_num_frames, base_id_from_id
from s2s_omni.sglang_code2wav_capture import stack_code_chunks


def test_base_id_from_speed_or_chunk_suffix():
    assert base_id_from_id("AUD0001__speed_2") == "AUD0001"
    assert base_id_from_id("AUD0001__chunk_0003") == "AUD0001"
    assert base_id_from_id("AUD0001") == "AUD0001"


def test_align_wav_to_num_frames_pads_and_trims():
    assert align_wav_to_num_frames(np.ones(3, dtype=np.float32), 2, 4).shape == (8,)
    trimmed = align_wav_to_num_frames(np.arange(10, dtype=np.float32), 2, 4)
    assert trimmed.tolist() == list(range(8))


def test_stack_code_chunks_returns_quantizer_first_codes():
    chunks = [
        torch.arange(16, dtype=torch.long),
        torch.arange(16, dtype=torch.long) + 100,
    ]
    codes = stack_code_chunks(chunks)
    assert codes.shape == (16, 2)
    assert codes[0].tolist() == [0, 100]
    assert codes[15].tolist() == [15, 115]
