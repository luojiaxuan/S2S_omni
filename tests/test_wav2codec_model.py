import pytest

torch = pytest.importorskip("torch")

from s2s_omni.wav2codec import Wav2OmniCodecConfig, Wav2OmniCodecModel, codec_accuracy, wav2codec_loss


def test_wav2codec_forward_loss_and_metrics_shape():
    cfg = Wav2OmniCodecConfig(
        sample_rate=24000,
        hop_length=64,
        num_quantizers=2,
        codebook_size=32,
        hidden_size=32,
        conv_channels=(8, 16),
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )
    model = Wav2OmniCodecModel(cfg)
    wav = torch.randn(2, 3 * cfg.hop_length)
    labels = torch.randint(0, cfg.codebook_size, (2, cfg.num_quantizers, 3))
    mask = torch.ones(2, 3, dtype=torch.bool)
    logits = model(wav, mask)
    assert logits.shape == (2, cfg.num_quantizers, 3, cfg.codebook_size)
    loss = wav2codec_loss(logits, labels)
    assert torch.isfinite(loss)
    metrics = codec_accuracy(logits, labels)
    assert metrics["valid_codes"] == 12

