import torch.nn as nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class ExistingDecoderLayer(nn.Module):
    pass


class ModelWithStaleNoSplitModule(nn.Module):
    _no_split_modules = ["ExistingDecoderLayer", "MissingVisionBlock"]

    def __init__(self):
        super().__init__()
        self.layer = ExistingDecoderLayer()


def test_fsdp_wrap_policy_ignores_missing_default_no_split_modules():
    model = ModelWithStaleNoSplitModule()

    assert get_fsdp_wrap_policy(model) is not None


def test_fsdp_wrap_policy_rejects_missing_explicit_wrap_class():
    model = ModelWithStaleNoSplitModule()

    try:
        get_fsdp_wrap_policy(model, config={"transformer_layer_cls_to_wrap": ["MissingVisionBlock"]})
    except Exception as exc:
        assert "MissingVisionBlock" in str(exc)
    else:
        raise AssertionError("Expected missing explicit wrap class to raise")
