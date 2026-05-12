from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18 as tv_resnet18,
    resnet50 as tv_resnet50,
)


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _default_resnet_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class DTypeAwareLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


class DTypeAwareConv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x,
            weight.to(x.dtype),
            None if bias is None else bias.to(x.dtype),
        )


class AbsEncoderVisual(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def prepare_images(self, images: Sequence[Any]) -> torch.Tensor:
        raise NotImplementedError


class Resnet18(AbsEncoderVisual):
    def __init__(self, pretrained: bool = False, image_size: int = 224):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_resnet18(weights=weights)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.image_transform = _default_resnet_transform(image_size=image_size)
        self._output_size = 512

    def forward(self, image: Tensor) -> Tensor:
        enc = self.feature_extractor(image.float())
        return enc.squeeze(-1).squeeze(-1)

    def output_size(self) -> int:
        return self._output_size

    def prepare_images(self, images: Sequence[Any]) -> Tensor:
        return torch.stack([self.image_transform(_to_pil_rgb(image)) for image in images], dim=0)


class Resnet50(AbsEncoderVisual):
    def __init__(self, pretrained: bool = False, image_size: int = 224):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_resnet50(weights=weights)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.image_transform = _default_resnet_transform(image_size=image_size)
        self._output_size = 2048

    def forward(self, image: Tensor) -> Tensor:
        enc = self.feature_extractor(image.float())
        return enc.squeeze(-1).squeeze(-1)

    def output_size(self) -> int:
        return self._output_size

    def prepare_images(self, images: Sequence[Any]) -> Tensor:
        return torch.stack([self.image_transform(_to_pil_rgb(image)) for image in images], dim=0)


class ResnetGMLP(AbsEncoderVisual):
    def __init__(
        self,
        num_gmlp_layers: int = 1,
        num_resnet_layers: int = 18,
        pretrained: bool = False,
        image_size: int = 224,
    ):
        super().__init__()

        if num_resnet_layers == 18:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet_model = tv_resnet18(weights=weights)
            self._output_size = 512
        elif num_resnet_layers == 50:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet_model = tv_resnet50(weights=weights)
            self._output_size = 2048
        else:
            raise ValueError(f"Unsupported ResNet depth: {num_resnet_layers}")

        self.seq_len = 7 * 7
        self.image_transform = _default_resnet_transform(image_size=image_size)
        self.feature_extractor = torch.nn.Sequential(*(list(resnet_model.children())[:-2]))

        if num_gmlp_layers > 0:
            try:
                from g_mlp_pytorch import Residual, PreNorm, gMLPBlock
            except ImportError as exc:
                try:
                    from g_mlp_pytorch import gMLPBlock
                except ImportError as inner_exc:
                    raise ImportError(
                        "ResnetGMLP requires g_mlp_pytorch. Install it or use another visual encoder."
                    ) from inner_exc

                class PreNorm(nn.Module):
                    def __init__(self, dim: int, fn: nn.Module):
                        super().__init__()
                        self.norm = nn.LayerNorm(dim)
                        self.fn = fn

                    def forward(self, x: Tensor) -> Tensor:
                        return self.fn(self.norm(x))

                class Residual(nn.Module):
                    def __init__(self, fn: nn.Module):
                        super().__init__()
                        self.fn = fn

                    def forward(self, x: Tensor) -> Tensor:
                        return self.fn(x) + x

            dim = self._output_size
            dim_ff = dim // 2
            self.gmlp = torch.nn.Sequential(
                *[
                    Residual(
                        PreNorm(
                            dim,
                            gMLPBlock(
                                dim=dim,
                                dim_ff=dim_ff,
                                seq_len=self.seq_len,
                            ),
                        )
                    )
                    for _ in range(num_gmlp_layers)
                ]
            )
        else:
            self.gmlp = None

    def forward(self, image: Tensor) -> Tensor:
        enc = self.feature_extractor(image.float())
        enc = enc.view(enc.shape[0], enc.shape[1], -1).permute(0, 2, 1)
        if self.gmlp is not None:
            enc = self.gmlp(enc)
        return enc

    def output_size(self) -> int:
        return self._output_size

    def prepare_images(self, images: Sequence[Any]) -> Tensor:
        return torch.stack([self.image_transform(_to_pil_rgb(image)) for image in images], dim=0)


class CLIPVisualEncoder(AbsEncoderVisual):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        return_sequence: bool = False,
    ):
        super().__init__()
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModel
        except ImportError as exc:
            raise ImportError(
                "CLIPVisualEncoder requires transformers with CLIPVisionModel support."
            ) from exc

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.return_sequence = return_sequence
        self._output_size = self.model.config.hidden_size

    def forward(self, image: Tensor) -> Tensor:
        outputs = self.model(pixel_values=image.float())
        if self.return_sequence:
            return outputs.last_hidden_state
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    def output_size(self) -> int:
        return self._output_size

    def prepare_images(self, images: Sequence[Any]) -> Tensor:
        pil_images = [_to_pil_rgb(image) for image in images]
        return self.processor(images=pil_images, return_tensors="pt")["pixel_values"]


class AbsFeatureFuser(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_temporal_concat = False

    @abstractmethod
    def forward(self, enc_speech: Tensor, enc_visual: Tensor) -> Tensor:
        raise NotImplementedError


def _visual_to_global(visual: Tensor) -> Tensor:
    if visual.dim() == 2:
        return visual
    if visual.dim() == 3:
        return visual.mean(dim=1)
    raise ValueError(f"Expected visual tensor with 2 or 3 dims, got {tuple(visual.shape)}")


def _visual_to_sequence(visual: Tensor) -> Tensor:
    if visual.dim() == 2:
        return visual.unsqueeze(1)
    if visual.dim() == 3:
        return visual
    raise ValueError(f"Expected visual tensor with 2 or 3 dims, got {tuple(visual.shape)}")


class SelectSpeech(AbsFeatureFuser):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        return speech


class ConcatProjFuser(AbsFeatureFuser):
    def __init__(self, dim_speech: int, dim_visual: int, use_residual: bool = True):
        super().__init__()
        self.proj = DTypeAwareConv1d(dim_speech + dim_visual, dim_speech, kernel_size=1)
        self.use_residual = use_residual

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        _, steps, _ = speech.shape
        visual = _visual_to_global(visual).to(dtype=speech.dtype)
        visual = visual.unsqueeze(1).repeat(1, steps, 1)
        out = torch.cat((speech, visual), dim=2).permute(0, 2, 1)
        out = self.proj(out).permute(0, 2, 1)
        if self.use_residual:
            out = out + speech
        return out


class ProjConcatFuser(AbsFeatureFuser):
    def __init__(
        self,
        dim_speech: int,
        dim_visual: int,
        p_speech: float = 0.5,
        use_residual: bool = True,
    ):
        super().__init__()
        assert 0 < p_speech < 1
        dim_speech_out = max(1, min(int(dim_speech * p_speech), dim_speech - 1))
        dim_visual_out = dim_speech - dim_speech_out
        self.proj_speech = DTypeAwareConv1d(dim_speech, dim_speech_out, kernel_size=1)
        self.proj_visual = DTypeAwareConv1d(dim_visual, dim_visual_out, kernel_size=1)
        self.use_residual = use_residual

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        _, steps, _ = speech.shape
        speech_out = self.proj_speech(speech.permute(0, 2, 1)).permute(0, 2, 1)
        visual = _visual_to_global(visual).to(dtype=speech.dtype)
        visual_out = self.proj_visual(visual.unsqueeze(2)).permute(0, 2, 1)
        visual_out = visual_out.repeat(1, steps, 1)
        out = torch.cat((speech_out, visual_out), dim=2)
        if self.use_residual:
            out = out + speech
        return out


class ProjConcatProjFuser(AbsFeatureFuser):
    def __init__(
        self,
        dim_speech: int,
        dim_visual: int,
        dim_speech_inter: int = 128,
        dim_visual_inter: int = 128,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if use_layer_norm:
            self.norm_speech = DTypeAwareLayerNorm(dim_speech)
            self.norm_visual = DTypeAwareLayerNorm(dim_visual)
        else:
            self.norm_speech = torch.nn.Identity()
            self.norm_visual = torch.nn.Identity()
        self.proj_speech = DTypeAwareConv1d(dim_speech, dim_speech_inter, kernel_size=1)
        self.proj_visual = DTypeAwareConv1d(dim_visual, dim_visual_inter, kernel_size=1)
        self.proj_back = DTypeAwareConv1d(
            dim_speech_inter + dim_visual_inter,
            dim_speech,
            kernel_size=1,
        )
        self.activation = torch.nn.GELU()

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        _, steps, _ = speech.shape
        speech_out = self.norm_speech(speech).permute(0, 2, 1)
        speech_out = self.proj_speech(speech_out)

        visual = _visual_to_global(visual).to(dtype=speech.dtype)
        visual_out = self.norm_visual(visual).unsqueeze(2)
        visual_out = self.proj_visual(visual_out)
        visual_out = visual_out.repeat(1, 1, steps)

        out = torch.cat((speech_out, visual_out), dim=1)
        out = self.activation(out)
        out = self.proj_back(out).permute(0, 2, 1)
        return out + speech


class ConcatTemp(AbsFeatureFuser):
    def __init__(self, dim_speech: int, dim_visual: int):
        super().__init__()
        self.is_temporal_concat = True
        self.norm_visual = DTypeAwareLayerNorm(dim_visual)
        self.proj_visual = DTypeAwareConv1d(dim_visual, dim_speech, kernel_size=1)

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        visual = _visual_to_sequence(visual).to(dtype=speech.dtype)
        visual_out = self.norm_visual(visual).permute(0, 2, 1)
        visual_out = self.proj_visual(visual_out).permute(0, 2, 1)
        return torch.cat((visual_out, speech), dim=1)


class AudioVisualCrossAttentionFuser(AbsFeatureFuser):
    def __init__(
        self,
        dim_speech: int,
        dim_visual: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = -4.0,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(dim_visual, dim_speech)
        self.audio_norm = nn.LayerNorm(dim_speech)
        self.visual_norm = nn.LayerNorm(dim_speech)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_speech,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(dim_speech)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        speech_fp32 = speech.float()
        visual_seq = _visual_to_sequence(visual).float()
        visual_proj = self.visual_proj(visual_seq)

        q = self.audio_norm(speech_fp32)
        k = self.visual_norm(visual_proj)
        attn_out, _ = self.cross_attn(
            query=q,
            key=k,
            value=k,
            need_weights=False,
        )

        gate = torch.sigmoid(self.gate_logit).to(dtype=attn_out.dtype, device=attn_out.device)
        fused = speech_fp32 + gate * attn_out
        fused = self.out_norm(fused)
        return fused.to(dtype=speech.dtype)


class VisualAttentionPrefixFuser(AbsFeatureFuser):
    def __init__(
        self,
        dim_speech: int,
        dim_visual: int,
        num_heads: int = 8,
        num_queries: int = 8,
        dropout: float = 0.1,
        gate_init: float = -4.0,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")

        self.is_temporal_concat = True
        self.visual_proj = nn.Linear(dim_visual, dim_speech)
        self.visual_norm = nn.LayerNorm(dim_speech)
        self.query_norm = nn.LayerNorm(dim_speech)
        self.prefix_norm = nn.LayerNorm(dim_speech)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_speech,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim_speech) * (dim_speech**-0.5))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        speech_fp32 = speech.float()
        visual_seq = _visual_to_sequence(visual).float()
        visual_proj = self.visual_norm(self.visual_proj(visual_seq))
        queries = self.query_norm(self.queries.expand(speech.shape[0], -1, -1))

        prefix, _ = self.attn(
            query=queries,
            key=visual_proj,
            value=visual_proj,
            need_weights=False,
        )
        gate = torch.sigmoid(self.gate_logit).to(dtype=prefix.dtype, device=prefix.device)
        prefix = gate * self.prefix_norm(prefix)
        fused = torch.cat((prefix, speech_fp32), dim=1)
        return fused.to(dtype=speech.dtype)


class GatedSequenceConcatFuser(AbsFeatureFuser):
    def __init__(
        self,
        dim_speech: int,
        dim_visual: int,
        num_heads: int = 8,
        num_queries: int = 8,
        dropout: float = 0.1,
        gate_init: float = -4.0,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")

        self.is_temporal_concat = True
        self.visual_proj = nn.Linear(dim_visual, dim_speech)
        self.visual_norm = nn.LayerNorm(dim_speech)
        self.query_norm = nn.LayerNorm(dim_speech)
        self.resampler_norm = nn.LayerNorm(dim_speech)
        self.resampler = nn.MultiheadAttention(
            embed_dim=dim_speech,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim_speech) * (dim_speech**-0.5))
        self.audio_type_embedding = nn.Parameter(torch.zeros(1, 1, dim_speech))
        self.visual_type_embedding = nn.Parameter(torch.zeros(1, 1, dim_speech))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, speech: Tensor, visual: Tensor) -> Tensor:
        speech_fp32 = speech.float()
        visual_seq = _visual_to_sequence(visual).float()
        visual_proj = self.visual_norm(self.visual_proj(visual_seq))
        queries = self.query_norm(self.queries.expand(speech.shape[0], -1, -1))

        visual_tokens, _ = self.resampler(
            query=queries,
            key=visual_proj,
            value=visual_proj,
            need_weights=False,
        )
        visual_tokens = self.resampler_norm(visual_tokens)

        gate = torch.sigmoid(self.gate_logit).to(dtype=visual_tokens.dtype, device=visual_tokens.device)
        audio_memory = speech_fp32 + self.audio_type_embedding
        visual_memory = gate * (visual_tokens + self.visual_type_embedding)
        fused = torch.cat((audio_memory, visual_memory), dim=1)
        return fused.to(dtype=speech.dtype)


def build_visual_encoder(
    name: str,
    *,
    pretrained: bool = False,
    image_size: int = 224,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_return_sequence: bool = False,
    num_gmlp_layers: int = 1,
    num_resnet_layers: int = 18,
) -> AbsEncoderVisual:
    key = str(name).strip().lower()
    if key == "resnet18":
        return Resnet18(pretrained=pretrained, image_size=image_size)
    if key == "resnet50":
        return Resnet50(pretrained=pretrained, image_size=image_size)
    if key == "resnet_gmlp":
        return ResnetGMLP(
            num_gmlp_layers=num_gmlp_layers,
            num_resnet_layers=num_resnet_layers,
            pretrained=pretrained,
            image_size=image_size,
        )
    if key == "clip":
        return CLIPVisualEncoder(
            model_name=clip_model_name,
            return_sequence=clip_return_sequence,
        )
    raise ValueError(f"Unsupported visual encoder: {name}")


def build_feature_fuser(
    name: str,
    *,
    dim_speech: int,
    dim_visual: int,
    p_speech: float = 0.5,
    use_residual: bool = True,
    dim_speech_inter: int = 128,
    dim_visual_inter: int = 128,
    use_layer_norm: bool = True,
    attn_num_heads: int = 8,
    attn_dropout: float = 0.1,
    attn_gate_init: float = -4.0,
    attn_num_queries: int = 8,
) -> AbsFeatureFuser:
    key = str(name).strip().lower()
    if key == "select_speech":
        return SelectSpeech()
    if key == "concat_proj":
        return ConcatProjFuser(
            dim_speech=dim_speech,
            dim_visual=dim_visual,
            use_residual=use_residual,
        )
    if key == "proj_concat":
        return ProjConcatFuser(
            dim_speech=dim_speech,
            dim_visual=dim_visual,
            p_speech=p_speech,
            use_residual=use_residual,
        )
    if key == "proj_concat_proj":
        return ProjConcatProjFuser(
            dim_speech=dim_speech,
            dim_visual=dim_visual,
            dim_speech_inter=dim_speech_inter,
            dim_visual_inter=dim_visual_inter,
            use_layer_norm=use_layer_norm,
        )
    if key == "concat_temp":
        return ConcatTemp(dim_speech=dim_speech, dim_visual=dim_visual)
    if key == "cross_attn_gate":
        return AudioVisualCrossAttentionFuser(
            dim_speech=dim_speech,
            dim_visual=dim_visual,
            num_heads=attn_num_heads,
            dropout=attn_dropout,
            gate_init=attn_gate_init,
        )
    if key == "attn_prefix":
        return VisualAttentionPrefixFuser(
            dim_speech=dim_speech,
            dim_visual=dim_visual,
            num_heads=attn_num_heads,
            num_queries=attn_num_queries,
            dropout=attn_dropout,
            gate_init=attn_gate_init,
        )
    if key == "gated_seq_concat":
        return GatedSequenceConcatFuser(
            dim_speech=dim_speech,
            dim_visual=dim_visual,
            num_heads=attn_num_heads,
            num_queries=attn_num_queries,
            dropout=attn_dropout,
            gate_init=attn_gate_init,
        )
    raise ValueError(f"Unsupported feature fuser: {name}")
