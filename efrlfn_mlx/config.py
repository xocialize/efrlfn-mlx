"""EfRLFN configuration (EvgeneyBogatyrev/EfRLFN, ICLR 2026)."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EfRLFNConfig:
    in_channels: int = 3
    out_channels: int = 3
    feature_channels: int = 52
    num_blocks: int = 6
    upscale: int = 4

    @classmethod
    def x2(cls) -> "EfRLFNConfig": return cls(upscale=2)
    @classmethod
    def x4(cls) -> "EfRLFNConfig": return cls(upscale=4)

GDRIVE_IDS = {"x2": "1VeoW94hN1X-8kxGXQSyR53YzRqF1htKQ", "x4": "1vJgrsz62IAMeS9i2ChDhQGO6UO1ZUXhr"}
