##### RNGHandler - the ONLY source of randomness. One seed, keyed derivation, nothing stored.

from __future__ import annotations
from numpy.random import SeedSequence, Generator, PCG64
from hashlib import blake2b
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def _name_key(name) -> int:
    return int.from_bytes(blake2b(str(name).encode("utf-8"), digest_size=8).digest(), "big")


class RNGHandler:
    """
    One seed in, keyed Generators out.

    Args:
        seed = global seed. None -> fresh OS entropy (recorded on self.seed so it CAN be reproduced).
        bitgenerator = numpy BitGenerator name. PCG64 only for now.

    Usage (domain names are the convention, not enforced):
        rng("network", synapse_group)   builder: one stream per synapse group
        rng("noise", noise_type)        noise module: "membrane", "synapse", "threshold"
        rng("phase", phase_type)        phase-internal randomness
        rng("data", purpose)            splits and shuffles: "test_split", "folds", "shuffle"
        rng("fold", (k, e))             fold k's epoch-e shuffle (Operations.run_fit), INDEPENDENT of the folds and epochs before it   
        
    """
    _bitgenerators = {"PCG64": PCG64}

    def __init__(self, seed: int | None = None, bitgenerator: str = "PCG64"):
        if bitgenerator not in self._bitgenerators:
            raise ValueError(f"Unsupported bitgenerator {bitgenerator!r}. Supported: {tuple(self._bitgenerators)}")
        self.bitgenerator = bitgenerator
        self.seed = int(SeedSequence(seed).entropy)       ##resolves None -> a concrete, recordable seed

    def rng(self, domain: str, key="") -> Generator:
        """Deterministic Generator for (domain, key). Idempotent and order-independent."""
        seq = SeedSequence(self.seed, spawn_key=(_name_key(domain), _name_key(key)))
        return Generator(self._bitgenerators[self.bitgenerator](seq))

    @property
    def rng_spec(self) -> Dict:
        """Everything needed to rebuild this handler."""
        return {"seed": self.seed, "bitgenerator": self.bitgenerator,
                "derivation": "SeedSequence(seed, spawn_key=(blake2b(domain), blake2b(key)))"}

    @classmethod
    def from_rng_spec(cls, spec: Dict) -> "RNGHandler":
        return cls(seed=spec["seed"], bitgenerator=spec.get("bitgenerator", "PCG64"))

    def __repr__(self):
        return f"RNGHandler(seed={self.seed}, bitgenerator={self.bitgenerator!r})"
