"""Independent GGUF v3 structural oracle for one tiny OLMoE block.

This writes bytes directly with Python's struct module. It does not use
Overgo's GGUF writer or model code, and makes no real-model accuracy claim.
"""

from hashlib import sha256
from pathlib import Path
from struct import pack
import json


ROOT = Path(__file__).resolve().parent
TENSORS = [
    ("token_embd.weight", (8, 32)),
    ("output_norm.weight", (8,)),
    ("output.weight", (8, 32)),
    ("blk.0.attn_norm.weight", (8,)),
    ("blk.0.attn_q.weight", (8, 8)),
    ("blk.0.attn_k.weight", (8, 8)),
    ("blk.0.attn_v.weight", (8, 8)),
    ("blk.0.attn_output.weight", (8, 8)),
    ("blk.0.attn_q_norm.weight", (8,)),
    ("blk.0.attn_k_norm.weight", (8,)),
    ("blk.0.ffn_norm.weight", (8,)),
    ("blk.0.ffn_gate_inp.weight", (8, 4)),
    ("blk.0.ffn_gate_exps.weight", (8, 12, 4)),
    ("blk.0.ffn_up_exps.weight", (8, 12, 4)),
    ("blk.0.ffn_down_exps.weight", (12, 8, 4)),
]
ALIGNMENT = 32


def string(value):
    encoded = value.encode("utf-8")
    return pack("<Q", len(encoded)) + encoded


def aligned(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tensor_data(seed, dimensions):
    count = 1
    for dimension in dimensions:
        count *= dimension
    return b"".join(pack("<f", ((index + seed) % 17 - 8) / 16) for index in range(count))


def main():
    header = bytearray(b"GGUF" + pack("<IQQ", 3, len(TENSORS), 1))
    header += string("general.architecture") + pack("<I", 8) + string("olmoe")
    payload = bytearray()
    inventory = []
    q_values = []
    for seed, (name, shape) in enumerate(TENSORS, 1):
        data = tensor_data(seed, shape)
        offset = aligned(len(payload))
        payload += b"\x00" * (offset - len(payload)) + data
        header += string(name) + pack("<I", len(shape))
        for dimension in shape:
            header += pack("<Q", dimension)
        header += pack("<IQ", 0, offset)  # GGML F32 and relative data offset
        inventory.append({"name": name, "shape": list(shape), "sha256": sha256(data).hexdigest()})
        if name == "blk.0.attn_q.weight":
            q_values = [((index + seed) % 17 - 8) / 16 for index in range(64)]
    header += b"\x00" * (aligned(len(header)) - len(header))
    blob = bytes(header + payload)
    (ROOT / "olmoe_tiny.gguf").write_bytes(blob)
    manifest = {
        "format": "GGUF-v3-independent-python",
        "architecture": "olmoe",
        "sha256": sha256(blob).hexdigest(),
        "tensors": inventory,
        "q_row_sums": [sum(q_values[row * 8:(row + 1) * 8]) for row in range(8)],
    }
    (ROOT / "olmoe_tiny.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
