// Generate the CPU CTC fixture with the explicitly selected Python interpreter:
// go run ./internal/trainingprogram/ctcoracle -python <python.exe>
// Output is JSON on stdout; this tool never edits a fixture or loads a GPU.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
)

func main() {
	clioptions.MainNamed("ctc-oracle", run)
}

func run() error {
	python := flag.String("python", "python", "interpreter containing the pinned PyTorch oracle")
	projection := flag.Bool("projection", false, "generate frozen-output input-projection CTC gradients")
	trainingShard := flag.String("training-shard", "", "Parquet training shard for the first-record tokenizer oracle")
	tokenizer := flag.String("tokenizer", "", "tokenizer.json for the training-record oracle")
	flag.Parse()
	script := oracle
	if (*trainingShard == "") != (*tokenizer == "") || *projection && *trainingShard != "" {
		return fmt.Errorf("select either projection or training-shard plus tokenizer")
	}
	if *projection {
		script = projectionOracle
	}
	if *trainingShard != "" {
		script = trainingRecordOracle
	}
	digest := sha256.Sum256([]byte(script))
	args := []string{"-c", script, hex.EncodeToString(digest[:])}
	if *trainingShard != "" {
		args = append(args, *trainingShard, *tokenizer)
	}
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path: *python, Args: args,
		Env:    append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1"),
		Stdout: os.Stdout, Stderr: os.Stderr,
	})
	if err != nil {
		return err
	}
	if receipt.ExitCode != 0 {
		return fmt.Errorf("CTC oracle exited %d", receipt.ExitCode)
	}
	return nil
}

// The fixture keeps source text and its rejected raw tokenization alongside
// the explicitly lowercased training target. FLAC STREAMINFO supplies geometry;
// this generator does not decode audio or claim a waveform oracle.
const trainingRecordOracle = `
import hashlib, json, pathlib, sys
import pyarrow
import pyarrow.parquet as pq
import tokenizers
if pyarrow.__version__ != "24.0.0" or tokenizers.__version__ != "0.22.2":
    raise RuntimeError("training-record oracle libraries differ")
path = pathlib.Path(sys.argv[2])
row = next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=["audio.bytes", "text", "id"])).to_pylist()[0]
payload = row["audio"]["bytes"]
if payload[:4] != b"fLaC" or payload[4] & 127 != 0:
    raise RuntimeError("fixture requires FLAC STREAMINFO")
geometry = int.from_bytes(payload[18:26], "big")
if ((geometry >> 41) & 7) + 1 != 1:
    raise RuntimeError("fixture requires mono audio")
tok = tokenizers.Tokenizer.from_file(sys.argv[3])
target = row["text"].lower()
with path.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
result = dict(schema="overgo/ctc-training-record/v1", split=path.parent.name, shard=path.name,
    shard_sha256=digest, shard_bytes=path.stat().st_size, row=0, row_id=row["id"], text=row["text"],
    audio_sha256=hashlib.sha256(payload).hexdigest(), samples=geometry & ((1 << 36) - 1), sample_rate=geometry >> 44,
    normalization="lowercase", target_text=target, target_ids=tok.encode(target, add_special_tokens=False).ids,
    source_target_ids=tok.encode(row["text"], add_special_tokens=False).ids,
    pyarrow=pyarrow.__version__, tokenizers=tokenizers.__version__, generator_sha256=sys.argv[1])
print(json.dumps(result, indent=2, allow_nan=False))
`

const oraclePrelude = `
import json
import sys
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
revision = "7661cd9c6b841b62b7f411aa52ec51f05457263b"
if torch.version.git_version != revision:
    raise RuntimeError("CTC oracle revision differs from the reviewed CPU implementation")
`

const oracle = oraclePrelude + `
cases = [
    ("one_frame", 1, 3, 0, [1], [0.0, 1.0, -1.0]),
    ("empty_target", 3, 3, 0, [], [0.5, -0.5, 1.0, 1.0, 0.0, -1.0, -0.5, 0.25, 0.75]),
    ("multiple_paths", 3, 3, 0, [1], [0.5, -0.5, 1.0, 1.0, 0.0, -1.0, -0.5, 0.25, 0.75]),
    ("repeated_minimum", 3, 3, 0, [1, 1], [0.25, 1.0, -1.0, 0.5, -0.25, 0.75, -0.5, 1.0, 0.0]),
    ("repeated_many_paths", 5, 3, 0, [1, 1], [((i % 7) - 3) / 4 for i in range(15)]),
    ("nonzero_blank", 4, 3, 2, [0, 1], [((i % 5) - 2) / 2 for i in range(12)]),
    ("alternating_targets", 7, 4, 0, [1, 2, 1, 3], [((i % 11) - 5) / 4 for i in range(28)]),
    ("many_alignments", 16, 4, 0, [1, 2, 2, 3, 1], [((i % 13) - 6) / 4 for i in range(64)]),
    ("large_logits", 3, 3, 0, [1, 2], [1000.0, -1000.0, 0.0, -1000.0, 1000.0, 0.0, 1000.0, 0.0, -1000.0]),
    ("common_offset", 3, 3, 0, [1], [100000.5, 99999.5, 100001.0, 100001.0, 100000.0, 99999.0, 99999.5, 100000.25, 100000.75]),
    ("single_class_empty", 4, 1, 0, [], [0.0, 1.0, -1.0, 1000.0]),
    ("impossible_repeat", 2, 3, 0, [1, 1], [0.0] * 6),
    ("impossible_length", 1, 3, 0, [1, 2], [0.0] * 3),
]
result = []
for name, frames, vocabulary, blank, targets, values in cases:
    # Start with exactly the float32 values the Go API receives, then perform
    # the independent loss and autograd computation in CPU float64.
    logits = torch.tensor(values, device="cpu", dtype=torch.float32).to(torch.float64).reshape(frames, vocabulary).requires_grad_()
    loss = F.ctc_loss(logits.log_softmax(-1), torch.tensor(targets, dtype=torch.long),
                      torch.tensor(frames), torch.tensor(len(targets)), blank=blank,
                      reduction="sum", zero_infinity=False)
    case = dict(name=name, frames=frames, vocabulary=vocabulary, blank=blank,
                targets=targets, logits=logits.detach().flatten().tolist())
    if torch.isfinite(loss):
        loss.backward()
        case.update(loss=loss.item(), gradient=logits.grad.flatten().tolist())
    else:
        case["refusal"] = "impossible-alignment"
    result.append(case)
print(json.dumps(dict(schema="overgo/ctc-objective-oracle/v1", torch_version=torch.__version__,
                     torch_revision=revision, device="cpu", dtype="float64-from-float32",
                     reduction="sum", zero_infinity=False,
                     source="https://github.com/pytorch/pytorch/blob/" + revision + "/aten/src/ATen/native/LossCTC.cpp",
                     cases=result), indent=2, allow_nan=False))
`

const projectionOracle = oraclePrelude + `
cases = [
    ("identity", 5, 3, 4, 0, [1, 2], True),
    ("mixed_projection", 5, 3, 4, 0, [1, 2], False),
    ("repeated_target", 6, 3, 4, 0, [1, 1], True),
    ("empty_target", 4, 2, 3, 0, [], True),
    ("nonzero_blank", 5, 4, 3, 2, [0, 1], False),
]
result = []
for name, frames, width, vocabulary, blank, targets, identity in cases:
    def values(count, period, divisor):
        return [((index % period) - period // 2) / divisor for index in range(count)]
    def tensor(values, shape):
        return torch.tensor(values, device="cpu", dtype=torch.float32).to(torch.float64).reshape(shape)
    hidden = tensor(values(frames * width, 7, 4), (frames, width)).requires_grad_()
    output = tensor(values(vocabulary * width, 11, 8), (vocabulary, width))
    bias = tensor(values(vocabulary, 5, 16), (vocabulary,))
    projection = torch.eye(width, dtype=torch.float64) if identity else tensor(values(width * width, 9, 8), (width, width))
    projection.requires_grad_()
    logits = F.linear(F.linear(hidden, projection), output, bias)
    logits.retain_grad()
    loss = F.ctc_loss(logits.log_softmax(-1), torch.tensor(targets, dtype=torch.long),
                      torch.tensor(frames), torch.tensor(len(targets)), blank=blank,
                      reduction="sum", zero_infinity=False)
    loss.backward()
    result.append(dict(name=name, frames=frames, width=width, vocabulary=vocabulary,
                       blank=blank, targets=targets, hidden=hidden.detach().flatten().tolist(),
                       output_weight=output.flatten().tolist(), output_bias=bias.flatten().tolist(),
                       projection=projection.detach().flatten().tolist(),
                       logits=logits.detach().flatten().tolist(), loss=loss.item(),
                       projection_gradient=projection.grad.flatten().tolist(),
                       logits_gradient=logits.grad.flatten().tolist()))
print(json.dumps(dict(schema="overgo/linear-ctc-oracle/v1", torch_revision=revision,
                     device="cpu", dtype="float64-from-float32", reduction="sum",
                     generator_sha256=sys.argv[1], cases=result), indent=2, allow_nan=False))
`
