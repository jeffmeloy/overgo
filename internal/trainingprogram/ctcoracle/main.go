// Generate the CPU CTC fixture with the explicitly selected Python interpreter:
// go run ./internal/trainingprogram/ctcoracle -python <python.exe>
// Output is JSON on stdout; this tool never edits a fixture or loads a GPU.
package main

import (
	"context"
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
	flag.Parse()
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path: *python, Args: []string{"-c", oracle},
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

const oracle = `
import json
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
revision = "7661cd9c6b841b62b7f411aa52ec51f05457263b"
if torch.version.git_version != revision:
    raise RuntimeError("CTC oracle revision differs from the reviewed CPU implementation")

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
