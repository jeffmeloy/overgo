package trainingworkflow

import (
	"errors"
	"path/filepath"
	"strings"

	"overgo/internal/densecausal"
	"overgo/internal/loratrain"
	"overgo/internal/peft"
)

const (
	// LoRAAdapterDirectory names, within a LoRA run's checkpoint, the directory
	// that holds the trained adapter as a Hugging Face PEFT adapter.
	LoRAAdapterDirectory = "adapter"
	// LoRAAlphaPerRank is how many times its rank a LoRA adapter's alpha is
	// unless a run names another, so its scale is this many: the default
	// LlamaFactory uses.
	LoRAAlphaPerRank = 2
)

// LoRA configures a held-out text run that trains a low-rank adapter over the
// frozen model: every module's scale is Alpha over Rank, and Targets name the
// projections to adapt, all seven linear ones when empty. From, when set,
// names a PEFT adapter directory to continue instead: the adapter brings its
// own rank, alpha and modules, and the run's baseline is the model as that
// adapter leaves it.
type LoRA struct {
	Rank    int
	Alpha   float64
	Targets []string
	// Seed makes the factors' initialization repeat.
	Seed uint64
	// RSLoRA scales every module by alpha over the root of its rank, and UpRate
	// trains the up factor that many times faster than the down one (loratrain.Config).
	RSLoRA bool
	UpRate float64
	From   string
	// Quantized keeps each adapted projection of a GGUF base in the quantization the file
	// stores it in, dequantizing it as each layer reads it, so the run holds those weights
	// as stored bytes rather than in full precision. The model it publishes is then the base
	// as stored, the adapter it trained in the checkpoint's PEFT directory, which the server
	// loads beside the base or gguf merge-lora writes into it; a base that is not a GGUF
	// is refused.
	Quantized bool
}

// frozenWeights is the predicate naming the weights a Quantized run holds as
// stored: those the adapter adapts, by the targets that start one or by the
// modules of the adapter that continues.
func (lora LoRA) frozenWeights() (func(weight string) bool, error) {
	if lora.From == "" {
		return loratrain.Adapts(lora.Targets), nil
	}
	adapter, err := peft.Load(lora.From)
	if err != nil {
		return nil, err
	}
	return func(weight string) bool {
		module, found := strings.CutSuffix(weight, ".weight")
		_, adapted := adapter.Modules[module]
		return found && adapted
	}, nil
}

// trainer starts the run's adapter over the model read from modelDirectory,
// which a new adapter's PEFT config names as the model it was trained on.
func (lora LoRA) trainer(model *densecausal.Model, modelDirectory string) (*loratrain.Trainer, error) {
	if lora.From == "" {
		return loratrain.New(model, loratrain.Config{
			Rank: lora.Rank, Alpha: lora.Alpha, Targets: lora.Targets, Seed: lora.Seed, BaseModel: filepath.Base(modelDirectory),
			RSLoRA: lora.RSLoRA, UpRate: lora.UpRate,
		})
	}
	if lora.Rank != 0 || lora.Alpha != 0 || len(lora.Targets) != 0 || lora.RSLoRA || lora.UpRate != 0 {
		return nil, errors.New("training workflow: a LoRA run that continues an adapter takes the adapter's own rank, alpha, targets and scaling")
	}
	adapter, err := peft.Load(lora.From)
	if err != nil {
		return nil, err
	}
	return loratrain.Resume(model, adapter)
}

// adapterCompanion writes the adapter into a staged checkpoint as a PEFT directory.
func adapterCompanion(adapter peft.Adapter) func(stage string) error {
	return func(stage string) error { return peft.Save(filepath.Join(stage, LoRAAdapterDirectory), adapter) }
}
