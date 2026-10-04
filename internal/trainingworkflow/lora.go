package trainingworkflow

import (
	"errors"
	"path/filepath"

	"overgo/internal/densecausal"
	"overgo/internal/loratrain"
	"overgo/internal/peft"
)

// LoRAAdapterDirectory names, within a LoRA run's checkpoint, the directory
// that holds the trained adapter as a Hugging Face PEFT adapter.
const LoRAAdapterDirectory = "adapter"

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
	From string
}

// trainer starts the run's adapter over the model read from modelDirectory,
// which a new adapter's PEFT config names as the model it was trained on.
func (lora LoRA) trainer(model *densecausal.Model, modelDirectory string) (*loratrain.Trainer, error) {
	if lora.From == "" {
		return loratrain.New(model, loratrain.Config{Rank: lora.Rank, Alpha: lora.Alpha, Targets: lora.Targets, Seed: lora.Seed, BaseModel: filepath.Base(modelDirectory)})
	}
	if lora.Rank != 0 || lora.Alpha != 0 || len(lora.Targets) != 0 {
		return nil, errors.New("training workflow: a LoRA run that continues an adapter takes the adapter's own rank, alpha and targets")
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
