package trainingworkflow

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

const (
	// resumeDirectory holds the F32 weights of a checkpoint that serves below F32, in a directory of its own for the
	// reason the average keeps one, and resumeWeightsFile names them: a run that trains on from the checkpoint reads
	// these, and a server the weights at the base's precision.
	resumeDirectory   = "resume"
	resumeWeightsFile = "weights.safetensors"
	// servedBF16 is the stored type a base's BF16 tensor keeps in its trained checkpoint.
	servedBF16 = "BF16"
)

// servedTypes is the stored type of each trained tensor its base holds in BF16: a trained model serves at its base's
// precision. A tensor the base holds otherwise, or does not hold, stays F32.
func servedTypes(base string, weights map[string][]float32) (map[string]string, error) {
	source, err := safetensors.OpenSource(base)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	types := map[string]string{}
	for name := range weights {
		if tensor, held := source.Tensors[name]; held && tensor.DType == servedBF16 {
			types[name] = servedBF16
		}
	}
	return types, nil
}

// saveServedWeights writes a trained model's weights into a staged checkpoint at its base's precision, and, where that
// is below F32, the F32 weights a run trains on from in resumeDirectory.
func saveServedWeights(stage, base string, weights map[string][]float32, shapes map[string][]int, metadata map[string]string) error {
	types, err := servedTypes(base, weights)
	if err != nil {
		return err
	}
	if err := safetensors.SaveTyped(filepath.Join(stage, trainingprogram.CheckpointWeights), weights, shapes, types, metadata); err != nil {
		return err
	}
	if len(types) == 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Join(stage, resumeDirectory), clioptions.OutputDirectoryMode); err != nil {
		return err
	}
	return safetensors.Save(filepath.Join(stage, resumeDirectory, resumeWeightsFile), weights, shapes, nil)
}

// restoreResumeWeights replaces a loaded checkpoint's weights by the F32 weights it keeps for a run that trains on from
// it, where it keeps them; a model directory that keeps none is left as it loaded.
func restoreResumeWeights(directory string, model *densecausal.Model) error {
	source, err := safetensors.OpenFile(filepath.Join(directory, resumeDirectory, resumeWeightsFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	defer source.Close()
	for name, values := range model.Weights {
		tensor, found := source.Tensors[name]
		if !found {
			return fmt.Errorf("training workflow: the checkpoint's resume weights have no %s", name)
		}
		read, err := safetensors.ReadF32(tensor)
		if err != nil {
			return err
		}
		if len(read) != len(values) {
			return fmt.Errorf("training workflow: the checkpoint's resume weights hold %d values for %s, the model's are %d", len(read), name, len(values))
		}
		copy(values, read)
	}
	return nil
}
