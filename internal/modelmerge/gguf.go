package modelmerge

import (
	"errors"
	"fmt"
	"io"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/quant"
)

// GGUFTask is a fine-tuned GGUF merged into a base: its file and the scale
// its difference from the base carries.
type GGUFTask struct {
	Path  string
	Scale float32
}

// MergeGGUF writes the task-arithmetic merge base + sum(scale * (task -
// base)) of GGUF models sharing the base's tensor inventory, as an F32 GGUF
// with the base's metadata; each tensor is merged as the writer reaches it,
// so memory holds one tensor per model.
func MergeGGUF(basePath string, tasks []GGUFTask, destination io.Writer) error {
	if len(tasks) == 0 {
		return errors.New("model merge: task arithmetic needs a fine-tuned model")
	}
	base, err := gguf.Open(basePath)
	if err != nil {
		return err
	}
	defer base.Close()
	tuned := make([]*gguf.File, 0, len(tasks))
	defer func() {
		for _, file := range tuned {
			_ = file.Close()
		}
	}()
	inventories := make([]map[string]gguf.TensorInfo, len(tasks))
	for index, task := range tasks {
		if !checked.Finite32(task.Scale) {
			return fmt.Errorf("model merge: %s has no finite scale", task.Path)
		}
		file, err := gguf.Open(task.Path)
		if err != nil {
			return err
		}
		tuned = append(tuned, file)
		inventories[index] = make(map[string]gguf.TensorInfo, len(file.Tensors))
		for _, info := range file.Tensors {
			inventories[index][info.Name] = info
		}
		if len(file.Tensors) != len(base.Tensors) {
			return fmt.Errorf("model merge: %s does not share the base's tensor inventory", task.Path)
		}
		for _, info := range base.Tensors {
			other, found := inventories[index][info.Name]
			if !found || other.Dimensions != info.Dimensions || other.Shape != info.Shape {
				return fmt.Errorf("model merge: %s does not share the base's tensor %q", task.Path, info.Name)
			}
		}
	}
	return base.RewriteTo(destination, gguf.DTypeF32, func(info gguf.TensorInfo, values []float32) error {
		baseValues := slices.Clone(values)
		for index, task := range tasks {
			other := inventories[index][info.Name]
			raw := make([]byte, other.Size)
			if err := tuned[index].ReadTensorData(other, raw); err != nil {
				return err
			}
			tunedValues, err := quant.Dequantize(other.Type, raw, uint64(len(values)))
			if err != nil {
				return err
			}
			for position := range values {
				values[position] += task.Scale * (tunedValues[position] - baseValues[position])
				if !checked.Finite32(values[position]) {
					return errors.New("model merge: task arithmetic produced a non-finite weight")
				}
			}
		}
		return nil
	})
}
