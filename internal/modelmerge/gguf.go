package modelmerge

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/quant"
)

// tokenizerPrefix opens the metadata keys that declare a GGUF's vocabulary and
// its special tokens; the chat template, which a fine-tune may change, is not one.
const tokenizerPrefix = "tokenizer.ggml."

// differingTokenizer is the first tokenizer key the base declares that other
// does not declare the same, or "": a model of another tokenizer reads the
// base's token ids as other tokens, so a merge of the two is noise that the
// shared tensor shapes would not show.
func differingTokenizer(base, other *gguf.File) string {
	for _, entry := range base.Metadata {
		if !strings.HasPrefix(entry.Key, tokenizerPrefix) {
			continue
		}
		index := slices.IndexFunc(other.Metadata, func(candidate gguf.Metadata) bool { return candidate.Key == entry.Key })
		if index < 0 || !reflect.DeepEqual(other.Metadata[index].Value, entry.Value) {
			return entry.Key
		}
	}
	return ""
}

// GGUFTask is a fine-tuned GGUF merged into a base: its file and the scale
// its difference from the base carries.
type GGUFTask struct {
	Path  string
	Scale float32
}

// MergeGGUF writes the task-arithmetic merge base + sum(scale * (task -
// base)) of GGUF models sharing the base's tensor inventory, as an F32 GGUF
// with the base's metadata; each tensor is merged as the writer reaches it,
// so memory holds one tensor of the base, one of the model being added and the
// bytes it decodes from.
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
		if key := differingTokenizer(base, file); key != "" {
			return fmt.Errorf("model merge: %s declares another %s than the base, so its token ids are not the base's", task.Path, key)
		}
	}
	var kept float32 = 1
	for _, task := range tasks {
		kept -= task.Scale
	}
	// The sum is taken as kept*base + sum(scale*task), which is the same sum
	// and holds no copy of the base; the two buffers a tuned tensor is read
	// and decoded into are the largest tensor's, reused for every tensor.
	var raw []byte
	var decoded []float32
	return base.RewriteTo(destination, gguf.DTypeF32, func(info gguf.TensorInfo, values []float32) error {
		for position := range values {
			values[position] *= kept
		}
		for index, task := range tasks {
			other := inventories[index][info.Name]
			raw = slices.Grow(raw[:0], int(other.Size))[:other.Size]
			if err := tuned[index].ReadTensorData(other, raw); err != nil {
				return err
			}
			decoded = slices.Grow(decoded[:0], len(values))[:len(values)]
			if err := quant.DequantizeInto(other.Type, raw, decoded); err != nil {
				return err
			}
			for position, value := range decoded {
				values[position] += task.Scale * value
			}
		}
		for _, value := range values {
			if !checked.Finite32(value) {
				return errors.New("model merge: task arithmetic produced a non-finite weight")
			}
		}
		return nil
	})
}
