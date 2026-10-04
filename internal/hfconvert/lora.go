package hfconvert

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
	"overgo/internal/gguf"
	"overgo/internal/jsonfile"
	"overgo/internal/peft"
)

// unitScaleTolerance is how far a module's rescaling factor may sit from one
// before its up factor is multiplied by it: float arithmetic on a scale that is
// exactly one lands within it, and a rescaling inside it is rounding noise.
const unitScaleTolerance = 1e-9

// linearDestinations are the GGUF suffixes of the decoder's linear weights a
// LoRA adapter may carry a pair for; norms and the embedding are not among them.
var linearDestinations = []string{".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight", ".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight"}

// LoRAOptions is one PEFT adapter's conversion.
type LoRAOptions struct {
	// Directory is the HF checkpoint the adapter was trained on: its config.json
	// names the architecture whose tensor names and rotary layout the adapter's
	// tensors follow.
	Directory string
	// OutputPath is the new GGUF adapter file.
	OutputPath string
}

// LoRAReport counts what a conversion wrote.
type LoRAReport struct {
	Modules, Tensors int
	OutputBytes      int64
}

// ConvertLoRA writes the PEFT adapter as the GGUF LoRA adapter inference
// loads: each module's pair under the GGUF name of the weight it adapts, in
// GGUF's reversed extents, with the up factor's rows of a rotary query or key
// weight permuted as the converter permuted the weight itself. The file's alpha
// is the adapter's; a module whose own scale differs from alpha over its rank
// has its up factor rescaled to carry it, so every module's update is exact.
func ConvertLoRA(adapter peft.Adapter, options LoRAOptions) (LoRAReport, error) {
	if err := adapter.Validate(); err != nil {
		return LoRAReport{}, err
	}
	directory, err := filepath.Abs(options.Directory)
	if err != nil {
		return LoRAReport{}, err
	}
	var config modelConfig
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &config); err != nil {
		return LoRAReport{}, fmt.Errorf("HF converter: config: %w", err)
	}
	profile, ok := archProfiles[config.ModelType]
	if !ok {
		return LoRAReport{}, fmt.Errorf("HF converter: a LoRA adapter for model type %q is unsupported", config.ModelType)
	}
	if err := validateConfig(config); err != nil {
		return LoRAReport{}, err
	}
	var tensors []gguf.TensorData
	for _, module := range slices.Sorted(maps.Keys(adapter.Modules)) {
		pair, err := loRATensors(adapter, module, profile, config)
		if err != nil {
			return LoRAReport{}, err
		}
		tensors = append(tensors, pair...)
	}
	metadata := []gguf.Metadata{
		gguf.StringMetadata("general.type", "adapter"),
		gguf.StringMetadata("general.architecture", profile.prefix),
		gguf.StringMetadata("adapter.type", "lora"),
		gguf.Float32Metadata("adapter.lora.alpha", float32(adapter.Config.Alpha)),
	}
	if err := gguf.WriteFileExclusive(options.OutputPath, metadata, tensors, gguf.WriteOptions{}); err != nil {
		return LoRAReport{}, err
	}
	written, err := os.Stat(options.OutputPath)
	if err != nil {
		return LoRAReport{}, err
	}
	return LoRAReport{Modules: len(adapter.Modules), Tensors: len(tensors), OutputBytes: written.Size()}, nil
}

// loRATensors is one module's pair as GGUF tensors.
func loRATensors(adapter peft.Adapter, name string, profile archProfile, config modelConfig) ([]gguf.TensorData, error) {
	module := adapter.Modules[name]
	destination, named := modelTensorName(name+".weight", profile)
	if !named || !isLinearDestination(destination) {
		return nil, fmt.Errorf("HF converter: adapter module %q is not a linear projection of a %s decoder layer", name, profile.prefix)
	}
	if layer, err := strconv.Atoi(strings.Split(strings.TrimPrefix(destination, "blk."), ".")[0]); err != nil || layer >= int(config.HiddenLayers) {
		return nil, fmt.Errorf("HF converter: adapter module %q is past the checkpoint's %d layers", name, config.HiddenLayers)
	}
	scale, err := adapter.Scale(name)
	if err != nil {
		return nil, err
	}
	// Inference scales a pair by the file's alpha over the pair's rank.
	up := module.Up
	if factor := scale * float64(module.Rank) / adapter.Config.Alpha; math.Abs(factor-1) > unitScaleTolerance {
		up = make([]float32, len(module.Up))
		for index, value := range module.Up {
			up[index] = value * float32(factor)
		}
	}
	var upReader io.Reader = float32Reader(up)
	if heads := ropePermuteHeads(destination, profile, config); heads != 0 {
		if upReader, err = permuteRopeRows(upReader, []uint64{uint64(module.Out), uint64(module.Rank)}, gguf.DTypeF32, heads); err != nil {
			return nil, fmt.Errorf("HF converter: adapter module %q: %w", name, err)
		}
	}
	return []gguf.TensorData{
		{Name: destination + ".lora_a", Shape: gguf.ReverseShape([]uint64{uint64(module.Rank), uint64(module.In)}), Type: gguf.DTypeF32, Data: float32Reader(module.Down)},
		{Name: destination + ".lora_b", Shape: gguf.ReverseShape([]uint64{uint64(module.Out), uint64(module.Rank)}), Type: gguf.DTypeF32, Data: upReader},
	}, nil
}

func isLinearDestination(destination string) bool {
	return slices.ContainsFunc(linearDestinations, func(suffix string) bool { return strings.HasSuffix(destination, suffix) })
}

// float32Reader is the little-endian bytes of values as a reader.
func float32Reader(values []float32) *bytes.Reader {
	data := make([]byte, len(values)*binaryschema.Uint32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*binaryschema.Uint32Bytes:], math.Float32bits(value))
	}
	return bytes.NewReader(data)
}
