package peft

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// moduleWeight is the tensor an adapted module's update lands in.
const moduleWeight = ".weight"

// MergeReport counts what a merge wrote.
type MergeReport struct {
	// Tensors is every tensor of the merged checkpoint; Merged is how many of
	// them received an adapter update.
	Tensors, Merged int
}

// Merge adds each module's update into its weight in the safetensors
// checkpoint at base and publishes the result at destination, with the same
// shards, tensor names and dtypes: F32 and BF16 weights are merged in F32 and
// rounded back to their storage. The checkpoint's config and tokenizer files
// are the caller's to copy beside it.
func Merge(adapter Adapter, base, destination string) (MergeReport, error) {
	if err := adapter.Validate(); err != nil {
		return MergeReport{}, err
	}
	source, err := safetensors.OpenSource(base)
	if err != nil {
		return MergeReport{}, fmt.Errorf("peft: open base checkpoint: %w", err)
	}
	defer source.Close()
	adapted := make(map[string]string, len(adapter.Modules))
	for module, factors := range adapter.Modules {
		name := module + moduleWeight
		weight, found := source.Tensors[name]
		switch {
		case !found:
			return MergeReport{}, fmt.Errorf("peft: base checkpoint has no tensor %q for adapter module %q", name, module)
		case weight.DType != "F32" && weight.DType != "BF16":
			return MergeReport{}, fmt.Errorf("peft: base tensor %q is %s: only F32 and BF16 weights merge", name, weight.DType)
		case len(weight.Shape) != factorRank || weight.Shape[0] != uint64(factors.Out) || weight.Shape[1] != uint64(factors.In):
			return MergeReport{}, fmt.Errorf("peft: base tensor %q has shape %v, the adapter module is %d by %d", name, weight.Shape, factors.Out, factors.In)
		}
		adapted[name] = module
	}
	names := source.Names()
	plan, err := mergePlan(source, adapter)
	if err != nil {
		return MergeReport{}, err
	}
	writer, err := safetensors.NewStreamingWriter(destination, plan)
	if err != nil {
		return MergeReport{}, err
	}
	for _, name := range names {
		if writer.Completed(name) {
			continue
		}
		tensor := source.Tensors[name]
		module, isAdapted := adapted[name]
		if !isAdapted {
			if err := writer.WriteTensor(name, uint64(tensor.Size()), tensor.Reader()); err != nil {
				return MergeReport{}, err
			}
			continue
		}
		payload, err := mergedPayload(adapter, module, tensor)
		if err != nil {
			return MergeReport{}, err
		}
		if err := writer.WriteTensor(name, uint64(len(payload)), bytes.NewReader(payload)); err != nil {
			return MergeReport{}, err
		}
	}
	if err := writer.Finalize(); err != nil {
		return MergeReport{}, err
	}
	return MergeReport{Tensors: len(names), Merged: len(adapted)}, nil
}

// adapterMetadata names the plan entry that binds a merge's staged output to
// the adapter it merges: a staged partial of another adapter has another plan,
// so it is refused, never resumed into a checkpoint of two adapters mixed.
const adapterMetadata = "overgo_adapter"

// baseMetadata names the plan entry that binds a staged merge to the base it
// reads: two bases of one tensor catalog and different bytes have different
// plans, so a partial staged from one is refused when the other is merged,
// never resumed into a checkpoint whose tensors came from two models.
const baseMetadata = "overgo_base"

// baseDigest is the identity of the bytes a merge reads: the digest of each opened
// shard, in name order, which the source computes from the handles it will read.
func baseDigest(digests []safetensors.ShardDigest) string {
	hash := sha256.New()
	for _, shard := range digests {
		_ = binary.Write(hash, binary.LittleEndian, []uint64{uint64(len(shard.Name)), shard.Size})
		_, _ = io.WriteString(hash, shard.Name)
		_, _ = hash.Write(shard.Digest[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// mergePlan is the output catalog of merging adapter into source: the source's
// shards, tensor names, dtypes and shapes, bound to the adapter and to the
// content of the base. Binding the base reads it once more than the merge does.
func mergePlan(source *safetensors.Source, adapter Adapter) (safetensors.StreamingPlan, error) {
	digests, err := source.ShardDigests()
	if err != nil {
		return safetensors.StreamingPlan{}, fmt.Errorf("peft: identify the base checkpoint: %w", err)
	}
	names := source.Names()
	plan := safetensors.StreamingPlan{Metadata: map[string]string{"format": "pt", adapterMetadata: adapter.Digest(), baseMetadata: baseDigest(digests)}}
	for _, shard := range source.Shards() {
		spec := safetensors.StreamingShardSpec{Name: shard}
		for _, name := range names {
			if tensor := source.Tensors[name]; tensor.Shard == shard {
				spec.Tensors = append(spec.Tensors, safetensors.StreamingTensorSpec{Name: name, DType: tensor.DType, Shape: tensor.Shape})
			}
		}
		plan.Shards = append(plan.Shards, spec)
	}
	return plan, nil
}

// Digest is the identity of what the adapter adds to a weight: its scale rule
// and, per module in name order, the name, shape, alpha and factors. A hash
// never fails a write. A staged merge and what a change corpus asks of its
// runs are bound to the value, so a change to what is hashed moves both.
func (adapter Adapter) Digest() string {
	hash := sha256.New()
	_ = binary.Write(hash, binary.LittleEndian, adapter.Config.RSLoRA)
	for _, name := range slices.Sorted(maps.Keys(adapter.Modules)) {
		module := adapter.Modules[name]
		_ = binary.Write(hash, binary.LittleEndian, []int64{int64(len(name)), int64(module.Rank), int64(module.In), int64(module.Out)})
		_, _ = io.WriteString(hash, name)
		_ = binary.Write(hash, binary.LittleEndian, module.Alpha)
		_ = binary.Write(hash, binary.LittleEndian, module.Down)
		_ = binary.Write(hash, binary.LittleEndian, module.Up)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// mergedPayload is the module's base weight plus its update, in the weight's own storage.
func mergedPayload(adapter Adapter, module string, weight safetensors.Tensor) ([]byte, error) {
	values, err := safetensors.ReadF32(weight)
	if err != nil {
		return nil, err
	}
	if err := adapter.AddDelta(values, module); err != nil {
		return nil, err
	}
	if weight.DType == "BF16" {
		payload := make([]byte, len(values)*binaryschema.Uint16Bytes)
		for index, value := range values {
			binary.LittleEndian.PutUint16(payload[index*binaryschema.Uint16Bytes:], dtype.Float32ToBF16(value))
		}
		return payload, nil
	}
	payload := make([]byte, len(values)*binaryschema.Uint32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(payload[index*binaryschema.Uint32Bytes:], math.Float32bits(value))
	}
	return payload, nil
}
