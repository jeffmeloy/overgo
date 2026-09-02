package inference

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// LoRAConfig: startup adapter path and scale.
type LoRAConfig struct {
	Path  string
	Scale float32
}

// LoRAScale: adapter ID and requested scale.
type LoRAScale struct {
	ID    int     `json:"id"`
	Scale float32 `json:"scale"`
}

// LoRAAdapterInfo: loaded adapter control-plane state.
type LoRAAdapterInfo struct {
	ID                    int      `json:"id"`
	Path                  string   `json:"path"`
	Scale                 float32  `json:"scale"`
	ALoRAInvocationTokens []uint32 `json:"alora_invocation_tokens,omitempty"`
}

func (r *Runner) activeALoRA(ids []tokenizer.TokenID) (int, int, error) {
	enabled := -1
	for index, loaded := range r.loraAdapters {
		if loaded.scale == 0 {
			continue
		}
		if len(loaded.adapter.InvocationTokens) == 0 {
			return -1, -1, nil
		}
		if enabled >= 0 {
			return -1, -1, errors.New("inference: multiple aLoRA adapters are enabled")
		}
		enabled = index
	}
	if enabled < 0 {
		return -1, -1, nil
	}
	invocation := r.loraAdapters[enabled].adapter.InvocationTokens
	for start := len(ids) - len(invocation); start >= 0; start-- {
		matched := true
		for offset, expected := range invocation {
			if ids[start+offset] < 0 || uint32(ids[start+offset]) != expected {
				matched = false
				break
			}
		}
		if matched {
			return enabled, start, nil
		}
	}
	return enabled, -1, nil
}

type loadedLoRA struct {
	adapter   *model.LoRAAdapter
	scale     float32
	signature [32]byte
}

func loRAStaticSignature(adapter *model.LoRAAdapter) [32]byte {
	if adapter == nil {
		return [32]byte{}
	}
	hasher := sha256.New()
	writeFingerprintString(hasher, adapter.Path)
	var encoded [4]byte
	binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(adapter.Alpha))
	_, _ = hasher.Write(encoded[:])
	for _, token := range adapter.InvocationTokens {
		binary.LittleEndian.PutUint32(encoded[:], token)
		_, _ = hasher.Write(encoded[:])
	}
	names := make([]string, 0, len(adapter.Weights))
	for name := range adapter.Weights {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		writeFingerprintString(hasher, name)
		weight := adapter.Weights[name]
		for _, values := range [][]float32{weight.A.Data, weight.B.Data} {
			for _, value := range values {
				binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
				_, _ = hasher.Write(encoded[:])
			}
		}
	}
	var result [32]byte
	copy(result[:], hasher.Sum(nil))
	return result
}

func (r *Runner) currentLoRASignature() [32]byte {
	if r == nil || len(r.loraAdapters) == 0 {
		return [32]byte{}
	}
	hasher := sha256.New()
	for _, loaded := range r.loraAdapters {
		signature := loaded.signature
		if signature == ([32]byte{}) {
			signature = loRAStaticSignature(loaded.adapter)
		}
		_, _ = hasher.Write(signature[:])
		var encoded [4]byte
		binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(loaded.scale))
		_, _ = hasher.Write(encoded[:])
	}
	var result [32]byte
	copy(result[:], hasher.Sum(nil))
	return result
}

func validatedLoRAScales(count int, scales []LoRAScale) ([]float32, error) {
	next := make([]float32, count)
	seen := make(map[int]struct{}, len(scales))
	for _, requested := range scales {
		if requested.ID < 0 || requested.ID >= len(next) {
			return nil, fmt.Errorf("inference: LoRA adapter ID %d is out of range", requested.ID)
		}
		if _, duplicate := seen[requested.ID]; duplicate {
			return nil, fmt.Errorf("inference: LoRA adapter ID %d is duplicated", requested.ID)
		}
		if math.IsNaN(float64(requested.Scale)) || math.IsInf(float64(requested.Scale), 0) {
			return nil, fmt.Errorf("inference: LoRA adapter ID %d scale is invalid", requested.ID)
		}
		seen[requested.ID] = struct{}{}
		next[requested.ID] = requested.Scale
	}
	return next, nil
}

func loRAScale(loaded loadedLoRA, weight model.LoRAWeight) float32 {
	bridge, _, valid := weight.B.MatrixExtents()
	if loaded.adapter == nil || loaded.scale == 0 || !valid {
		return 0
	}
	scale := loaded.scale
	if loaded.adapter.Alpha != 0 {
		scale *= loaded.adapter.Alpha / float32(bridge)
	}
	return scale
}

func (r *Runner) newGraphBuilder() *tensor.Builder {
	builder := tensor.NewBuilder()
	// Inference graphs multiply half-precision weights in their own dtype
	// on tensor cores; the host reference executes the same graph exactly.
	builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
	if r == nil || len(r.loraAdapters) == 0 {
		return builder
	}
	bindings := make(map[string][]tensor.LoRADefinition)
	for adapterID, loaded := range r.loraAdapters {
		if loaded.adapter == nil || loaded.scale == 0 {
			continue
		}
		for baseName, weight := range loaded.adapter.Weights {
			scale := loRAScale(loaded, weight)
			prefix := fmt.Sprintf("adapter.%d.%s", adapterID, baseName)
			bindings[baseName] = append(bindings[baseName], tensor.LoRADefinition{
				AName: prefix + ".lora_a", BName: prefix + ".lora_b",
				AShape: weight.A.Shape, BShape: weight.B.Shape,
				AData: weight.A.Data, BData: weight.B.Data,
				Scale: scale, Embedding: weight.Embedding,
			})
		}
	}
	builder.SetLoRA(bindings)
	return builder
}

func (r *Runner) applyLoRAEmbeddingSelection(name string, indices []uint32, base reference.Value) (reference.Value, error) {
	for _, loaded := range r.loraAdapters {
		weight, ok := loaded.adapter.Weights[name]
		if !ok || !weight.Embedding {
			continue
		}
		scale := loRAScale(loaded, weight)
		if scale == 0 {
			continue
		}
		bridge, entries, _ := weight.A.MatrixExtents()
		_, outputExtent, _ := weight.B.MatrixExtents()
		for token, index := range indices {
			if uint64(index) >= uint64(entries) {
				return reference.Value{}, fmt.Errorf("inference: LoRA embedding index %d is out of range", index)
			}
			a := weight.A.Data[int(index)*bridge : (int(index)+1)*bridge]
			for output := range outputExtent {
				var delta float32
				b := weight.B.Data[output*bridge : (output+1)*bridge]
				for inner := range bridge {
					delta += b[inner] * a[inner]
				}
				base.Data[token*outputExtent+output] += scale * delta
			}
		}
	}
	return base, nil
}

func (r *Runner) applyLoRALogits(name string, hidden, logits []float32) {
	for _, loaded := range r.loraAdapters {
		weight, ok := loaded.adapter.Weights[name]
		if !ok || weight.Embedding {
			continue
		}
		scale := loRAScale(loaded, weight)
		if scale == 0 {
			continue
		}
		inputExtent, bridge, _ := weight.A.MatrixExtents()
		_, outputExtent, _ := weight.B.MatrixExtents()
		projected := make([]float32, bridge)
		for row := range bridge {
			for column, value := range hidden {
				projected[row] += weight.A.Data[row*inputExtent+column] * value
			}
		}
		for row := range outputExtent {
			var delta float32
			for column, value := range projected {
				delta += weight.B.Data[row*bridge+column] * value
			}
			logits[row] += scale * delta
		}
	}
}

// LoRAAdapters: detached global adapter state.
func (r *Runner) LoRAAdapters() []LoRAAdapterInfo {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]LoRAAdapterInfo, len(r.loraAdapters))
	for index, loaded := range r.loraAdapters {
		result[index] = LoRAAdapterInfo{
			ID: index, Path: loaded.adapter.Path, Scale: loaded.scale,
			ALoRAInvocationTokens: slices.Clone(loaded.adapter.InvocationTokens),
		}
	}
	return result
}

// SetLoRAScales: replaces global scales; omitted adapters disabled.
func (r *Runner) SetLoRAScales(ctx context.Context, scales []LoRAScale) error {
	if r == nil {
		return errRunnerNil
	}
	if ctx == nil {
		return errors.New("inference: LoRA context is nil")
	}
	if err := r.lockOpen(); err != nil {
		return err
	}
	next, err := validatedLoRAScales(len(r.loraAdapters), scales)
	if err != nil {
		r.mu.Unlock()
		return err
	}
	changed := false
	for index := range next {
		if r.loraAdapters[index].scale != next[index] {
			r.loraAdapters[index].scale = next[index]
			changed = true
		}
	}
	if !changed {
		r.mu.Unlock()
		return nil
	}
	caches := r.detachPromptCaches()
	r.mu.Unlock()
	failed, releaseErr := releasePromptCaches(ctx, caches)
	if len(failed) > 0 {
		r.mu.Lock()
		closed := r.closed
		if !closed {
			r.promptCaches = append(r.promptCaches, failed...)
		}
		r.mu.Unlock()
		if closed {
			_, retryErr := releasePromptCaches(context.Background(), failed)
			releaseErr = errors.Join(releaseErr, retryErr)
		}
	}
	return releaseErr
}
