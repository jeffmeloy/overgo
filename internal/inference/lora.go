package inference

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

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
	sort.Strings(names)
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
	if loaded.adapter == nil || loaded.scale == 0 || weight.B.Shape.Dims[0] == 0 {
		return 0
	}
	scale := loaded.scale
	if loaded.adapter.Alpha != 0 {
		scale *= loaded.adapter.Alpha / float32(weight.B.Shape.Dims[0])
	}
	return scale
}

func (r *Runner) newGraphBuilder() *tensor.Builder {
	builder := tensor.NewBuilder()
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

func (r *Runner) applyLoRAEmbeddingRows(name string, rows []uint32, base reference.Value) (reference.Value, error) {
	for _, loaded := range r.loraAdapters {
		weight, ok := loaded.adapter.Weights[name]
		if !ok || !weight.Embedding {
			continue
		}
		scale := loRAScale(loaded, weight)
		if scale == 0 {
			continue
		}
		rank := int(weight.A.Shape.Dims[0])
		width := int(weight.B.Shape.Dims[1])
		for token, row := range rows {
			if uint64(row) >= weight.A.Shape.Dims[1] {
				return reference.Value{}, fmt.Errorf("inference: LoRA embedding row %d is out of range", row)
			}
			a := weight.A.Data[int(row)*rank : (int(row)+1)*rank]
			for output := 0; output < width; output++ {
				var delta float32
				b := weight.B.Data[output*rank : (output+1)*rank]
				for index := 0; index < rank; index++ {
					delta += b[index] * a[index]
				}
				base.Data[token*width+output] += scale * delta
			}
		}
	}
	return base, nil
}

func (r *Runner) applyLoRALogits(name string, hidden, logits []float32) error {
	for _, loaded := range r.loraAdapters {
		weight, ok := loaded.adapter.Weights[name]
		if !ok || weight.Embedding {
			continue
		}
		scale := loRAScale(loaded, weight)
		if scale == 0 {
			continue
		}
		input, rank, output := len(hidden), int(weight.A.Shape.Dims[1]), len(logits)
		if int(weight.A.Shape.Dims[0]) != input || int(weight.B.Shape.Dims[0]) != rank || int(weight.B.Shape.Dims[1]) != output {
			return fmt.Errorf("inference: LoRA output tensor %q shape changed", name)
		}
		projected := make([]float32, rank)
		for row := 0; row < rank; row++ {
			for column, value := range hidden {
				projected[row] += weight.A.Data[row*input+column] * value
			}
		}
		for row := 0; row < output; row++ {
			var delta float32
			for column, value := range projected {
				delta += weight.B.Data[row*rank+column] * value
			}
			logits[row] += scale * delta
		}
	}
	return nil
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
		return errors.New("inference: runner is nil")
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
