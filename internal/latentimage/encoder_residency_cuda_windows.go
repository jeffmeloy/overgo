//go:build windows

// Resident device execution for the Qwen3-VL selected-layer text encoder: every
// tapped decoder layer's weights upload to the 4090D exactly once (BF16 for the
// rank-2 projections, F32 for the norms), streamed from modelDir/text_encoder so
// no full-model F32 host copy is ever materialized. The per-token embedding rows
// are the only per-prompt host feed. This is the device bf16 encoder the golden
// serve runs; it produces the 12 tapped hidden states the text-fusion stream
// consumes. Mirrors NewResidentDenoiser exactly (same upload/compile/execute
// discipline), reusing weightPayload/storageBytes from the denoiser sibling.
package latentimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// ResidentEncoder: the compiled encoder graph plus device-resident weights.
type ResidentEncoder struct {
	Program     *EncoderProgram
	WeightBytes uint64

	worker     *device.Worker
	exec       *executor.Executor
	compiled   *executor.CompiledGraph
	weightPtrs map[*tensor.Tensor]driver.DevicePtr
	allocs     []driver.DevicePtr
	ctx        context.Context
}

// NewResidentEncoder uploads every tapped-layer weight resident (streamed from
// modelDir/text_encoder) and compiles the selected-layer graph. The caller owns
// Close.
func NewResidentEncoder(program *EncoderProgram, modelDir string, ordinal int) (re *ResidentEncoder, err error) {
	if program == nil {
		return nil, errors.New("resident encoder: program is nil")
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident encoder: device: %w", err)
	}
	exec, err := executor.NewWithWorker(worker)
	if err != nil {
		worker.Close()
		return nil, fmt.Errorf("resident encoder: executor: %w", err)
	}
	re = &ResidentEncoder{
		Program:    program,
		worker:     worker,
		exec:       exec,
		weightPtrs: make(map[*tensor.Tensor]driver.DevicePtr, len(program.weightInputs)),
		ctx:        context.Background(),
	}
	defer func() {
		if err != nil {
			re.Close()
		}
	}()

	src, err := safetensors.OpenSource(modelDir + `\text_encoder`)
	if err != nil {
		return nil, fmt.Errorf("resident encoder: open text_encoder: %w", err)
	}
	defer src.Close()

	for name, node := range program.weightInputs {
		tt, ok := src.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("resident encoder: missing tensor %s", name)
		}
		payload, perr := weightPayload(tt, node.Type)
		if perr != nil {
			return nil, fmt.Errorf("resident encoder: %s: %w", name, perr)
		}
		elements, _ := node.Shape.Elements()
		want := int(elements) * storageBytes(node.Type)
		if len(payload) != want {
			return nil, fmt.Errorf("resident encoder: %s payload=%d want %d", name, len(payload), want)
		}
		var ptr driver.DevicePtr
		if derr := worker.Do(re.ctx, func(state *device.State) error {
			p, allocErr := state.Driver.MemAlloc(uint64(len(payload)))
			if allocErr != nil {
				return allocErr
			}
			if copyErr := state.Driver.MemcpyHtoD(p, payload); copyErr != nil {
				freeErr := state.Driver.MemFree(p)
				return errors.Join(copyErr, freeErr)
			}
			ptr = p
			return nil
		}); derr != nil {
			return nil, fmt.Errorf("resident encoder: upload %s: %w", name, derr)
		}
		re.allocs = append(re.allocs, ptr)
		re.weightPtrs[node] = ptr
		re.WeightBytes += uint64(len(payload))
	}

	re.compiled, err = executor.Compile(program.Selected...)
	if err != nil {
		return nil, fmt.Errorf("resident encoder: compile: %w", err)
	}
	return re, nil
}

// Encode runs the encoder with resident weights over the per-token embedding rows
// embedRows ([seq*Hidden], token-major F32 host feed) and returns the tapped
// hidden states in [Seq, LayerCount, Hidden] layout.
func (re *ResidentEncoder) Encode(embedRows []float32) (*SelectedHiddenStates, error) {
	p := re.Program
	if len(embedRows) != p.Seq*p.E.Hidden {
		return nil, fmt.Errorf("resident encode: embed len=%d want %d", len(embedRows), p.Seq*p.E.Hidden)
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{
		p.Embed: {Shape: p.Embed.Shape, Data: embedRows},
	}
	results, err := re.exec.ExecuteCompiledWithDeviceFeeds(re.ctx, re.compiled, hostFeeds, re.weightPtrs)
	if err != nil {
		return nil, fmt.Errorf("resident encode: %w", err)
	}
	return p.assemble(results)
}

// readEmbedRowsF32 gathers the per-token embedding rows for ids from
// modelDir/text_encoder as token-major F32 ([len(ids)*Hidden]) -- the small
// per-prompt host feed the resident encoder consumes. Wraps the f64 host reader.
func readEmbedRowsF32(modelDir string, e TextEncoderSpec, ids []int) ([]float32, error) {
	src, err := safetensors.OpenSource(modelDir + `\text_encoder`)
	if err != nil {
		return nil, fmt.Errorf("resident encoder embed: open text_encoder: %w", err)
	}
	defer src.Close()
	rows, err := readEmbedRows(src, e, ids)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(rows))
	for i, v := range rows {
		out[i] = float32(v)
	}
	return out, nil
}

// Close releases every device allocation and the worker.
func (re *ResidentEncoder) Close() error {
	var errs []error
	if re.exec != nil {
		errs = append(errs, re.exec.Close())
		re.exec = nil
	}
	if re.worker != nil && len(re.allocs) > 0 {
		errs = append(errs, re.worker.Do(re.ctx, func(state *device.State) error {
			var e []error
			for _, ptr := range re.allocs {
				e = append(e, state.Driver.MemFree(ptr))
			}
			return errors.Join(e...)
		}))
		re.allocs = nil
	}
	if re.worker != nil {
		re.worker.Close()
		re.worker = nil
	}
	return errors.Join(errs...)
}
