//go:build windows

// Resident device execution for the Krea2 text-fusion stream: every fusion weight
// uploads to the 4090D exactly once (BF16 for the rank-2 projections, F32 for the
// norms/biases and the layer projector), streamed from modelDir/transformer so no
// full-model F32 host copy is ever materialized. The selected hidden states are
// the only per-prompt host feed. This is the device bf16 fusion the golden serve
// runs; it consumes the encoder's 12 tapped hidden states and produces the
// [textSeq, Hidden] conditioning the denoiser text stream co-attends over. Mirrors
// NewResidentEncoder exactly (same upload/compile/execute discipline), reusing
// weightPayload/storageBytes from the denoiser sibling.
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

// ResidentFusion: the compiled fusion graph plus device-resident weights.
type ResidentFusion struct {
	Program     *FusionProgram
	WeightBytes uint64

	worker     *device.Worker
	exec       *executor.Executor
	compiled   *executor.CompiledGraph
	weightPtrs map[*tensor.Tensor]driver.DevicePtr
	allocs     []driver.DevicePtr
	ctx        context.Context
}

// NewResidentFusion uploads every fusion weight resident (streamed from
// modelDir/transformer) and compiles the fusion graph. The caller owns Close.
func NewResidentFusion(program *FusionProgram, modelDir string, ordinal int) (rf *ResidentFusion, err error) {
	if program == nil {
		return nil, errors.New("resident fusion: program is nil")
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident fusion: device: %w", err)
	}
	exec, err := executor.NewWithWorker(worker)
	if err != nil {
		worker.Close()
		return nil, fmt.Errorf("resident fusion: executor: %w", err)
	}
	rf = &ResidentFusion{
		Program:    program,
		worker:     worker,
		exec:       exec,
		weightPtrs: make(map[*tensor.Tensor]driver.DevicePtr, len(program.weightInputs)),
		ctx:        context.Background(),
	}
	defer func() {
		if err != nil {
			rf.Close()
		}
	}()

	src, err := safetensors.OpenSource(modelDir + `\transformer`)
	if err != nil {
		return nil, fmt.Errorf("resident fusion: open transformer: %w", err)
	}
	defer src.Close()

	for name, node := range program.weightInputs {
		tt, ok := src.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("resident fusion: missing tensor %s", name)
		}
		payload, perr := weightPayload(tt, node.Type)
		if perr != nil {
			return nil, fmt.Errorf("resident fusion: %s: %w", name, perr)
		}
		elements, _ := node.Shape.Elements()
		want := int(elements) * storageBytes(node.Type)
		if len(payload) != want {
			return nil, fmt.Errorf("resident fusion: %s payload=%d want %d", name, len(payload), want)
		}
		var ptr driver.DevicePtr
		if derr := worker.Do(rf.ctx, func(state *device.State) error {
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
			return nil, fmt.Errorf("resident fusion: upload %s: %w", name, derr)
		}
		rf.allocs = append(rf.allocs, ptr)
		rf.weightPtrs[node] = ptr
		rf.WeightBytes += uint64(len(payload))
	}

	rf.compiled, err = executor.Compile(program.Fused)
	if err != nil {
		return nil, fmt.Errorf("resident fusion: compile: %w", err)
	}
	return rf, nil
}

// Fuse runs the fusion graph with resident weights over the selected hidden states
// encoderHidden ([textSeq*TextLayers*TextHidden], the SelectedHiddenStates.Data
// layout, token-major F32 host feed) and returns the fused conditioning
// [textSeq*Hidden] (token-major).
func (rf *ResidentFusion) Fuse(encoderHidden []float32) ([]float32, error) {
	p := rf.Program
	if want := p.TextSeq * p.T.TextLayers * p.T.TextHidden; len(encoderHidden) != want {
		return nil, fmt.Errorf("resident fuse: encoder hidden len=%d want %d", len(encoderHidden), want)
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{
		p.InEncoder: {Shape: p.InEncoder.Shape, Data: encoderHidden},
	}
	results, err := rf.exec.ExecuteCompiledWithDeviceFeeds(rf.ctx, rf.compiled, hostFeeds, rf.weightPtrs)
	if err != nil {
		return nil, fmt.Errorf("resident fuse: %w", err)
	}
	return p.assemble(results)
}

// Close releases every device allocation and the worker.
func (rf *ResidentFusion) Close() error {
	var errs []error
	if rf.exec != nil {
		errs = append(errs, rf.exec.Close())
		rf.exec = nil
	}
	if rf.worker != nil && len(rf.allocs) > 0 {
		errs = append(errs, rf.worker.Do(rf.ctx, func(state *device.State) error {
			var e []error
			for _, ptr := range rf.allocs {
				e = append(e, state.Driver.MemFree(ptr))
			}
			return errors.Join(e...)
		}))
		rf.allocs = nil
	}
	if rf.worker != nil {
		rf.worker.Close()
		rf.worker = nil
	}
	return errors.Join(errs...)
}
