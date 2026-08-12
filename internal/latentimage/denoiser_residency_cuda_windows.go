//go:build windows

// Resident device execution for the Krea2 step graph: every transformer weight
// uploads to the 4090D exactly once (BF16 for the rank-2 projections, F32 for
// norms/biases/tables), streamed from the sharded checkpoint so no full-model
// F32 host copy is ever materialized. Per step the graph runs through the
// generic executor with the resident weights as device feeds and the tiny
// per-step conditioning (latent, text, timestep) as host feeds. This is the
// 12.82B-param dual-stream forward on the device; the g2 per-step distribution
// oracle drives it.
package latentimage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// ResidentDenoiser: the compiled step graph plus device-resident weights.
type ResidentDenoiser struct {
	Program     *DenoiserProgram
	WeightBytes uint64

	worker     *device.Worker
	exec       *executor.Executor
	compiled   *executor.CompiledGraph
	weightPtrs map[*tensor.Tensor]driver.DevicePtr
	allocs     []driver.DevicePtr
	ctx        context.Context
}

// NewResidentDenoiser uploads every step-graph weight resident (streamed from
// modelDir/transformer) and compiles the graph. The caller owns Close.
func NewResidentDenoiser(program *DenoiserProgram, modelDir string, ordinal int) (rd *ResidentDenoiser, err error) {
	if program == nil {
		return nil, errors.New("resident denoiser: program is nil")
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident denoiser: device: %w", err)
	}
	exec, err := executor.NewWithWorker(worker)
	if err != nil {
		worker.Close()
		return nil, fmt.Errorf("resident denoiser: executor: %w", err)
	}
	rd = &ResidentDenoiser{
		Program:    program,
		worker:     worker,
		exec:       exec,
		weightPtrs: make(map[*tensor.Tensor]driver.DevicePtr, len(program.weightInputs)),
		ctx:        context.Background(),
	}
	defer func() {
		if err != nil {
			rd.Close()
		}
	}()

	src, err := safetensors.OpenSource(modelDir + `\transformer`)
	if err != nil {
		return nil, fmt.Errorf("resident denoiser: open transformer: %w", err)
	}
	defer src.Close()

	for name, node := range program.weightInputs {
		tt, ok := src.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("resident denoiser: missing tensor %s", name)
		}
		payload, perr := weightPayload(tt, node.Type)
		if perr != nil {
			return nil, fmt.Errorf("resident denoiser: %s: %w", name, perr)
		}
		elements, _ := node.Shape.Elements()
		want := int(elements) * storageBytes(node.Type)
		if len(payload) != want {
			return nil, fmt.Errorf("resident denoiser: %s payload=%d want %d", name, len(payload), want)
		}
		var ptr driver.DevicePtr
		if derr := worker.Do(rd.ctx, func(state *device.State) error {
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
			return nil, fmt.Errorf("resident denoiser: upload %s: %w", name, derr)
		}
		rd.allocs = append(rd.allocs, ptr)
		rd.weightPtrs[node] = ptr
		rd.WeightBytes += uint64(len(payload))
	}

	outputs := append(append([]*tensor.Tensor(nil), program.BlockOutputs...), program.Velocity)
	rd.compiled, err = executor.Compile(outputs...)
	if err != nil {
		return nil, fmt.Errorf("resident denoiser: compile: %w", err)
	}
	return rd, nil
}

// Step runs one denoise step with resident weights. latentPatches
// ([imgSeq*InChannels]), text ([textSeq*Hidden]), temb ([Hidden]) and tembMod
// ([6*Hidden]) are token-major F32 host feeds. Returns per-block hidden states
// and the image-token velocity.
func (rd *ResidentDenoiser) Step(latentPatches, text, temb, tembMod []float32) (ForwardResult, error) {
	p := rd.Program
	for _, chk := range []struct {
		name      string
		got, want int
	}{
		{"latent", len(latentPatches), p.ImgSeq * p.T.InChannels},
		{"text", len(text), p.TextSeq * p.T.Hidden},
		{"temb", len(temb), p.T.Hidden},
		{"tembMod", len(tembMod), 6 * p.T.Hidden},
	} {
		if chk.got != chk.want {
			return ForwardResult{}, fmt.Errorf("resident step: %s len=%d want %d", chk.name, chk.got, chk.want)
		}
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{
		p.InLatent:  {Shape: p.InLatent.Shape, Data: latentPatches},
		p.InText:    {Shape: p.InText.Shape, Data: text},
		p.InTemb:    {Shape: p.InTemb.Shape, Data: temb},
		p.InTembMod: {Shape: p.InTembMod.Shape, Data: tembMod},
	}
	results, err := rd.exec.ExecuteCompiledWithDeviceFeeds(rd.ctx, rd.compiled, hostFeeds, rd.weightPtrs)
	if err != nil {
		return ForwardResult{}, fmt.Errorf("resident step: %w", err)
	}
	full := results[p.Velocity].Data
	if len(full) != p.Seq*p.T.InChannels {
		return ForwardResult{}, fmt.Errorf("resident step: velocity len=%d want %d", len(full), p.Seq*p.T.InChannels)
	}
	res := ForwardResult{
		Velocity:    append([]float32(nil), full[p.TextSeq*p.T.InChannels:]...),
		BlockHidden: make([][]float32, len(p.BlockOutputs)),
	}
	for i, node := range p.BlockOutputs {
		res.BlockHidden[i] = results[node].Data
	}
	return res, nil
}

// Close releases every device allocation and the worker.
func (rd *ResidentDenoiser) Close() error {
	var errs []error
	if rd.exec != nil {
		errs = append(errs, rd.exec.Close())
		rd.exec = nil
	}
	if rd.worker != nil && len(rd.allocs) > 0 {
		errs = append(errs, rd.worker.Do(rd.ctx, func(state *device.State) error {
			var e []error
			for _, ptr := range rd.allocs {
				e = append(e, state.Driver.MemFree(ptr))
			}
			return errors.Join(e...)
		}))
		rd.allocs = nil
	}
	if rd.worker != nil {
		rd.worker.Close()
		rd.worker = nil
	}
	return errors.Join(errs...)
}

func storageBytes(t dtype.Type) int {
	if t == dtype.BF16 {
		return 2
	}
	return 4
}

// weightPayload streams tensor tt into a device payload of storage type. BF16
// node + BF16 source is a raw byte copy (no conversion); otherwise the tensor is
// promoted to F32 and (for a BF16 node) round-to-nearest-even encoded.
func weightPayload(tt safetensors.Tensor, storage dtype.Type) ([]byte, error) {
	elements := 1
	for _, d := range tt.Shape {
		elements *= int(d)
	}
	if storage == dtype.BF16 && tt.DType == "BF16" {
		buf := make([]byte, elements*2)
		if _, err := io.ReadFull(tt.Reader(), buf); err != nil {
			return nil, err
		}
		return buf, nil
	}
	reader, err := safetensors.F32Reader(tt)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, elements*4)
	if _, err := io.ReadFull(reader, raw); err != nil {
		return nil, err
	}
	if storage == dtype.F32 {
		return raw, nil
	}
	// F32 source -> BF16 node.
	out := make([]byte, elements*2)
	for i := 0; i < elements; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		binary.LittleEndian.PutUint16(out[2*i:], dtype.Float32ToBF16(f))
	}
	return out, nil
}
