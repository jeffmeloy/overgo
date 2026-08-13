//go:build windows

package latentimage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type ResidentDenoiser struct {
	Program     *DenoiserProgram
	WeightBytes uint64
	runtime     *residentRuntime
	graph       *residentGraph
}

func NewResidentDenoiser(
	ctx context.Context,
	program *DenoiserProgram,
	modelDir string,
	ordinal int,
) (denoiser *ResidentDenoiser, err error) {
	if program == nil {
		return nil, errors.New("resident denoiser: program is nil")
	}
	runtime, err := newResidentRuntime(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident denoiser: device: %w", err)
	}
	denoiser = &ResidentDenoiser{Program: program, runtime: runtime}
	defer func() {
		if err != nil {
			_ = denoiser.Close(ctx)
		}
	}()
	outputs := append(append([]*tensor.Tensor(nil), program.BlockOutputs...), program.Velocity)
	denoiser.graph, err = runtime.compile(
		ctx, "resident denoiser", filepath.Join(modelDir, "transformer"),
		program.weightInputs, outputs...,
	)
	if err != nil {
		return nil, err
	}
	denoiser.WeightBytes = denoiser.graph.bytes
	return denoiser, nil
}

func (r *ResidentDenoiser) Step(
	ctx context.Context,
	latentPatches, text, temb, tembMod []float32,
) (ForwardResult, error) {
	if r == nil || r.Program == nil {
		return ForwardResult{}, errors.New("resident denoiser: unavailable")
	}
	program := r.Program
	for _, check := range []struct {
		name      string
		got, want int
	}{
		{"latent", len(latentPatches), program.ImgSeq * program.T.InChannels},
		{"text", len(text), program.TextSeq * program.T.Hidden},
		{"temb", len(temb), program.T.Hidden},
		{"tembMod", len(tembMod), 6 * program.T.Hidden},
	} {
		if check.got != check.want {
			return ForwardResult{}, fmt.Errorf("resident step: %s len=%d want %d", check.name, check.got, check.want)
		}
	}
	results, err := r.runtime.execute(ctx, r.graph, map[*tensor.Tensor]reference.Value{
		program.InLatent:  {Shape: program.InLatent.Shape, Data: latentPatches},
		program.InText:    {Shape: program.InText.Shape, Data: text},
		program.InTemb:    {Shape: program.InTemb.Shape, Data: temb},
		program.InTembMod: {Shape: program.InTembMod.Shape, Data: tembMod},
	})
	if err != nil {
		return ForwardResult{}, fmt.Errorf("resident step: %w", err)
	}
	full := results[program.Velocity].Data
	if len(full) != program.Seq*program.T.InChannels {
		return ForwardResult{}, fmt.Errorf(
			"resident step: velocity len=%d want %d", len(full), program.Seq*program.T.InChannels,
		)
	}
	result := ForwardResult{
		Velocity:    append([]float32(nil), full[program.TextSeq*program.T.InChannels:]...),
		BlockHidden: make([][]float32, len(program.BlockOutputs)),
	}
	for index, node := range program.BlockOutputs {
		result.BlockHidden[index] = results[node].Data
	}
	return result, nil
}

func (r *ResidentDenoiser) Close(ctx context.Context) error {
	if r == nil || r.runtime == nil {
		return nil
	}
	err := r.runtime.close(ctx)
	r.runtime = nil
	r.graph = nil
	return err
}

func storageBytes(dataType dtype.Type) int {
	if dataType == dtype.BF16 {
		return 2
	}
	return 4
}

// weightPayload converts checkpoint storage to graph input storage.
func weightPayload(value safetensors.Tensor, storage dtype.Type) ([]byte, error) {
	elements := 1
	for _, dimension := range value.Shape {
		elements *= int(dimension)
	}
	if storage == dtype.BF16 && value.DType == "BF16" {
		buffer := make([]byte, elements*2)
		if _, err := io.ReadFull(value.Reader(), buffer); err != nil {
			return nil, err
		}
		return buffer, nil
	}
	reader, err := safetensors.F32Reader(value)
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
	result := make([]byte, elements*2)
	for index := range elements {
		item := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
		binary.LittleEndian.PutUint16(result[2*index:], dtype.Float32ToBF16(item))
	}
	return result, nil
}
