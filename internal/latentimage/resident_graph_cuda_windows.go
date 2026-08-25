//go:build windows

package latentimage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func weightPayload(value safetensors.Tensor, storage dtype.Type) ([]byte, error) {
	elements := int(value.Elements())
	if storage == dtype.BF16 && value.DType == "BF16" {
		buffer := make([]byte, elements*2)
		_, err := io.ReadFull(value.Reader(), buffer)
		return buffer, err
	}
	reader, err := safetensors.F32Reader(value)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, elements*4)
	if _, err := io.ReadFull(reader, raw); err != nil || storage == dtype.F32 {
		return raw, err
	}
	result := make([]byte, elements*2)
	for index := range elements {
		item := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
		binary.LittleEndian.PutUint16(result[2*index:], dtype.Float32ToBF16(item))
	}
	return result, nil
}

func compileResidentWeights(
	ctx context.Context,
	session *graphruntime.ResidentSession,
	label, sourcePath string,
	inputs map[string]*tensor.Tensor,
	dynamic []*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (*graphruntime.ResidentProgram, error) {
	program, err := session.Compile(ctx, label, dynamic, outputs...)
	if err != nil || len(inputs) == 0 {
		return program, err
	}
	source, err := safetensors.OpenSource(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("%s: open weights: %w", label, err)
	}
	defer source.Close()
	for name, node := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		value, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("%s: missing tensor %s", label, name)
		}
		if err := session.Bind(ctx, program, sourcePath, name, node, func() ([]byte, error) {
			return weightPayload(value, node.Type)
		}); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", label, name, err)
		}
	}
	return program, nil
}

func compileResidentStatic(
	ctx context.Context,
	session *graphruntime.ResidentSession,
	label string,
	inputs map[*tensor.Tensor][]float32,
	dynamic []*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (*graphruntime.ResidentProgram, error) {
	program, err := session.Compile(ctx, label, dynamic, outputs...)
	if err != nil {
		return nil, err
	}
	for node, values := range inputs {
		if err := session.BindF32(ctx, program, "static:"+label, node, values); err != nil {
			return nil, errors.Join(fmt.Errorf("%s: tensor %s: %w", label, node.Name, err), session.Release(context.Background(), program))
		}
	}
	return program, nil
}
