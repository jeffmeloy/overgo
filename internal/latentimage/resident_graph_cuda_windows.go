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

// bf16Payload is a decoded weight's bfloat16 bits in the storage a graph binds it in.
func bf16Payload(bits []uint16, storage dtype.Type) []byte {
	width := 2
	if storage != dtype.BF16 {
		width = 4
	}
	payload := make([]byte, len(bits)*width)
	for index, value := range bits {
		if width == 2 {
			binary.LittleEndian.PutUint16(payload[2*index:], value)
		} else {
			binary.LittleEndian.PutUint32(payload[4*index:], math.Float32bits(dtype.BF16ToFloat32(value)))
		}
	}
	return payload
}

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
	inputs tensor.WeightBindings,
	dynamic []*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (_ *graphruntime.ResidentProgram, err error) {
	program, err := session.Compile(ctx, label, dynamic, outputs...)
	if err != nil || len(inputs) == 0 {
		return program, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Release(context.WithoutCancel(ctx), program))
		}
	}()
	source, err := safetensors.OpenSource(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("%s: open weights: %w", label, err)
	}
	defer source.Close()
	for _, node := range inputs {
		if _, required := program.Graph.InputSlot(node); !required {
			continue
		}
		name := node.Name
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		value, ok := source.Tensors[name]
		if !ok {
			// A ComfyUI ConvRot file holds a weight in int8, or as its third of a fused qkv, decoded here at load.
			decoded, shape, quantized, err := convRotWeight(source, name)
			if err != nil {
				return nil, fmt.Errorf("%s: tensor %s: %w", label, name, err)
			}
			if !quantized {
				return nil, fmt.Errorf("%s: missing tensor %s", label, name)
			}
			if node.Shape.Rank != tensor.SingletonExtent && (len(shape) != int(node.Shape.Rank) || shape[0] != node.Shape.Dims[1] || shape[1] != node.Shape.Dims[0]) {
				return nil, fmt.Errorf("%s: tensor %s shape differs from graph", label, name)
			}
			if err := session.Bind(ctx, program, sourcePath, name, node, func() ([]byte, error) {
				return bf16Payload(decoded, node.Type), nil
			}); err != nil {
				return nil, fmt.Errorf("%s: tensor %s: %w", label, name, err)
			}
			continue
		}
		// Flat graph inputs preserve explicit linear views of source tables.
		// Matrix inputs preserve source axes as well as the payload extent.
		if node.Shape.Rank != tensor.SingletonExtent {
			if len(value.Shape) != int(node.Shape.Rank) {
				return nil, fmt.Errorf("%s: tensor %s rank differs from graph", label, name)
			}
			for axis, size := range value.Shape {
				if size != node.Shape.Dims[len(value.Shape)-1-axis] {
					return nil, fmt.Errorf("%s: tensor %s shape differs from graph", label, name)
				}
			}
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
