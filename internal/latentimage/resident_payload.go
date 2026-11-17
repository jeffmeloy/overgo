package latentimage

import (
	"encoding/binary"
	"errors"
	"io"
	"math"

	"overgo/internal/comfyquant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// residentPayload is how a resident session reads a graph's weight input from a checkpoint: where the weight is stored
// and that its shape is the graph's are read from the file's header, before any payload, and the payload is read, and
// decoded where the file holds the weight quantized, only when the session's Bind calls for it, which a session that
// holds the weight already does not. A ComfyUI ConvRot file holds a weight in int8, under its own name or as its third
// of a fused qkv (comfyquant.Find); any other weight is read as the file stores it.
func residentPayload(source *safetensors.Source, node *tensor.Tensor) (func() ([]byte, error), error) {
	stored, quantized, err := comfyquant.Find(source, node.Name)
	if err != nil {
		return nil, err
	}
	if quantized {
		if shape := stored.Shape(); node.Shape.Rank != tensor.SingletonExtent && (int(node.Shape.Rank) != len(shape) || shape[0] != node.Shape.Dims[1] || shape[1] != node.Shape.Dims[0]) {
			return nil, errors.New("shape differs from graph")
		}
		return func() ([]byte, error) {
			decoded, err := stored.Decode()
			if err != nil {
				return nil, err
			}
			return bf16Payload(decoded, node.Type), nil
		}, nil
	}
	value, ok := source.Tensors[node.Name]
	if !ok {
		return nil, errors.New("missing from the checkpoint")
	}
	// Flat graph inputs preserve explicit linear views of source tables.
	// Matrix inputs preserve source axes as well as the payload extent.
	if node.Shape.Rank != tensor.SingletonExtent {
		if len(value.Shape) != int(node.Shape.Rank) {
			return nil, errors.New("rank differs from graph")
		}
		for axis, size := range value.Shape {
			if size != node.Shape.Dims[len(value.Shape)-1-axis] {
				return nil, errors.New("shape differs from graph")
			}
		}
	}
	return func() ([]byte, error) { return weightPayload(value, node.Type) }, nil
}

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
