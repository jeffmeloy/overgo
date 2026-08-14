package safetensors

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"overgo/internal/tensor/dtype"
)

const (
	f32PromotionBufferBytes = 4 << 10
	float16StorageBytes     = 2
	float32StorageBytes     = 4
)

// F32Reader: stream F32, F16, or BF16 payload as F32.
func F32Reader(tensor Tensor) (io.Reader, error) {
	return PromoteF32Reader(tensor.Reader(), tensor.DType)
}

// ReadF32 promotes one tensor into its final F32 slab without a tensor-sized
// byte copy.
func ReadF32(tensor Tensor) ([]float32, error) {
	elements := tensor.Elements()
	count := int(elements)
	if count < 0 || uint64(count) != elements {
		return nil, errors.New("safetensors: tensor is too large for host memory")
	}
	reader, err := F32Reader(tensor)
	if err != nil {
		return nil, err
	}
	values := make([]float32, count)
	var encoded [f32PromotionBufferBytes]byte
	for offset := 0; offset < count; {
		batch := min(count-offset, len(encoded)/float32StorageBytes)
		payload := encoded[:batch*float32StorageBytes]
		if _, err := io.ReadFull(reader, payload); err != nil {
			return nil, err
		}
		for index := range batch {
			values[offset+index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*float32StorageBytes:]))
		}
		offset += batch
	}
	return values, nil
}

// PromoteF32Reader: stream one floating storage type as F32.
func PromoteF32Reader(source io.Reader, dataType string) (io.Reader, error) {
	switch dataType {
	case "F32":
		return source, nil
	case "F16":
		return &f32Reader{source: source, convert: dtype.Float16ToFloat32}, nil
	case "BF16":
		return &f32Reader{source: source, convert: dtype.BF16ToFloat32}, nil
	default:
		return nil, fmt.Errorf("safetensors: cannot promote %s to F32", dataType)
	}
}

type f32Reader struct {
	source  io.Reader
	convert func(uint16) float32
	input   []byte
	output  []byte
	offset  int
}

func (r *f32Reader) Read(destination []byte) (int, error) {
	written := 0
	for len(destination) > 0 {
		if r.offset < len(r.output) {
			count := copy(destination, r.output[r.offset:])
			r.offset += count
			written += count
			destination = destination[count:]
			continue
		}
		if r.input == nil {
			r.input = make([]byte, f32PromotionBufferBytes)
		}
		count, err := r.source.Read(r.input)
		if count%float16StorageBytes != 0 {
			return written, errors.New("safetensors: 16-bit source returned a partial element")
		}
		if count > 0 {
			r.output = make([]byte, count*float32StorageBytes/float16StorageBytes)
			for index := 0; index < count/float16StorageBytes; index++ {
				value := r.convert(binary.LittleEndian.Uint16(r.input[index*float16StorageBytes:]))
				binary.LittleEndian.PutUint32(r.output[index*float32StorageBytes:], math.Float32bits(value))
			}
			r.offset = 0
			continue
		}
		if err != nil {
			if written > 0 && err == io.EOF {
				return written, nil
			}
			return written, err
		}
		return written, io.ErrNoProgress
	}
	return written, nil
}
