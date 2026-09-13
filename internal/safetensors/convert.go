package safetensors

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"unsafe"

	"overgo/internal/binaryschema"
	"overgo/internal/tensor/dtype"
)

var f32PromotionBuffers = sync.Pool{
	New: func() any { return make([]byte, f32PromotionBufferBytes) },
}

const (
	f32PromotionBufferBytes = 1 << 20
)

// F32Reader: stream F32, F16, or BF16 payload as F32.
func F32Reader(tensor Tensor) (io.Reader, error) {
	return PromoteF32Reader(tensor.Reader(), tensor.DType)
}

// ReadF32 promotes one tensor into its final F32 slab without a tensor-sized
// byte copy.
func ReadF32(tensor Tensor) ([]float32, error) {
	switch tensor.DType {
	case "F32", "F16", "BF16":
	default:
		return nil, fmt.Errorf("safetensors: cannot promote %s to F32", tensor.DType)
	}
	elements := tensor.Elements()
	count := int(elements)
	if count < 0 || uint64(count) != elements {
		return nil, errors.New("safetensors: tensor is too large for host memory")
	}
	values := make([]float32, count)
	if tensor.DType == "F32" && nativeLittleEndian() {
		byteCount := tensor.Size()
		if byteCount < 0 || int64(int(byteCount)) != byteCount {
			return nil, errors.New("safetensors: tensor byte count exceeds host memory")
		}
		payload := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), int(byteCount))
		if _, err := io.ReadFull(tensor.Reader(), payload); err != nil {
			return nil, err
		}
		return values, nil
	}
	encoded := f32PromotionBuffers.Get().([]byte)
	defer f32PromotionBuffers.Put(encoded)
	switch tensor.DType {
	case "F32":
		for offset := 0; offset < count; {
			batch := min(count-offset, len(encoded)/binaryschema.Uint32Bytes)
			payload := encoded[:batch*binaryschema.Uint32Bytes]
			if _, err := tensor.ReadAt(payload, int64(offset)*binaryschema.Uint32Bytes); err != nil {
				return nil, err
			}
			for index := range batch {
				values[offset+index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*binaryschema.Uint32Bytes:]))
			}
			offset += batch
		}
	case "F16", "BF16":
		convert := dtype.Float16ToFloat32
		if tensor.DType == "BF16" {
			convert = dtype.BF16ToFloat32
		}
		for offset := 0; offset < count; {
			batch := min(count-offset, len(encoded)/binaryschema.Uint16Bytes)
			payload := encoded[:batch*binaryschema.Uint16Bytes]
			if _, err := tensor.ReadAt(payload, int64(offset)*binaryschema.Uint16Bytes); err != nil {
				return nil, err
			}
			for index := range batch {
				values[offset+index] = convert(binary.LittleEndian.Uint16(payload[index*binaryschema.Uint16Bytes:]))
			}
			offset += batch
		}
	}
	return values, nil
}

// F32Selection controls materialization and retained shape inventory.
type F32Selection struct {
	Keep         func(string) bool
	RetainShapes bool
}

// F32Catalog owns selected F32 values and optional host shapes.
type F32Catalog struct {
	Values map[string][]float32
	Shapes map[string][]int
}

// MaterializeF32 decodes selected tensors once.
func (s *Source) MaterializeF32(selection F32Selection) (F32Catalog, error) {
	if s == nil {
		return F32Catalog{}, errors.New("safetensors: nil source")
	}
	catalog := F32Catalog{Values: make(map[string][]float32)}
	if selection.Keep == nil {
		catalog.Values = make(map[string][]float32, len(s.Tensors))
	}
	if selection.RetainShapes {
		catalog.Shapes = make(map[string][]int)
		if selection.Keep == nil {
			catalog.Shapes = make(map[string][]int, len(s.Tensors))
		}
	}
	for _, name := range s.Names() {
		if selection.Keep != nil && !selection.Keep(name) {
			continue
		}
		tensor := s.Tensors[name]
		if selection.RetainShapes {
			shape, err := hostShape(name, tensor.Shape)
			if err != nil {
				return F32Catalog{}, err
			}
			catalog.Shapes[name] = shape
		}
		decoded, err := ReadF32(tensor)
		if err != nil {
			return F32Catalog{}, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		catalog.Values[name] = decoded
	}
	return catalog, nil
}

func nativeLittleEndian() bool {
	word := uint16(1)
	return *(*byte)(unsafe.Pointer(&word)) == 1
}

// ReadBF16 retains one BF16 tensor in native word storage.
func ReadBF16(tensor Tensor) ([]uint16, error) {
	if tensor.DType != "BF16" {
		return nil, fmt.Errorf("safetensors: cannot read %s as BF16", tensor.DType)
	}
	elements := tensor.Elements()
	count := int(elements)
	if count < 0 || uint64(count) != elements {
		return nil, errors.New("safetensors: tensor is too large for host memory")
	}
	values := make([]uint16, count)
	encoded := f32PromotionBuffers.Get().([]byte)
	defer f32PromotionBuffers.Put(encoded)
	for offset := 0; offset < count; {
		batch := min(count-offset, len(encoded)/binaryschema.Uint16Bytes)
		payload := encoded[:batch*binaryschema.Uint16Bytes]
		if _, err := tensor.ReadAt(payload, int64(offset)*binaryschema.Uint16Bytes); err != nil {
			return nil, err
		}
		for index := range batch {
			values[offset+index] = binary.LittleEndian.Uint16(payload[index*binaryschema.Uint16Bytes:])
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
		if count%binaryschema.Uint16Bytes != 0 {
			return written, errors.New("safetensors: 16-bit source returned a partial element")
		}
		if count > 0 {
			size := count * binaryschema.Uint32Bytes / binaryschema.Uint16Bytes
			if cap(r.output) < size {
				r.output = make([]byte, size)
			} else {
				r.output = r.output[:size]
			}
			for index := 0; index < count/binaryschema.Uint16Bytes; index++ {
				value := r.convert(binary.LittleEndian.Uint16(r.input[index*binaryschema.Uint16Bytes:]))
				binary.LittleEndian.PutUint32(r.output[index*binaryschema.Uint32Bytes:], math.Float32bits(value))
			}
			r.offset = 0
			continue
		}
		if err != nil {
			if written > 0 && errors.Is(err, io.EOF) {
				return written, nil
			}
			return written, err
		}
		return written, io.ErrNoProgress
	}
	return written, nil
}
