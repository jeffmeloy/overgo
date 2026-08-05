package gemma4convert

import (
	"encoding/binary"
	"errors"
	"io"
	"math"

	"llamacpp2go/internal/safetensors"
	"llamacpp2go/internal/tensor/dtype"
)

const (
	float32StorageBytes = 4
	bf16StorageBytes    = 2
	rgbChannelCount     = 3
	positionAxisCount   = 2

	fp8SignMask       = 0x80
	fp8ExponentShift  = 3
	fp8ExponentMask   = 0x0f
	fp8MantissaMask   = 0x07
	fp8ExponentBias   = 7
	fp8SubnormalScale = 1.0 / 512.0
)

type fp8BF16Reader struct {
	weight safetensors.Tensor
	scales []float32
	width  int
	row    int
	input  []byte
	buffer []byte
	offset int
}

func newFP8BF16Reader(weight safetensors.Tensor, scale safetensors.Tensor) (*fp8BF16Reader, error) {
	if weight.DType != "F8_E4M3" || len(weight.Shape) != 2 || scale.DType != "F32" ||
		len(scale.Shape) != 1 || scale.Shape[0] != weight.Shape[0] {
		return nil, errors.New("FP8 weight/scale shape is incompatible")
	}
	if weight.Shape[0] > math.MaxInt || weight.Shape[1] > math.MaxInt {
		return nil, errors.New("FP8 weight shape exceeds platform limits")
	}
	scales := make([]float32, int(scale.Shape[0]))
	encoded := make([]byte, len(scales)*float32StorageBytes)
	if _, err := io.ReadFull(scale.Reader(), encoded); err != nil {
		return nil, err
	}
	for index := range scales {
		scales[index] = math.Float32frombits(binary.LittleEndian.Uint32(
			encoded[index*float32StorageBytes:],
		))
	}
	width := int(weight.Shape[1])
	return &fp8BF16Reader{
		weight: weight, scales: scales, width: width,
		input:  make([]byte, width),
		buffer: make([]byte, width*bf16StorageBytes),
		offset: width * bf16StorageBytes,
	}, nil
}

func (r *fp8BF16Reader) Read(destination []byte) (int, error) {
	written := 0
	for len(destination) > 0 {
		if r.offset < len(r.buffer) {
			count := copy(destination, r.buffer[r.offset:])
			r.offset += count
			written += count
			destination = destination[count:]
			continue
		}
		if r.row >= len(r.scales) {
			if written > 0 {
				return written, nil
			}
			return 0, io.EOF
		}
		if _, err := r.weight.ReadAt(r.input, int64(r.row*r.width)); err != nil {
			return written, err
		}
		for column, value := range r.input {
			converted := fp8E4M3FN(value) * r.scales[r.row]
			binary.LittleEndian.PutUint16(
				r.buffer[column*bf16StorageBytes:], dtype.Float32ToBF16(converted),
			)
		}
		r.row++
		r.offset = 0
	}
	return written, nil
}

func fp8E4M3FN(encoded byte) float32 {
	sign := float32(1)
	if encoded&fp8SignMask != 0 {
		sign = -1
	}
	exponent := int((encoded >> fp8ExponentShift) & fp8ExponentMask)
	mantissa := int(encoded & fp8MantissaMask)
	if exponent == 0 {
		return sign * float32(mantissa) * fp8SubnormalScale
	}
	if exponent == fp8ExponentMask && mantissa == fp8MantissaMask {
		return float32(math.NaN())
	}
	return sign * float32(math.Ldexp(
		1+float64(mantissa)/(fp8MantissaMask+1), exponent-fp8ExponentBias,
	))
}

type positionReader struct {
	source safetensors.Tensor
	axis   int
	row    int
	rows   int
	width  int
	buffer []byte
	offset int
}

type patchPermutationReader struct {
	source safetensors.Tensor
	rows   int
	width  int
	row    int
	input  []byte
	buffer []byte
	offset int
}

func newPatchPermutationReader(source safetensors.Tensor) (*patchPermutationReader, error) {
	if source.DType != "BF16" || (len(source.Shape) != 1 && len(source.Shape) != 2) {
		return nil, errors.New("patch tensor must be rank-1 or rank-2 BF16")
	}
	width := source.Shape[len(source.Shape)-1]
	rows := uint64(1)
	if len(source.Shape) == 2 {
		rows = source.Shape[0]
	}
	if width%rgbChannelCount != 0 || width > math.MaxInt || rows > math.MaxInt {
		return nil, errors.New("patch tensor width is incompatible with RGB")
	}
	byteWidth := int(width) * bf16StorageBytes
	return &patchPermutationReader{
		source: source, rows: int(rows), width: int(width),
		input: make([]byte, byteWidth), buffer: make([]byte, byteWidth), offset: byteWidth,
	}, nil
}

func (r *patchPermutationReader) Read(destination []byte) (int, error) {
	written := 0
	for len(destination) > 0 {
		if r.offset < len(r.buffer) {
			count := copy(destination, r.buffer[r.offset:])
			r.offset += count
			written += count
			destination = destination[count:]
			continue
		}
		if r.row >= r.rows {
			if written > 0 {
				return written, nil
			}
			return 0, io.EOF
		}
		if _, err := r.source.ReadAt(r.input, int64(r.row*r.width*bf16StorageBytes)); err != nil {
			return written, err
		}
		area := r.width / rgbChannelCount
		for pixel := 0; pixel < area; pixel++ {
			for channel := 0; channel < rgbChannelCount; channel++ {
				sourceOffset := (pixel*rgbChannelCount + channel) * bf16StorageBytes
				destinationOffset := (channel*area + pixel) * bf16StorageBytes
				copy(
					r.buffer[destinationOffset:destinationOffset+bf16StorageBytes],
					r.input[sourceOffset:sourceOffset+bf16StorageBytes],
				)
			}
		}
		r.row++
		r.offset = 0
	}
	return written, nil
}

func newPositionReader(source safetensors.Tensor) (*positionReader, error) {
	if source.DType != "BF16" || len(source.Shape) != 3 || source.Shape[1] != positionAxisCount ||
		source.Shape[0] > math.MaxInt || source.Shape[2] > math.MaxInt {
		return nil, errors.New("vision position tensor must be BF16 [position,2,hidden]")
	}
	return &positionReader{
		source: source, rows: int(source.Shape[0]), width: int(source.Shape[2]) * bf16StorageBytes,
	}, nil
}

func (r *positionReader) Read(destination []byte) (int, error) {
	written := 0
	for len(destination) > 0 {
		if r.offset < len(r.buffer) {
			count := copy(destination, r.buffer[r.offset:])
			r.offset += count
			written += count
			destination = destination[count:]
			continue
		}
		if r.axis >= positionAxisCount {
			if written > 0 {
				return written, nil
			}
			return 0, io.EOF
		}
		r.buffer = make([]byte, r.width)
		sourceRow := r.row*positionAxisCount + r.axis
		if _, err := r.source.ReadAt(r.buffer, int64(sourceRow*r.width)); err != nil {
			return written, err
		}
		r.offset = 0
		r.row++
		if r.row == r.rows {
			r.row = 0
			r.axis++
		}
	}
	return written, nil
}
