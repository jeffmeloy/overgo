package gemma4convert

import (
	"encoding/binary"
	"errors"
	"io"
	"math"

	"llamacpp2go/internal/tensor/dtype"
)

type fp8BF16Reader struct {
	weight Tensor
	scales []float32
	width  int
	row    int
	input  []byte
	buffer []byte
	offset int
}

func newFP8BF16Reader(weight Tensor, scale Tensor) (*fp8BF16Reader, error) {
	if weight.DType != "F8_E4M3" || len(weight.Shape) != 2 || scale.DType != "F32" ||
		len(scale.Shape) != 1 || scale.Shape[0] != weight.Shape[0] {
		return nil, errors.New("FP8 weight/scale shape is incompatible")
	}
	if weight.Shape[0] > math.MaxInt || weight.Shape[1] > math.MaxInt {
		return nil, errors.New("FP8 weight shape exceeds platform limits")
	}
	scales := make([]float32, int(scale.Shape[0]))
	encoded := make([]byte, len(scales)*4)
	if _, err := io.ReadFull(scale.Reader(), encoded); err != nil {
		return nil, err
	}
	for index := range scales {
		scales[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[index*4:]))
	}
	width := int(weight.Shape[1])
	return &fp8BF16Reader{
		weight: weight, scales: scales, width: width,
		input: make([]byte, width), buffer: make([]byte, width*2), offset: width * 2,
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
			binary.LittleEndian.PutUint16(r.buffer[column*2:], dtype.Float32ToBF16(converted))
		}
		r.row++
		r.offset = 0
	}
	return written, nil
}

func fp8E4M3FN(encoded byte) float32 {
	sign := float32(1)
	if encoded&0x80 != 0 {
		sign = -1
	}
	exponent := int((encoded >> 3) & 0xf)
	mantissa := int(encoded & 0x7)
	if exponent == 0 {
		return sign * float32(mantissa) * (1.0 / 512.0)
	}
	if exponent == 15 && mantissa == 7 {
		return float32(math.NaN())
	}
	return sign * float32(math.Ldexp(1+float64(mantissa)/8, exponent-7))
}

type bf16F32Reader struct {
	source io.Reader
	input  []byte
	output []byte
	offset int
}

func (r *bf16F32Reader) Read(destination []byte) (int, error) {
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
			r.input = make([]byte, 4096)
		}
		count, err := r.source.Read(r.input)
		if count%2 != 0 {
			return written, errors.New("BF16 source returned partial element")
		}
		if count > 0 {
			r.output = make([]byte, count*2)
			for index := 0; index < count/2; index++ {
				bits := uint32(binary.LittleEndian.Uint16(r.input[index*2:])) << 16
				binary.LittleEndian.PutUint32(r.output[index*4:], bits)
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

type positionReader struct {
	source Tensor
	axis   int
	row    int
	rows   int
	width  int
	buffer []byte
	offset int
}

type patchPermutationReader struct {
	source Tensor
	rows   int
	width  int
	row    int
	input  []byte
	buffer []byte
	offset int
}

func newPatchPermutationReader(source Tensor) (*patchPermutationReader, error) {
	if source.DType != "BF16" || (len(source.Shape) != 1 && len(source.Shape) != 2) {
		return nil, errors.New("patch tensor must be rank-1 or rank-2 BF16")
	}
	width := source.Shape[len(source.Shape)-1]
	rows := uint64(1)
	if len(source.Shape) == 2 {
		rows = source.Shape[0]
	}
	if width%3 != 0 || width > math.MaxInt || rows > math.MaxInt {
		return nil, errors.New("patch tensor width is incompatible with RGB")
	}
	byteWidth := int(width) * 2
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
		if _, err := r.source.ReadAt(r.input, int64(r.row*r.width*2)); err != nil {
			return written, err
		}
		area := r.width / 3
		for pixel := 0; pixel < area; pixel++ {
			for channel := 0; channel < 3; channel++ {
				sourceOffset := (pixel*3 + channel) * 2
				destinationOffset := (channel*area + pixel) * 2
				copy(r.buffer[destinationOffset:destinationOffset+2], r.input[sourceOffset:sourceOffset+2])
			}
		}
		r.row++
		r.offset = 0
	}
	return written, nil
}

func newPositionReader(source Tensor) (*positionReader, error) {
	if source.DType != "BF16" || len(source.Shape) != 3 || source.Shape[1] != 2 ||
		source.Shape[0] > math.MaxInt || source.Shape[2] > math.MaxInt {
		return nil, errors.New("vision position tensor must be BF16 [position,2,hidden]")
	}
	return &positionReader{source: source, rows: int(source.Shape[0]), width: int(source.Shape[2]) * 2}, nil
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
		if r.axis >= 2 {
			if written > 0 {
				return written, nil
			}
			return 0, io.EOF
		}
		r.buffer = make([]byte, r.width)
		sourceRow := r.row*2 + r.axis
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
