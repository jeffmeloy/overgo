package statecodec

import (
	"encoding/binary"
	"errors"
	"math"

	"llamacpp2go/internal/checked"
)

var ErrLimit = errors.New("state codec size limit exceeded")
var ErrTruncated = errors.New("state codec input is truncated")
var ErrTrailing = errors.New("state codec input has trailing data")

const (
	uint32Bytes = 4
	uint64Bytes = 8
)

type Encoder struct {
	data  []byte
	limit uint64
	err   error
}

func NewEncoder(limit uint64) *Encoder {
	return &Encoder{limit: limit}
}

func NewEncoderCapacity(limit, capacity uint64) *Encoder {
	encoder := NewEncoder(limit)
	count, ok := checked.Int(capacity)
	if !ok || capacity > limit {
		encoder.err = ErrLimit
		return encoder
	}
	encoder.data = make([]byte, 0, count)
	return encoder
}

func (e *Encoder) Raw(value []byte) {
	if !e.grow(uint64(len(value))) {
		return
	}
	copy(e.data[len(e.data)-len(value):], value)
}

func (e *Encoder) U32(value uint32) {
	if !e.grow(uint32Bytes) {
		return
	}
	binary.LittleEndian.PutUint32(e.data[len(e.data)-uint32Bytes:], value)
}

func (e *Encoder) U64(value uint64) {
	if !e.grow(uint64Bytes) {
		return
	}
	binary.LittleEndian.PutUint64(e.data[len(e.data)-uint64Bytes:], value)
}

func (e *Encoder) F32(value float32) {
	e.U32(math.Float32bits(value))
}

func (e *Encoder) I32(value int32) {
	e.U32(uint32(value))
}

func (e *Encoder) F64(value float64) {
	e.U64(math.Float64bits(value))
}

func (e *Encoder) String32(value string) {
	if uint64(len(value)) > math.MaxUint32 {
		e.err = ErrLimit
		return
	}
	e.U32(uint32(len(value)))
	e.Raw([]byte(value))
}

func (e *Encoder) Data() ([]byte, error) {
	if e.err != nil {
		return nil, e.err
	}
	return e.data, nil
}

func (e *Encoder) grow(size uint64) bool {
	if e.err != nil {
		return false
	}
	total, ok := checked.Add64(uint64(len(e.data)), size)
	if !ok || total > e.limit {
		e.err = ErrLimit
		return false
	}
	count, ok := checked.Int(total)
	if !ok {
		e.err = ErrLimit
		return false
	}
	e.data = append(e.data, make([]byte, count-len(e.data))...)
	return true
}

type Decoder struct {
	data   []byte
	offset uint64
	err    error
}

func NewDecoder(data []byte, limit uint64) *Decoder {
	decoder := &Decoder{data: data}
	if uint64(len(data)) > limit {
		decoder.err = ErrLimit
	}
	return decoder
}

func (d *Decoder) Raw(size uint64) []byte {
	if d.err != nil {
		return nil
	}
	end, ok := checked.Add64(d.offset, size)
	if !ok || end > uint64(len(d.data)) {
		d.err = ErrTruncated
		return nil
	}
	startIndex, _ := checked.Int(d.offset)
	endIndex, _ := checked.Int(end)
	d.offset = end
	return d.data[startIndex:endIndex]
}

func (d *Decoder) U32() uint32 {
	data := d.Raw(uint32Bytes)
	if data == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(data)
}

func (d *Decoder) U64() uint64 {
	data := d.Raw(uint64Bytes)
	if data == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(data)
}

func (d *Decoder) F32() float32 {
	return math.Float32frombits(d.U32())
}

func (d *Decoder) I32() int32 {
	return int32(d.U32())
}

func (d *Decoder) F64() float64 {
	return math.Float64frombits(d.U64())
}

func (d *Decoder) String32(limit uint64) string {
	size := uint64(d.U32())
	if d.err != nil {
		return ""
	}
	if size > limit {
		d.err = ErrLimit
		return ""
	}
	return string(d.Raw(size))
}

func (d *Decoder) Remaining() uint64 {
	if d.offset >= uint64(len(d.data)) {
		return 0
	}
	return uint64(len(d.data)) - d.offset
}

func (d *Decoder) Done() error {
	if d.err != nil {
		return d.err
	}
	if d.offset != uint64(len(d.data)) {
		return ErrTrailing
	}
	return nil
}

func (d *Decoder) Err() error {
	return d.err
}
