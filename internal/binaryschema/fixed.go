package binaryschema

import "encoding/binary"

const (
	Uint16Bytes = 2
	Uint32Bytes = 4
	Uint64Bytes = 8
)

// Fixed: byte order for fixed-width scalar storage.
type Fixed struct {
	Order binary.ByteOrder
}

var LittleEndian = Fixed{Order: binary.LittleEndian}

func (s Fixed) Uint16(data []byte) uint16 {
	return s.Order.Uint16(data)
}

func (s Fixed) Uint32(data []byte) uint32 {
	return s.Order.Uint32(data)
}

func (s Fixed) Uint64(data []byte) uint64 {
	return s.Order.Uint64(data)
}

func (s Fixed) PutUint16(data []byte, value uint16) {
	s.Order.PutUint16(data, value)
}

func (s Fixed) PutUint32(data []byte, value uint32) {
	s.Order.PutUint32(data, value)
}

func (s Fixed) PutUint64(data []byte, value uint64) {
	s.Order.PutUint64(data, value)
}
