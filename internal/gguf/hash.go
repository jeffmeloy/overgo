package gguf

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"math/bits"
)

// HashOptions: selects llama-gguf-hash-compatible payload digests
type HashOptions struct {
	XXH64     bool
	SHA1      bool
	SHA256    bool
	UUID      bool
	PerTensor bool
}

// HashValues: contains selected digests; Unselected fields are empty
type HashValues struct {
	XXH64  string
	SHA1   string
	SHA256 string
	UUID   string
}

// TensorHash: identifies one tensor's raw payload digests
type TensorHash struct {
	Name   string
	Values HashValues
}

// HashResult: contains per-tensor and concatenated logical-model payload hashes
type HashResult struct {
	Tensors []TensorHash
	Model   HashValues
}

const hashReadBufferBytes = 1 << 20

var llamaCPPHashNamespace = [16]byte{
	0xef, 0x00, 0x12, 0x06, 0xda, 0xdc, 0x5f, 0x6d,
	0xa1, 0x5f, 0x33, 0x59, 0xe5, 0x77, 0xd4, 0xe5,
}

// Hash: computes digests over raw tensor payload bytes in logical directory
// order; Padding and GGUF metadata are excluded, matching
// llama-gguf-hash
func (f *File) Hash(options HashOptions) (HashResult, error) {
	if f == nil {
		return HashResult{}, errors.New("GGUF file is nil")
	}
	if !options.XXH64 && !options.SHA1 && !options.SHA256 && !options.UUID {
		return HashResult{}, errors.New("no GGUF hash algorithm selected")
	}
	model := newHashAccumulator(options)
	result := HashResult{}
	if options.PerTensor {
		result.Tensors = make([]TensorHash, 0, len(f.Tensors))
	}
	buffer := make([]byte, hashReadBufferBytes)
	for _, tensor := range f.Tensors {
		var perTensor *hashAccumulator
		if options.PerTensor {
			value := newHashAccumulator(HashOptions{
				XXH64:  options.XXH64,
				SHA1:   options.SHA1,
				SHA256: options.SHA256,
			})
			perTensor = &value
		}
		var offset uint64
		for offset < tensor.Size {
			count := min(uint64(len(buffer)), tensor.Size-offset)
			chunk := buffer[:count]
			if err := f.ReadTensorRange(tensor, offset, chunk); err != nil {
				return HashResult{}, fmt.Errorf("hash tensor %q: %w", tensor.Name, err)
			}
			model.write(chunk)
			if perTensor != nil {
				perTensor.write(chunk)
			}
			offset += count
		}
		if perTensor != nil {
			result.Tensors = append(result.Tensors, TensorHash{
				Name:   tensor.Name,
				Values: perTensor.values(false),
			})
		}
	}
	result.Model = model.values(options.UUID)
	return result, nil
}

type hashAccumulator struct {
	xxh64  *xxh64Digest
	sha1   hash.Hash
	sha256 hash.Hash
	uuid   hash.Hash
}

func newHashAccumulator(options HashOptions) hashAccumulator {
	var result hashAccumulator
	if options.XXH64 {
		digest := newXXH64()
		result.xxh64 = &digest
	}
	if options.SHA1 {
		result.sha1 = sha1.New()
	}
	if options.SHA256 {
		result.sha256 = sha256.New()
	}
	if options.UUID {
		result.uuid = sha1.New()
		_, _ = result.uuid.Write(llamaCPPHashNamespace[:])
	}
	return result
}

func (a *hashAccumulator) write(data []byte) {
	if a.xxh64 != nil {
		a.xxh64.Write(data)
	}
	if a.sha1 != nil {
		_, _ = a.sha1.Write(data)
	}
	if a.sha256 != nil {
		_, _ = a.sha256.Write(data)
	}
	if a.uuid != nil {
		_, _ = a.uuid.Write(data)
	}
}

func (a *hashAccumulator) values(includeUUID bool) HashValues {
	var result HashValues
	if a.xxh64 != nil {
		result.XXH64 = fmt.Sprintf("%016x", a.xxh64.Sum64())
	}
	if a.sha1 != nil {
		result.SHA1 = hex.EncodeToString(a.sha1.Sum(nil))
	}
	if a.sha256 != nil {
		result.SHA256 = hex.EncodeToString(a.sha256.Sum(nil))
	}
	if includeUUID && a.uuid != nil {
		sum := a.uuid.Sum(nil)
		var value [16]byte
		copy(value[:], sum)
		value[6] = value[6]&0x0f | 0x50
		value[8] = value[8]&0x3f | 0x80
		result.UUID = fmt.Sprintf(
			"%08x-%04x-%04x-%04x-%012x",
			binary.BigEndian.Uint32(value[0:4]),
			binary.BigEndian.Uint16(value[4:6]),
			binary.BigEndian.Uint16(value[6:8]),
			binary.BigEndian.Uint16(value[8:10]),
			uint64(value[10])<<40|
				uint64(value[11])<<32|
				uint64(value[12])<<24|
				uint64(value[13])<<16|
				uint64(value[14])<<8|
				uint64(value[15]),
		)
	}
	return result
}

const (
	xxhPrime1 = uint64(11400714785074694791)
	xxhPrime2 = uint64(14029467366897019727)
	xxhPrime3 = uint64(1609587929392839161)
	xxhPrime4 = uint64(9650029242287828579)
	xxhPrime5 = uint64(2870177450012600261)
)

type xxh64Digest struct {
	total uint64
	v1    uint64
	v2    uint64
	v3    uint64
	v4    uint64
	tail  [32]byte
	used  int
}

func newXXH64() xxh64Digest {
	first := xxhPrime1
	first += xxhPrime2
	fourth := uint64(0)
	fourth -= xxhPrime1
	return xxh64Digest{
		v1: first,
		v2: xxhPrime2,
		v3: 0,
		v4: fourth,
	}
}

func (d *xxh64Digest) Write(data []byte) {
	d.total += uint64(len(data))
	if d.used+len(data) < len(d.tail) {
		d.used += copy(d.tail[d.used:], data)
		return
	}
	if d.used != 0 {
		needed := len(d.tail) - d.used
		copy(d.tail[d.used:], data[:needed])
		d.consume(d.tail[:])
		data = data[needed:]
		d.used = 0
	}
	for len(data) >= len(d.tail) {
		d.consume(data[:len(d.tail)])
		data = data[len(d.tail):]
	}
	d.used = copy(d.tail[:], data)
}

func (d *xxh64Digest) consume(block []byte) {
	d.v1 = xxhRound(d.v1, binary.LittleEndian.Uint64(block[0:8]))
	d.v2 = xxhRound(d.v2, binary.LittleEndian.Uint64(block[8:16]))
	d.v3 = xxhRound(d.v3, binary.LittleEndian.Uint64(block[16:24]))
	d.v4 = xxhRound(d.v4, binary.LittleEndian.Uint64(block[24:32]))
}

func (d *xxh64Digest) Sum64() uint64 {
	var result uint64
	if d.total >= 32 {
		result = bits.RotateLeft64(d.v1, 1) +
			bits.RotateLeft64(d.v2, 7) +
			bits.RotateLeft64(d.v3, 12) +
			bits.RotateLeft64(d.v4, 18)
		result = xxhMergeRound(result, d.v1)
		result = xxhMergeRound(result, d.v2)
		result = xxhMergeRound(result, d.v3)
		result = xxhMergeRound(result, d.v4)
	} else {
		result = xxhPrime5
	}
	result += d.total
	tail := d.tail[:d.used]
	for len(tail) >= 8 {
		lane := xxhRound(0, binary.LittleEndian.Uint64(tail[:8]))
		result ^= lane
		result = bits.RotateLeft64(result, 27)*xxhPrime1 + xxhPrime4
		tail = tail[8:]
	}
	if len(tail) >= 4 {
		result ^= uint64(binary.LittleEndian.Uint32(tail[:4])) * xxhPrime1
		result = bits.RotateLeft64(result, 23)*xxhPrime2 + xxhPrime3
		tail = tail[4:]
	}
	for _, value := range tail {
		result ^= uint64(value) * xxhPrime5
		result = bits.RotateLeft64(result, 11) * xxhPrime1
	}
	result ^= result >> 33
	result *= xxhPrime2
	result ^= result >> 29
	result *= xxhPrime3
	result ^= result >> 32
	return result
}

func xxhRound(accumulator, input uint64) uint64 {
	accumulator += input * xxhPrime2
	accumulator = bits.RotateLeft64(accumulator, 31)
	return accumulator * xxhPrime1
}

func xxhMergeRound(accumulator, value uint64) uint64 {
	accumulator ^= xxhRound(0, value)
	return accumulator*xxhPrime1 + xxhPrime4
}
