package safetensors

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/tensor/dtype"
)

// Save writes tensors as a single-file safetensors artifact at path. Every
// tensor stores F32 little-endian in name-sorted order with sequential disjoint
// data_offsets; shapes[name] gives each tensor's dims and must match its element
// count. metadata, when non-empty, becomes the reserved "__metadata__" header
// entry. The output round-trips through OpenSource (and Load). This is the
// persistence counterpart to OpenSource: without it a trained Model cannot be
// checkpointed, so the training lane has no production path.
func Save(path string, tensors map[string][]float32, shapes map[string][]int, metadata map[string]string) error {
	return SaveTyped(path, tensors, shapes, nil, metadata)
}

// SaveTyped is Save with each tensor stored as types names it, F32 where it names none: a BF16 tensor holds each value's
// nearest BF16, ties to even.
func SaveTyped(path string, tensors map[string][]float32, shapes map[string][]int, types map[string]string, metadata map[string]string) error {
	names := slices.Sorted(maps.Keys(tensors))
	specs := make([]streamTensorSpec, len(names))
	payloadSizes := make(map[string]uint64, len(names))
	for index, name := range names {
		shape, ok := shapes[name]
		if !ok {
			return fmt.Errorf("safetensors: no shape for tensor %q", name)
		}
		ushape := make([]uint64, len(shape))
		for index, dim := range shape {
			if dim <= 0 {
				return fmt.Errorf("safetensors: tensor %q has non-positive dim in %v", name, shape)
			}
			ushape[index] = uint64(dim)
		}
		stored := cmp.Or(types[name], "F32")
		if stored != "F32" && stored != "BF16" {
			return fmt.Errorf("safetensors: tensor %q cannot be saved as %s", name, stored)
		}
		size, err := TensorBytes(stored, ushape)
		scalar, _ := DTypeBytes(stored)
		actual, valid := checked.Mul64(uint64(len(tensors[name])), scalar)
		if err != nil || !valid || actual != size {
			return fmt.Errorf("safetensors: tensor %q shape %v does not match %d elements", name, shape, len(tensors[name]))
		}
		specs[index] = streamTensorSpec{Name: name, DType: stored, Shape: ushape}
		payloadSizes[name] = size
	}
	parent := filepath.Dir(path)
	temporary, err := os.MkdirTemp(parent, "."+filepath.Base(path)+".")
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(temporary)
		}
	}()
	destination := filepath.Join(temporary, "artifact")
	writer, err := newStreamWriter(destination, streamPlan{
		Shards: []streamShardSpec{{Name: streamSingleShardName, Tensors: specs}}, Metadata: metadata,
	})
	if err != nil {
		return err
	}
	for index, name := range names {
		var reader io.Reader = newF32SliceReader(tensors[name])
		if specs[index].DType == "BF16" {
			reader = &bf16SliceReader{values: tensors[name], offset: binaryschema.Uint16Bytes}
		}
		if err := writer.writeTensor(name, payloadSizes[name], reader); err != nil {
			return err
		}
	}
	if err := writer.finalize(); err != nil {
		return err
	}
	if err := os.Rename(filepath.Join(destination, streamSingleShardName), path); err != nil {
		return err
	}
	if err := os.Remove(destination); err != nil {
		return err
	}
	if err := os.Remove(temporary); err != nil {
		return err
	}
	committed = true
	return nil
}

func writeHeader(writer io.Writer, headerBytes []byte) error {
	lengthLE := make([]byte, headerLengthPrefixBytes())
	binary.LittleEndian.PutUint64(lengthLE, uint64(len(headerBytes)))
	if _, err := writer.Write(lengthLE); err != nil {
		return err
	}
	if _, err := writer.Write(headerBytes); err != nil {
		return err
	}
	return nil
}

func headerLengthPrefixBytes() int { return binaryschema.Uint64Bytes }

func padHeaderBytes(data []byte) []byte {
	for len(data)%binaryschema.Uint64Bytes != 0 {
		data = append(data, ' ')
	}
	return data
}

type f32SliceReader struct {
	values  []float32
	index   int
	pending [binaryschema.Uint32Bytes]byte
	offset  int
}

func newF32SliceReader(values []float32) *f32SliceReader {
	reader := &f32SliceReader{values: values}
	reader.offset = len(reader.pending)
	return reader
}

// Read encodes the next F32 payload bytes without materializing a second tensor.
func (reader *f32SliceReader) Read(destination []byte) (int, error) {
	var start int
	if len(destination) == start {
		return start, nil
	}
	written := start
	for written < len(destination) {
		if reader.offset == len(reader.pending) {
			if reader.index == len(reader.values) {
				if written == start {
					return start, io.EOF
				}
				return written, nil
			}
			binary.LittleEndian.PutUint32(reader.pending[:], math.Float32bits(reader.values[reader.index]))
			reader.index++
			reader.offset = start
		}
		count := copy(destination[written:], reader.pending[reader.offset:])
		written += count
		reader.offset += count
	}
	return written, nil
}

var _ io.Reader = (*f32SliceReader)(nil)

// bf16SliceReader encodes F32 values as their nearest BF16, little-endian, as Read is asked for bytes.
type bf16SliceReader struct {
	values  []float32
	index   int
	pending [binaryschema.Uint16Bytes]byte
	offset  int
}

// Read encodes the next BF16 payload bytes without materializing a second tensor.
func (reader *bf16SliceReader) Read(destination []byte) (int, error) {
	written := 0
	for written < len(destination) {
		if reader.offset == len(reader.pending) {
			if reader.index == len(reader.values) {
				if written == 0 {
					return 0, io.EOF
				}
				return written, nil
			}
			binary.LittleEndian.PutUint16(reader.pending[:], dtype.Float32ToBF16(reader.values[reader.index]))
			reader.index++
			reader.offset = 0
		}
		count := copy(destination[written:], reader.pending[reader.offset:])
		written += count
		reader.offset += count
	}
	return written, nil
}
