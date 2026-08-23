package safetensors

import (
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
)

// Save writes tensors as a single-file safetensors artifact at path. Every
// tensor stores F32 little-endian in name-sorted order with sequential disjoint
// data_offsets; shapes[name] gives each tensor's dims and must match its element
// count. metadata, when non-empty, becomes the reserved "__metadata__" header
// entry. The output round-trips through OpenSource (and Load). This is the
// persistence counterpart to OpenSource: without it a trained Model cannot be
// checkpointed, so the training lane has no production path.
func Save(path string, tensors map[string][]float32, shapes map[string][]int, metadata map[string]string) error {
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
		size, err := TensorBytes("F32", ushape)
		actual, valid := checked.Mul64(uint64(len(tensors[name])), binaryschema.Uint32Bytes)
		if err != nil || !valid || actual != size {
			return fmt.Errorf("safetensors: tensor %q shape %v does not match %d elements", name, shape, len(tensors[name]))
		}
		specs[index] = streamTensorSpec{Name: name, DType: "F32", Shape: ushape}
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
	for _, name := range names {
		reader := newF32SliceReader(tensors[name])
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
