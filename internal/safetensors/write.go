package safetensors

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"slices"

	"overgo/internal/checked"
)

// headerAlignment pads the JSON header to an 8-byte boundary (trailing spaces,
// valid JSON whitespace) so the payload starts 8-aligned for external loaders;
// this reader does not require it.
const headerAlignment = 8

// Save writes tensors as a single-file safetensors artifact at path. Every
// tensor stores F32 little-endian in name-sorted order with sequential disjoint
// data_offsets; shapes[name] gives each tensor's dims and must match its element
// count. metadata, when non-empty, becomes the reserved "__metadata__" header
// entry. The output round-trips through OpenSource (and Load). This is the
// persistence counterpart to OpenSource: without it a trained Model cannot be
// checkpointed, so the training lane has no production path.
func Save(path string, tensors map[string][]float32, shapes map[string][]int, metadata map[string]string) error {
	names := slices.Sorted(maps.Keys(tensors))

	header := make(map[string]json.RawMessage, len(tensors)+1)
	if len(metadata) > 0 {
		raw, err := json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("safetensors: marshal metadata: %w", err)
		}
		header["__metadata__"] = raw
	}
	var offset uint64
	for _, name := range names {
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
		actual, valid := checked.Mul64(uint64(len(tensors[name])), dwordStorageBytes)
		if err != nil || !valid || actual != size {
			return fmt.Errorf("safetensors: tensor %q shape %v does not match %d elements", name, shape, len(tensors[name]))
		}
		entry, err := json.Marshal(tensorHeader{DType: "F32", Shape: ushape, DataOffsets: []uint64{offset, offset + size}})
		if err != nil {
			return fmt.Errorf("safetensors: marshal tensor %q header: %w", name, err)
		}
		header[name] = entry
		offset += size
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("safetensors: marshal header: %w", err)
	}
	for len(headerBytes)%headerAlignment != 0 {
		headerBytes = append(headerBytes, ' ')
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := writeArtifact(file, headerBytes, names, tensors); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

// writeArtifact emits the 8-byte header length, the header, then each tensor's
// f32 payload in the same name-sorted order the offsets were assigned.
func writeArtifact(file io.Writer, headerBytes []byte, names []string, tensors map[string][]float32) error {
	writer := bufio.NewWriter(file)
	var lengthLE [headerLengthBytes]byte
	binary.LittleEndian.PutUint64(lengthLE[:], uint64(len(headerBytes)))
	if _, err := writer.Write(lengthLE[:]); err != nil {
		return err
	}
	if _, err := writer.Write(headerBytes); err != nil {
		return err
	}
	var scratch [dwordStorageBytes]byte
	for _, name := range names {
		for _, value := range tensors[name] {
			binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(value))
			if _, err := writer.Write(scratch[:]); err != nil {
				return err
			}
		}
	}
	return writer.Flush()
}
