package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"

	"llamacpp2go/internal/quant"
	"llamacpp2go/internal/tensor/dtype"
)

const (
	maxLegacyIMatrixEntries      = 1 << 20
	maxLegacyIMatrixNameBytes    = 1 << 20
	maxLegacyIMatrixDatasetBytes = 1 << 20
)

// ImportanceMatrix: normalized tensor-column importance weights.
type ImportanceMatrix struct {
	Entries    map[string][]float32
	Datasets   []string
	ChunkCount uint32
	ChunkSize  uint32
	Legacy     bool
}

// LoadImportanceMatrix: pinned GGUF or legacy imatrix loader.
func LoadImportanceMatrix(path string) (*ImportanceMatrix, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open importance matrix: %w", err)
	}
	var magic [4]byte
	_, readErr := io.ReadFull(file, magic[:])
	_ = file.Close()
	if readErr != nil {
		return nil, errors.New("importance matrix is truncated")
	}
	if string(magic[:]) == Magic {
		return loadGGUFImportanceMatrix(path)
	}
	return loadLegacyImportanceMatrix(path)
}

func loadGGUFImportanceMatrix(path string) (*ImportanceMatrix, error) {
	file, err := Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF importance matrix: %w", err)
	}
	defer file.Close()
	datasetsValue, datasetsOK := file.MetadataValue("imatrix.datasets")
	chunkCountValue, chunkCountOK := file.MetadataValue("imatrix.chunk_count")
	chunkSizeValue, chunkSizeOK := file.MetadataValue("imatrix.chunk_size")
	if !datasetsOK || !chunkCountOK || !chunkSizeOK ||
		datasetsValue.Type != ValueTypeArray || datasetsValue.ArrayType != ValueTypeString ||
		chunkCountValue.Type != ValueTypeUint32 || chunkSizeValue.Type != ValueTypeUint32 {
		return nil, errors.New("GGUF importance matrix metadata is incomplete")
	}
	datasets, datasetsStored := datasetsValue.Data.([]string)
	chunkCount, countStored := chunkCountValue.Data.(uint32)
	chunkSize, sizeStored := chunkSizeValue.Data.(uint32)
	if !datasetsStored || !countStored || !sizeStored {
		return nil, errors.New("GGUF importance matrix metadata storage is invalid")
	}
	type pair struct {
		sums, counts *TensorInfo
	}
	pairs := make(map[string]pair)
	for index := range file.Tensors {
		info := &file.Tensors[index]
		entry := pair{}
		var name string
		switch {
		case strings.HasSuffix(info.Name, ".in_sum2"):
			name = strings.TrimSuffix(info.Name, ".in_sum2")
			entry = pairs[name]
			entry.sums = info
		case strings.HasSuffix(info.Name, ".counts"):
			name = strings.TrimSuffix(info.Name, ".counts")
			entry = pairs[name]
			entry.counts = info
		default:
			continue
		}
		if name == "" {
			return nil, fmt.Errorf("importance tensor %q has empty entry name", info.Name)
		}
		pairs[name] = entry
	}
	if len(pairs) == 0 {
		return nil, errors.New("GGUF importance matrix has no entries")
	}
	result := &ImportanceMatrix{
		Entries: make(map[string][]float32, len(pairs)), Datasets: slices.Clone(datasets),
		ChunkCount: chunkCount, ChunkSize: chunkSize,
	}
	for name, pair := range pairs {
		if pair.sums == nil || pair.counts == nil {
			return nil, fmt.Errorf("importance entry %q has mismatched sums and counts", name)
		}
		sums, err := readImportanceTensor(file, *pair.sums)
		if err != nil {
			return nil, err
		}
		counts, err := readImportanceTensor(file, *pair.counts)
		if err != nil {
			return nil, err
		}
		if len(counts) == 0 || len(sums)%len(counts) != 0 {
			return nil, fmt.Errorf("importance entry %q shape is invalid", name)
		}
		width := len(sums) / len(counts)
		normalized := make([]float32, len(sums))
		for group, rawCount := range counts {
			if !finiteFloat32GGUF(rawCount) || rawCount < 0 {
				return nil, fmt.Errorf("importance entry %q count %d is invalid", name, group)
			}
			count := float32(math.Round(float64(rawCount)))
			for column := 0; column < width; column++ {
				index := group*width + column
				if !finiteFloat32GGUF(sums[index]) || sums[index] < 0 {
					return nil, fmt.Errorf("importance entry %q value %d is invalid", name, index)
				}
				if count > 0 {
					normalized[index] = sums[index] / count
				} else {
					normalized[index] = 1
				}
			}
		}
		result.Entries[name] = normalized
	}
	return result, nil
}

func readImportanceTensor(file *File, info TensorInfo) ([]float32, error) {
	if info.Type != dtype.F32 {
		return nil, fmt.Errorf("importance tensor %q is %s, want f32", info.Name, info.Type)
	}
	elements, err := tensorElements(info)
	if err != nil {
		return nil, err
	}
	if info.Size > uint64(math.MaxInt) {
		return nil, fmt.Errorf("importance tensor %q exceeds addressable memory", info.Name)
	}
	data := make([]byte, int(info.Size))
	if err := file.ReadTensorRange(info, 0, data); err != nil {
		return nil, err
	}
	values, err := quant.Dequantize(info.Type, data, elements)
	if err != nil {
		return nil, fmt.Errorf("decode importance tensor %q: %w", info.Name, err)
	}
	return values, nil
}

func loadLegacyImportanceMatrix(path string) (*ImportanceMatrix, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	readInt32 := func() (int32, error) {
		var value int32
		err := binary.Read(reader, binary.LittleEndian, &value)
		return value, err
	}
	entryCount, err := readInt32()
	if err != nil || entryCount < 1 || entryCount > maxLegacyIMatrixEntries {
		return nil, errors.New("legacy importance matrix entry count is invalid")
	}
	result := &ImportanceMatrix{Entries: make(map[string][]float32, entryCount), Legacy: true}
	for entryIndex := int32(0); entryIndex < entryCount; entryIndex++ {
		nameLength, err := readInt32()
		if err != nil || nameLength < 1 || nameLength > maxLegacyIMatrixNameBytes {
			return nil, fmt.Errorf("legacy importance entry %d name length is invalid", entryIndex)
		}
		nameBytes := make([]byte, int(nameLength))
		if _, err := io.ReadFull(reader, nameBytes); err != nil {
			return nil, fmt.Errorf("read legacy importance entry %d name: %w", entryIndex, err)
		}
		name := string(nameBytes)
		if _, duplicate := result.Entries[name]; duplicate {
			return nil, fmt.Errorf("legacy importance entry %q is duplicated", name)
		}
		calls, err := readInt32()
		if err != nil || calls < 0 {
			return nil, fmt.Errorf("legacy importance entry %q call count is invalid", name)
		}
		valueCount, err := readInt32()
		if err != nil || valueCount < 1 || uint64(valueCount) > uint64(math.MaxInt)/4 {
			return nil, fmt.Errorf("legacy importance entry %q value count is invalid", name)
		}
		values := make([]float32, int(valueCount))
		if err := binary.Read(reader, binary.LittleEndian, values); err != nil {
			return nil, fmt.Errorf("read legacy importance entry %q: %w", name, err)
		}
		for index, value := range values {
			if !finiteFloat32GGUF(value) || value < 0 {
				return nil, fmt.Errorf("legacy importance entry %q value %d is invalid", name, index)
			}
			if calls > 0 {
				values[index] = value / float32(calls)
			}
		}
		result.Entries[name] = values
	}
	if _, peekErr := reader.Peek(1); peekErr == nil {
		calls, trailingErr := readInt32()
		if trailingErr == nil && calls >= 0 {
			result.ChunkCount = uint32(calls)
			length, lengthErr := readInt32()
			if lengthErr == nil && length > 0 && length <= maxLegacyIMatrixDatasetBytes {
				dataset := make([]byte, int(length))
				if _, datasetErr := io.ReadFull(reader, dataset); datasetErr == nil {
					result.Datasets = []string{string(dataset)}
				}
			}
		}
	}
	return result, nil
}

func finiteFloat32GGUF(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
