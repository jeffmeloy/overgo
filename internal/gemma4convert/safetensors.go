package gemma4convert

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type safeTensorHeader struct {
	DType       string    `json:"dtype"`
	Shape       []uint64  `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
}

// Tensor: one safetensors payload range.
type Tensor struct {
	Name   string
	DType  string
	Shape  []uint64
	file   *os.File
	offset int64
	size   int64
}

func (t Tensor) Reader() *io.SectionReader {
	return io.NewSectionReader(t.file, t.offset, t.size)
}

func (t Tensor) ReadAt(destination []byte, offset int64) (int, error) {
	if offset < 0 || offset > t.size {
		return 0, errors.New("safetensors: tensor offset out of range")
	}
	if int64(len(destination)) > t.size-offset {
		return 0, io.ErrUnexpectedEOF
	}
	return t.file.ReadAt(destination, t.offset+offset)
}

// Source: open sharded safetensors catalog.
type Source struct {
	Tensors map[string]Tensor
	files   []*os.File
}

func OpenSource(directory string) (*Source, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read directory: %w", err)
	}
	paths := make([]string, 0)
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
			paths = append(paths, filepath.Join(directory, entry.Name()))
		}
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, errors.New("safetensors: no .safetensors files found")
	}
	source := &Source{Tensors: make(map[string]Tensor)}
	for _, path := range paths {
		if err := source.openShard(path); err != nil {
			_ = source.Close()
			return nil, err
		}
	}
	return source, nil
}

func (s *Source) Close() error {
	if s == nil {
		return nil
	}
	var result error
	for _, file := range s.files {
		result = errors.Join(result, file.Close())
	}
	s.files = nil
	return result
}

func (s *Source) openShard(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("safetensors: open %s: %w", filepath.Base(path), err)
	}
	s.files = append(s.files, file)
	info, err := file.Stat()
	if err != nil {
		return err
	}
	var sizeBytes [8]byte
	if _, err := io.ReadFull(file, sizeBytes[:]); err != nil {
		return fmt.Errorf("safetensors: read %s header size: %w", filepath.Base(path), err)
	}
	headerSize := binary.LittleEndian.Uint64(sizeBytes[:])
	if headerSize == 0 || headerSize > uint64(info.Size()-8) || headerSize > math.MaxInt {
		return fmt.Errorf("safetensors: %s header size is invalid", filepath.Base(path))
	}
	headerBytes := make([]byte, int(headerSize))
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return fmt.Errorf("safetensors: read %s header: %w", filepath.Base(path), err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return fmt.Errorf("safetensors: parse %s header: %w", filepath.Base(path), err)
	}
	payloadStart := int64(8 + headerSize)
	payloadSize := info.Size() - payloadStart
	for name, encoded := range raw {
		if name == "__metadata__" {
			continue
		}
		var header safeTensorHeader
		if err := json.Unmarshal(encoded, &header); err != nil {
			return fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		if _, duplicate := s.Tensors[name]; duplicate {
			return fmt.Errorf("safetensors: duplicate tensor %q", name)
		}
		start, end := header.DataOffsets[0], header.DataOffsets[1]
		if end < start || end > uint64(payloadSize) || start > math.MaxInt64 || end-start > math.MaxInt64 {
			return fmt.Errorf("safetensors: tensor %q range is invalid", name)
		}
		expected, err := safeTensorBytes(header.DType, header.Shape)
		if err != nil {
			return fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		if expected != end-start {
			return fmt.Errorf("safetensors: tensor %q has %d bytes, need %d", name, end-start, expected)
		}
		s.Tensors[name] = Tensor{
			Name: name, DType: header.DType, Shape: append([]uint64(nil), header.Shape...),
			file: file, offset: payloadStart + int64(start), size: int64(end - start),
		}
	}
	return nil
}

func safeTensorBytes(dataType string, shape []uint64) (uint64, error) {
	elementSize := uint64(0)
	switch dataType {
	case "F8_E4M3":
		elementSize = 1
	case "BF16", "F16":
		elementSize = 2
	case "F32", "I32":
		elementSize = 4
	case "F64", "I64":
		elementSize = 8
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dataType)
	}
	if len(shape) == 0 {
		return 0, errors.New("scalar tensors are unsupported")
	}
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension == 0 || elements > math.MaxUint64/dimension {
			return 0, errors.New("tensor shape is invalid")
		}
		elements *= dimension
	}
	if elements > math.MaxUint64/elementSize {
		return 0, errors.New("tensor byte size overflows")
	}
	return elements * elementSize, nil
}
