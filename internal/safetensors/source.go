package safetensors

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/binaryschema"
	"overgo/internal/strictjson"
)

const (
	defaultMaxHeaderBytes = 64 << 20
	defaultMaxIndexBytes  = 64 << 20
	defaultMaxShards      = 4096
	defaultMaxTensors     = 2_000_000
	defaultMaxRank        = 64
	defaultMaxNameBytes   = 16 << 10
	headerLengthBytes     = 8
	tensorRangeFieldCount = 2
	byteStorageBytes      = 1
	wordStorageBytes      = 2
	dwordStorageBytes     = 4
	qwordStorageBytes     = 8
)

// Limits: repository metadata bounds.
type Limits struct {
	MaxHeaderBytes uint64
	MaxIndexBytes  int64
	MaxShards      int
	MaxTensors     int
	MaxRank        int
	MaxNameBytes   int
}

// DefaultLimits: production repository bounds.
func DefaultLimits() Limits {
	return Limits{
		MaxHeaderBytes: defaultMaxHeaderBytes,
		MaxIndexBytes:  defaultMaxIndexBytes,
		MaxShards:      defaultMaxShards,
		MaxTensors:     defaultMaxTensors,
		MaxRank:        defaultMaxRank,
		MaxNameBytes:   defaultMaxNameBytes,
	}
}

type tensorHeader struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

type shardIndex struct {
	Metadata  map[string]json.RawMessage `json:"metadata"`
	WeightMap map[string]string          `json:"weight_map"`
}

// Tensor: one validated payload range.
type Tensor struct {
	Name  string
	DType string
	Shape []uint64
	Shard string

	reader io.ReaderAt
	offset int64
	size   int64
}

// NewTensor: validated tensor view over a random-access source.
func NewTensor(
	name string,
	dataType string,
	shape []uint64,
	reader io.ReaderAt,
	offset int64,
	size int64,
) (Tensor, error) {
	if reader == nil || offset < 0 || size < 0 || offset > math.MaxInt64-size {
		return Tensor{}, errors.New("safetensors: invalid tensor source")
	}
	expected, err := TensorBytes(dataType, shape)
	if err != nil {
		return Tensor{}, err
	}
	if expected > math.MaxInt64 || int64(expected) != size {
		return Tensor{}, fmt.Errorf("safetensors: tensor has %d bytes, need %d", size, expected)
	}
	return Tensor{
		Name: name, DType: dataType, Shape: slices.Clone(shape),
		reader: reader, offset: offset, size: size,
	}, nil
}

// Reader: bounded sequential payload view.
func (t Tensor) Reader() *io.SectionReader {
	return io.NewSectionReader(t.reader, t.offset, t.size)
}

// ReadAt: bounded payload read.
func (t Tensor) ReadAt(destination []byte, offset int64) (int, error) {
	if offset < 0 || offset > t.size {
		return 0, errors.New("safetensors: tensor offset out of range")
	}
	if int64(len(destination)) > t.size-offset {
		return 0, io.ErrUnexpectedEOF
	}
	return t.reader.ReadAt(destination, t.offset+offset)
}

// Size: payload bytes.
func (t Tensor) Size() int64 {
	return t.size
}

// Elements: tensor element count.
func (t Tensor) Elements() uint64 {
	elements := uint64(1)
	for _, dimension := range t.Shape {
		elements *= dimension
	}
	return elements
}

// Source: open sharded tensor catalog.
type Source struct {
	Tensors  map[string]Tensor
	Metadata map[string]map[string]string
	files    []*os.File
}

// OpenSource: open a repository with production bounds.
func OpenSource(directory string) (*Source, error) {
	return OpenSourceWithLimits(directory, DefaultLimits())
}

// OpenSourceWithLimits: open a bounded repository.
func OpenSourceWithLimits(directory string, limits Limits) (*Source, error) {
	if err := validateLimits(limits); err != nil {
		return nil, err
	}
	paths, weightMap, err := shardPaths(directory, limits)
	if err != nil {
		return nil, err
	}
	source := &Source{
		Tensors:  make(map[string]Tensor),
		Metadata: make(map[string]map[string]string),
	}
	for _, path := range paths {
		if err := source.openShard(directory, path, limits); err != nil {
			_ = source.Close()
			return nil, err
		}
	}
	if weightMap != nil {
		if err := validateIndexCatalog(source.Tensors, weightMap); err != nil {
			_ = source.Close()
			return nil, err
		}
	}
	return source, nil
}

// Names: sorted tensor names.
func (s *Source) Names() []string {
	if s == nil {
		return nil
	}
	names := make([]string, 0, len(s.Tensors))
	for name := range s.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Shards: sorted repository-relative shard names.
func (s *Source) Shards() []string {
	if s == nil {
		return nil
	}
	seen := make(map[string]struct{})
	for _, tensor := range s.Tensors {
		seen[tensor.Shard] = struct{}{}
	}
	shards := make([]string, 0, len(seen))
	for shard := range seen {
		shards = append(shards, shard)
	}
	sort.Strings(shards)
	return shards
}

// Close: release shard handles.
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

func validateLimits(limits Limits) error {
	if limits.MaxHeaderBytes == 0 || limits.MaxHeaderBytes > math.MaxInt64 ||
		limits.MaxIndexBytes <= 0 || limits.MaxShards <= 0 || limits.MaxTensors <= 0 ||
		limits.MaxRank <= 0 || limits.MaxNameBytes <= 0 {
		return errors.New("safetensors: invalid limits")
	}
	return nil
}

func shardPaths(directory string, limits Limits) ([]string, map[string]string, error) {
	indexPath := filepath.Join(directory, "model.safetensors.index.json")
	indexFile, err := os.Open(indexPath)
	if err == nil {
		defer indexFile.Close()
		var index shardIndex
		if err := strictjson.DecodeBounded(indexFile, limits.MaxIndexBytes, &index); err != nil {
			return nil, nil, fmt.Errorf("safetensors: parse shard index: %w", err)
		}
		if len(index.WeightMap) == 0 {
			return nil, nil, errors.New("safetensors: shard index has no weights")
		}
		shards := make(map[string]string)
		weightMap := make(map[string]string, len(index.WeightMap))
		for name, shard := range index.WeightMap {
			if name == "" || len(name) > limits.MaxNameBytes {
				return nil, nil, errors.New("safetensors: shard index has invalid tensor name")
			}
			clean, path, err := localShardPath(directory, shard)
			if err != nil {
				return nil, nil, err
			}
			shards[clean] = path
			weightMap[name] = clean
		}
		if len(shards) > limits.MaxShards {
			return nil, nil, fmt.Errorf("safetensors: shard count %d exceeds limit %d", len(shards), limits.MaxShards)
		}
		names := make([]string, 0, len(shards))
		for name := range shards {
			names = append(names, name)
		}
		sort.Strings(names)
		paths := make([]string, len(names))
		for index, name := range names {
			paths[index] = shards[name]
		}
		return paths, weightMap, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return nil, nil, fmt.Errorf("safetensors: open shard index: %w", err)
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: read directory: %w", err)
	}
	paths := make([]string, 0)
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
			paths = append(paths, filepath.Join(directory, entry.Name()))
		}
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, nil, errors.New("safetensors: no .safetensors files found")
	}
	if len(paths) > limits.MaxShards {
		return nil, nil, fmt.Errorf("safetensors: shard count %d exceeds limit %d", len(paths), limits.MaxShards)
	}
	return paths, nil, nil
}

func localShardPath(directory string, shard string) (string, string, error) {
	if shard == "" || filepath.IsAbs(shard) {
		return "", "", fmt.Errorf("safetensors: invalid shard path %q", shard)
	}
	clean := filepath.Clean(filepath.FromSlash(shard))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", "", fmt.Errorf("safetensors: shard path escapes repository: %q", shard)
	}
	return filepath.ToSlash(clean), filepath.Join(directory, clean), nil
}

func (s *Source) openShard(directory string, path string, limits Limits) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("safetensors: open %s: %w", filepath.Base(path), err)
	}
	s.files = append(s.files, file)
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("safetensors: stat %s: %w", filepath.Base(path), err)
	}
	if info.Size() < headerLengthBytes {
		return fmt.Errorf("safetensors: %s is shorter than its header prefix", filepath.Base(path))
	}
	var sizeBytes [headerLengthBytes]byte
	if _, err := io.ReadFull(file, sizeBytes[:]); err != nil {
		return fmt.Errorf("safetensors: read %s header size: %w", filepath.Base(path), err)
	}
	headerSize := binaryschema.LittleEndian.Uint64(sizeBytes[:])
	if headerSize == 0 || headerSize > uint64(info.Size()-headerLengthBytes) || headerSize > limits.MaxHeaderBytes {
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
	shard, err := filepath.Rel(directory, path)
	if err != nil {
		return fmt.Errorf("safetensors: resolve shard name: %w", err)
	}
	shard = filepath.ToSlash(shard)
	if metadata, ok := raw["__metadata__"]; ok {
		var values map[string]string
		if err := strictjson.DecodeBytes(metadata, &values); err != nil {
			return fmt.Errorf("safetensors: parse %s metadata: %w", shard, err)
		}
		s.Metadata[shard] = values
		delete(raw, "__metadata__")
	}
	if len(s.Tensors)+len(raw) > limits.MaxTensors {
		return fmt.Errorf("safetensors: tensor count exceeds limit %d", limits.MaxTensors)
	}
	payloadStart := int64(headerLengthBytes + headerSize)
	payloadSize := uint64(info.Size() - payloadStart)
	spans := make([]payloadSpan, 0, len(raw))
	for name, encoded := range raw {
		if name == "" || len(name) > limits.MaxNameBytes {
			return errors.New("safetensors: invalid tensor name")
		}
		var header tensorHeader
		if err := strictjson.DecodeBytes(encoded, &header); err != nil {
			return fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		if len(header.Shape) > limits.MaxRank || len(header.DataOffsets) != tensorRangeFieldCount {
			return fmt.Errorf("safetensors: tensor %q metadata is invalid", name)
		}
		if _, duplicate := s.Tensors[name]; duplicate {
			return fmt.Errorf("safetensors: duplicate tensor %q", name)
		}
		start, end := header.DataOffsets[0], header.DataOffsets[1]
		if end < start || end > payloadSize || start > math.MaxInt64 || end-start > math.MaxInt64 {
			return fmt.Errorf("safetensors: tensor %q range is invalid", name)
		}
		tensor, err := NewTensor(name, header.DType, header.Shape, file, payloadStart+int64(start), int64(end-start))
		if err != nil {
			return fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		tensor.Shard = shard
		s.Tensors[name] = tensor
		spans = append(spans, payloadSpan{name: name, start: start, end: end})
	}
	return validateDisjointSpans(shard, spans)
}

type payloadSpan struct {
	name       string
	start, end uint64
}

func validateDisjointSpans(shard string, spans []payloadSpan) error {
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start == spans[j].start {
			return spans[i].end < spans[j].end
		}
		return spans[i].start < spans[j].start
	})
	for index := 1; index < len(spans); index++ {
		if spans[index].start < spans[index-1].end {
			return fmt.Errorf("safetensors: %s tensor %q overlaps %q", shard, spans[index].name, spans[index-1].name)
		}
	}
	return nil
}

func validateIndexCatalog(tensors map[string]Tensor, weightMap map[string]string) error {
	if len(tensors) != len(weightMap) {
		return fmt.Errorf("safetensors: shard index describes %d tensors, files contain %d", len(weightMap), len(tensors))
	}
	for name, shard := range weightMap {
		tensor, ok := tensors[name]
		if !ok {
			return fmt.Errorf("safetensors: shard index tensor %q is missing", name)
		}
		if tensor.Shard != shard {
			return fmt.Errorf("safetensors: shard index tensor %q maps to %s, found in %s", name, shard, tensor.Shard)
		}
	}
	return nil
}

// TensorBytes: checked payload size.
func TensorBytes(dataType string, shape []uint64) (uint64, error) {
	elementSize, ok := DTypeBytes(dataType)
	if !ok {
		return 0, fmt.Errorf("unsupported dtype %q", dataType)
	}
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension != 0 && elements > math.MaxUint64/dimension {
			return 0, errors.New("tensor shape is invalid")
		}
		elements *= dimension
	}
	if elements > math.MaxUint64/elementSize {
		return 0, errors.New("tensor byte size overflows")
	}
	return elements * elementSize, nil
}

// DTypeBytes: storage width for supported Safetensors dtypes.
func DTypeBytes(dataType string) (uint64, bool) {
	switch strings.ToUpper(dataType) {
	case "BOOL", "U8", "I8", "F8_E4M3", "F8_E4M3FN", "F8_E5M2":
		return byteStorageBytes, true
	case "U16", "I16", "F16", "BF16":
		return wordStorageBytes, true
	case "U32", "I32", "F32":
		return dwordStorageBytes, true
	case "U64", "I64", "F64":
		return qwordStorageBytes, true
	default:
		return 0, false
	}
}
