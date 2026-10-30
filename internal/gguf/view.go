package gguf

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
)

// ViewTensor is one tensor of a view: what Write takes, with its stored bytes
// read when a read reaches them, so a view holds no payload of its own.
// Payload reads them in place from their start, a source file's bytes kept as
// they are; Open, in its place, streams them from their start on each call,
// for bytes that must be built (a converted dtype, a transform).
type ViewTensor struct {
	Name    string
	Shape   []uint64
	Type    DType
	Payload io.ReaderAt
	Open    func() (io.Reader, error)
}

// viewCacheBytes bounds what a view keeps of the tensors it built: a load reads
// each tensor once, front to back, so the ones read last are the ones read again
// by the next reads that reach them; a Qwen3.5-4B's largest built tensor, a
// linear-attention input projection with its value heads reordered, is 42 MB.
const viewCacheBytes = 1 << 30

// View parses, without writing it, the GGUF image Write would produce from
// metadata and tensors: the header Write lays out, then each tensor's bytes
// where Write puts them, read from the tensor when a read reaches them. A model
// served from the files it was converted from reads the bytes its conversion
// would have written, so a fix to the conversion reaches it at its next open.
// directory names where the tensors read from (ViewDirectory), and the file's
// Close closes closers, what they read through.
func View(metadata []Metadata, tensors []ViewTensor, options WriteOptions, directory string, closers ...io.Closer) (*File, error) {
	placeholders := make([]TensorData, len(tensors))
	for index, tensor := range tensors {
		if (tensor.Payload == nil) == (tensor.Open == nil) {
			return nil, fmt.Errorf("GGUF view: tensor %q needs one source of its bytes", tensor.Name)
		}
		placeholders[index] = TensorData{Name: tensor.Name, Shape: tensor.Shape, Type: tensor.Type, Data: bytes.NewReader(nil)}
	}
	var header bytes.Buffer
	writer := &countingWriter{destination: &header, schema: binaryschema.LittleEndian}
	prepared, alignment, err := writeHeader(writer, metadata, placeholders, options)
	if err != nil {
		return nil, err
	}
	source := &viewSource{header: header.Bytes(), size: uint64(header.Len())}
	for index, item := range prepared {
		padded, ok := checked.Align(item.info.Size, alignment)
		if !ok {
			return nil, fmt.Errorf("GGUF view: tensor %q size overflows", item.info.Name)
		}
		source.regions = append(source.regions, viewRegion{start: source.size, size: item.info.Size, tensor: tensors[index]})
		source.size += padded
	}
	file, err := Parse(source, source.size, Options{})
	if err != nil {
		return nil, err
	}
	file.closers, file.viewDirectory = append(file.closers, closers...), directory
	return file, nil
}

// viewRegion is one tensor's bytes in a view's image.
type viewRegion struct {
	start, size uint64
	tensor      ViewTensor
}

// viewSource serves a view's image: its header, each tensor's bytes from the
// tensor, and zeros for the padding between them.
type viewSource struct {
	header  []byte
	regions []viewRegion
	size    uint64

	mu    sync.Mutex
	built []builtTensor
}

// builtTensor is the bytes Open gave for one region, newest last.
type builtTensor struct {
	region int
	data   []byte
}

// ReadAt fills destination with the image from offset: header bytes, tensor
// bytes and padding zeros, as far as the image reaches.
func (s *viewSource) ReadAt(destination []byte, offset int64) (int, error) {
	if offset < 0 {
		return 0, errors.New("GGUF view: negative offset")
	}
	read := 0
	for read < len(destination) {
		at := uint64(offset) + uint64(read)
		if at >= s.size {
			return read, io.EOF
		}
		part := destination[read:]
		if at < uint64(len(s.header)) {
			read += copy(part, s.header[at:])
			continue
		}
		// The region at or before at: its bytes, or the zeros of its padding.
		index := max(0, regionAt(s.regions, at))
		region := s.regions[index]
		if at >= region.start+region.size {
			next := s.size
			if index+1 < len(s.regions) {
				next = s.regions[index+1].start
			}
			zeros := part[:min(uint64(len(part)), next-at)]
			clear(zeros)
			read += len(zeros)
			continue
		}
		within := at - region.start
		part = part[:min(uint64(len(part)), region.size-within)]
		if err := s.tensorBytes(index, within, part); err != nil {
			return read, fmt.Errorf("GGUF view: tensor %q: %w", region.tensor.Name, err)
		}
		read += len(part)
	}
	return read, nil
}

// regionAt is the index of the last region starting at or before at, or -1.
func regionAt(regions []viewRegion, at uint64) int {
	found, _ := slices.BinarySearchFunc(regions, at, func(region viewRegion, at uint64) int {
		switch {
		case region.start < at:
			return -1
		case region.start > at:
			return 1
		}
		return 0
	})
	if found < len(regions) && regions[found].start == at {
		return found
	}
	return found - 1
}

// tensorBytes fills part with region index's bytes from within: read in place,
// or from what Open gave, built once while it stays among the newest kept.
func (s *viewSource) tensorBytes(index int, within uint64, part []byte) error {
	region := s.regions[index]
	if region.tensor.Payload != nil {
		_, err := region.tensor.Payload.ReadAt(part, int64(within))
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	at := slices.IndexFunc(s.built, func(built builtTensor) bool { return built.region == index })
	if at < 0 {
		stream, err := region.tensor.Open()
		if err != nil {
			return err
		}
		data := make([]byte, region.size)
		if _, err := io.ReadFull(stream, data); err != nil {
			return err
		}
		kept := uint64(len(data))
		for _, built := range s.built {
			kept += uint64(len(built.data))
		}
		for kept > viewCacheBytes && len(s.built) > 0 {
			kept -= uint64(len(s.built[0].data))
			s.built = s.built[1:]
		}
		s.built = append(s.built, builtTensor{region: index, data: data})
		at = len(s.built) - 1
	}
	copy(part, s.built[at].data[within:])
	return nil
}
