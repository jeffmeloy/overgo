package gguf

import (
	"bytes"
	"errors"
	"fmt"
	"io"

	"overgo/internal/quant"
)

// RewriteTo writes the file with every tensor's values passed through
// transform and stored as target, the metadata naming target as the file
// type. Each tensor is read, decoded to F32, transformed and encoded as the
// writer reaches it, so memory holds one tensor at a time.
func (f *File) RewriteTo(destination io.Writer, target DType, transform func(TensorInfo, []float32) error) error {
	if f == nil || destination == nil || transform == nil {
		return errors.New("GGUF rewrite needs a file, a destination and a transform")
	}
	metadata, err := quantizedMetadata(f.Metadata, target, QuantizeOptions{})
	if err != nil {
		return err
	}
	tensors := make([]TensorData, len(f.Tensors))
	for index, tensor := range f.Tensors {
		shape := make([]uint64, tensor.Dimensions)
		copy(shape, tensor.Shape[:tensor.Dimensions])
		tensors[index] = TensorData{Name: tensor.Name, Shape: shape, Type: target,
			Data: &rewriteReader{file: f, tensor: tensor, target: target, transform: transform}}
	}
	return Write(destination, metadata, tensors, WriteOptions{Version: f.Version, Alignment: f.Alignment})
}

// rewriteReader yields one rewritten tensor, computed on its first read and
// released when its last byte is read.
type rewriteReader struct {
	file      *File
	tensor    TensorInfo
	target    DType
	transform func(TensorInfo, []float32) error
	encoded   *bytes.Reader
	done      bool
}

// Read yields the rewritten tensor's encoded bytes. The writer holds every
// tensor's reader until the file is written, so a reader that kept its bytes
// would hold the whole rewritten file in memory: it drops them at the last
// byte, and a read after that is the end.
func (reader *rewriteReader) Read(buffer []byte) (int, error) {
	if reader.done {
		return 0, io.EOF
	}
	if reader.encoded == nil {
		encoded, err := reader.rewrite()
		if err != nil {
			return 0, err
		}
		reader.encoded = bytes.NewReader(encoded)
	}
	count, err := reader.encoded.Read(buffer)
	if reader.encoded.Len() == 0 {
		reader.encoded, reader.done = nil, true
	}
	return count, err
}

func (reader *rewriteReader) rewrite() ([]byte, error) {
	elements, err := reader.tensor.ElementCount()
	if err != nil {
		return nil, err
	}
	raw := make([]byte, reader.tensor.Size)
	if err := reader.file.ReadTensorData(reader.tensor, raw); err != nil {
		return nil, err
	}
	values, err := quant.Dequantize(reader.tensor.Type, raw, elements)
	if err != nil {
		return nil, fmt.Errorf("tensor %q: %w", reader.tensor.Name, err)
	}
	if err := reader.transform(reader.tensor, values); err != nil {
		return nil, fmt.Errorf("tensor %q: %w", reader.tensor.Name, err)
	}
	return quant.Quantize(reader.target, values)
}
