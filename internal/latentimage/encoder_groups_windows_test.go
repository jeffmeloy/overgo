//go:build windows

package latentimage

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"maps"
	"math"
	"sync/atomic"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
)

// countedReader is a weight's stored rows that count the bytes read from them.
type countedReader struct {
	data []byte
	read atomic.Int64
}

func (r *countedReader) ReadAt(destination []byte, offset int64) (int, error) {
	n, err := bytes.NewReader(r.data).ReadAt(destination, offset)
	r.read.Add(int64(n))
	return n, err
}

// TestAnEncoderWeightTheCardCannotTurnIsReadOnTheHost holds the encoder's card decode to the groups it turns
// (decode-groups, SQA-455): of an encoder file's untransformed BF16 bindings, a weight held in int8 ConvRot rotated in
// groups of 256 is decoded on the card, one rotated in groups of 4096, a power of 4 comfyquant admits and the card's
// exact decode does not turn, is left to the catalog's host decode with none of its rows read, and a plain weight is
// left to the catalog. moss-first-run/do had sent every such ConvRot binding to the card, whose decode refused the wider
// group only after the binding had read its 50 MB of rows and reserved its place, failing the prompt.
func TestAnEncoderWeightTheCardCannotTurnIsReadOnTheHost(t *testing.T) {
	t.Parallel()
	tensors := map[string]safetensors.Tensor{}
	add := func(name, dataType string, shape []uint64, reader interface {
		ReadAt([]byte, int64) (int, error)
	}, size int) {
		tensors[name] = testcheck.Must(safetensors.NewTensor(name, dataType, shape, reader, 0, int64(size)))(t)
	}
	quantized := func(base string, rows, columns int) *countedReader {
		record := fmt.Appendf(nil, `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":%d}`, columns)
		values := &countedReader{data: make([]byte, rows*columns)}
		scales := make([]byte, 4*rows)
		for row := range rows {
			binary.LittleEndian.PutUint32(scales[4*row:], math.Float32bits(0.5))
		}
		add(base+".weight", "I8", []uint64{uint64(rows), uint64(columns)}, values, len(values.data))
		add(base+".weight_scale", "F32", []uint64{uint64(rows), 1}, bytes.NewReader(scales), len(scales))
		add(base+".comfy_quant", "U8", []uint64{uint64(len(record))}, bytes.NewReader(record), len(record))
		return values
	}
	turned, wide := quantized("turned", 2, 256), quantized("wide", 2, 4096)
	add("plain.weight", "BF16", []uint64{2}, bytes.NewReader([]byte{1, 2, 3, 4}), 4)
	bf16 := gguf.DTypeBF16.String()
	bindings := []hfgguf.TensorBinding{
		{Source: "turned.weight", Target: "blk.0.turned.weight", TargetStorage: bf16},
		{Source: "wide.weight", Target: "blk.0.wide.weight", TargetStorage: bf16},
		{Source: "plain.weight", Target: "blk.0.plain.weight", TargetStorage: bf16},
	}
	decoded := testcheck.Must(cardDecoded(&safetensors.Source{Tensors: tensors}, bindings))(t)
	if want := map[string]string{"blk.0.turned.weight": "turned.weight"}; !maps.Equal(decoded, want) {
		t.Errorf("the card decodes %v; want %v, the weight rotated in groups of 4096 and the plain one left to the host", decoded, want)
	}
	if read := wide.read.Load() + turned.read.Load(); read != 0 {
		t.Errorf("choosing the card's weights read %d bytes of their rows; want their records alone read", read)
	}
}
