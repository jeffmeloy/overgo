package latentimage

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// countedReads is a tensor's payload in memory, counting the reads made of it.
type countedReads struct {
	data  []byte
	reads int
}

func (r *countedReads) ReadAt(destination []byte, offset int64) (int, error) {
	r.reads++
	return bytes.NewReader(r.data).ReadAt(destination, offset)
}

// TestAConvRotWeightBindsThroughTheResidentPath holds the resident path's read of a ComfyUI ConvRot weight
// (residentPayload, the payload ResidentSession.Bind calls only for a weight it does not hold): a weight the file holds
// quantized under its own name, which the loader read as a plain tensor and refused, binds as its decoded bfloat16 rows,
// none of its values or scales read until the payload is called, so a second bind the session's cache answers decodes
// nothing; and a weight whose scales are two a row, which the loader admitted and read the wrong scales of, is refused
// from the header with neither its values nor its scales read.
func TestAConvRotWeightBindsThroughTheResidentPath(t *testing.T) {
	t.Parallel()
	const rows, columns = 2, convRotGroup
	record := []byte(`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`)
	quantized := make([]byte, rows*columns)
	for index := range quantized {
		quantized[index] = byte(int8(index%7 - 3))
	}
	scales := []float32{0.5, 0.25, 4, 8}
	scaleBytes := make([]byte, 4*len(scales))
	for index, scale := range scales {
		binary.LittleEndian.PutUint32(scaleBytes[4*index:], math.Float32bits(scale))
	}
	reads, tensors := map[string]*countedReads{}, map[string]safetensors.Tensor{}
	add := func(name, dataType string, shape []uint64, data []byte) {
		reads[name] = &countedReads{data: data}
		tensors[name] = testcheck.Must(safetensors.NewTensor(name, dataType, shape, reads[name], 0, int64(len(data))))(t)
	}
	add("held.weight", "I8", []uint64{rows, columns}, quantized)
	add("held.weight_scale", "F32", []uint64{rows, 1}, scaleBytes[:4*rows])
	add("held.comfy_quant", "U8", []uint64{uint64(len(record))}, record)
	add("wide.weight", "I8", []uint64{rows, columns}, quantized)
	add("wide.weight_scale", "F32", []uint64{rows, 2}, scaleBytes)
	add("wide.comfy_quant", "U8", []uint64{uint64(len(record))}, record)
	source := &safetensors.Source{Tensors: tensors}
	payloadReads := func(base string) int { return reads[base+".weight"].reads + reads[base+".weight_scale"].reads }

	b := tensor.NewBuilder()
	held := b.Input("held.weight", dtype.BF16, tensor.MustShape(columns, rows))
	payload := testcheck.Must(residentPayload(source, held))(t)
	if read := payloadReads("held"); read != 0 {
		t.Fatalf("resolving the weight made %d reads of its values and scales; a bind the cache answers would decode it again", read)
	}
	want := make([]uint16, rows*columns)
	for row := range rows {
		decodeConvRotRow(want[row*columns:(row+1)*columns], quantized[row*columns:(row+1)*columns], scales[row])
	}
	if got := testcheck.Must(payload())(t); !bytes.Equal(got, bf16Payload(want, dtype.BF16)) {
		t.Fatal("the weight held under its own name did not bind as its decoded rows")
	}

	wide := b.Input("wide.weight", dtype.BF16, tensor.MustShape(columns, rows))
	if _, err := residentPayload(source, wide); err == nil || payloadReads("wide") != 0 {
		t.Fatalf("a weight with two scales a row resolved with error %v after %d reads of its values and scales; want it refused from the header, unread", err, payloadReads("wide"))
	}
}
