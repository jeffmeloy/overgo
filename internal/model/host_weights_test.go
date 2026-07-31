package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func TestLoadHostTensor(t *testing.T) {
	data := hostTensorFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	value, err := LoadHostTensor(context.Background(), file, file.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	if !value.Shape.Equal(tensor.MustShape(4)) {
		t.Fatalf("shape = %v", value.Shape.Slice())
	}
	for index, want := range []float32{1, 2, 3, 4} {
		if value.Data[index] != want {
			t.Fatalf("value[%d] = %v, want %v", index, value.Data[index], want)
		}
	}
}

func TestLoadHostRowsAndArgmaxDot(t *testing.T) {
	data := hostTableFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	value, err := LoadHostRows(context.Background(), file, file.Tensors[0], []uint32{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{5, 6, 1, 2}
	for index := range want {
		if value.Data[index] != want[index] {
			t.Fatalf("row value[%d] = %v, want %v", index, value.Data[index], want[index])
		}
	}
	row, score, err := ArgmaxDot(context.Background(), file, file.Tensors[0], []float32{1, 1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if row != 2 || score != 11 {
		t.Fatalf("argmax = row %d score %v, want row 2 score 11", row, score)
	}
}

func TestHostLayerGraphInputs(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	qNorm := value(4)
	kNorm := value(4)
	ropeFactors := value(2)
	attentionQBias := value(8)
	attentionKBias := value(4)
	attentionVBias := value(4)
	attentionOutputBias := value(8)
	feedForwardGateBias := value(12)
	feedForwardUpBias := value(12)
	feedForwardDownBias := value(8)
	layer := HostLayer{
		AttentionNorm:       value(8),
		AttentionQ:          value(8, 8),
		AttentionK:          value(8, 4),
		AttentionV:          value(8, 4),
		AttentionOutput:     value(8, 8),
		AttentionQNorm:      &qNorm,
		AttentionKNorm:      &kNorm,
		RopeFactors:         &ropeFactors,
		AttentionQBias:      &attentionQBias,
		AttentionKBias:      &attentionKBias,
		AttentionVBias:      &attentionVBias,
		AttentionOutputBias: &attentionOutputBias,
		FeedForwardNorm:     value(8),
		FeedForwardGate:     value(8, 12),
		FeedForwardUp:       value(8, 12),
		FeedForwardDown:     value(12, 8),
		FeedForwardGateBias: &feedForwardGateBias,
		FeedForwardUpBias:   &feedForwardUpBias,
		FeedForwardDownBias: &feedForwardDownBias,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 19 ||
		graph.AttentionQNorm == nil ||
		graph.RopeFactors == nil ||
		graph.AttentionOutputBias == nil ||
		graph.FeedForwardDownBias == nil ||
		graph.FeedForwardDown == nil {
		t.Fatalf("unexpected graph inputs or feed count: %d", len(feeds))
	}
}

func TestHostLayerGraphInputsPermitDenseFusedQKV(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	qkv := value(8, 24)
	qkvBias := value(24)
	layer := HostLayer{
		AttentionNorm:    value(8),
		AttentionQKV:     &qkv,
		AttentionQKVBias: &qkvBias,
		AttentionOutput:  value(8, 8),
		FeedForwardUp:    value(8, 12),
		FeedForwardDown:  value(12, 8),
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 6 || graph.AttentionQKV == nil || graph.AttentionQKVBias == nil ||
		graph.AttentionQ != nil || graph.AttentionOutput == nil {
		t.Fatalf("unexpected fused-QKV graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitPostNormalizedBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	qNorm := value(8)
	kNorm := value(4)
	attentionPostNorm := value(8)
	feedForwardPostNorm := value(8)
	layer := HostLayer{
		AttentionQ:          value(8, 8),
		AttentionK:          value(8, 4),
		AttentionV:          value(8, 4),
		AttentionOutput:     value(8, 8),
		AttentionQNorm:      &qNorm,
		AttentionKNorm:      &kNorm,
		AttentionPostNorm:   &attentionPostNorm,
		FeedForwardGate:     value(8, 12),
		FeedForwardUp:       value(8, 12),
		FeedForwardDown:     value(12, 8),
		FeedForwardPostNorm: &feedForwardPostNorm,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if graph.AttentionNorm != nil ||
		graph.FeedForwardNorm != nil ||
		graph.AttentionQNorm == nil ||
		graph.AttentionPostNorm == nil ||
		graph.FeedForwardPostNorm == nil ||
		len(feeds) != 11 {
		t.Fatalf("unexpected post-normalized graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitSequentialFFN(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	attentionNormBias := value(8)
	attentionOutputBias := value(8)
	feedForwardNormBias := value(8)
	feedForwardUpBias := value(12)
	feedForwardDownBias := value(8)
	layer := HostLayer{
		AttentionNorm:       value(8),
		AttentionNormBias:   &attentionNormBias,
		AttentionQ:          value(8, 8),
		AttentionK:          value(8, 4),
		AttentionV:          value(8, 4),
		AttentionOutput:     value(8, 8),
		AttentionOutputBias: &attentionOutputBias,
		FeedForwardNorm:     value(8),
		FeedForwardNormBias: &feedForwardNormBias,
		FeedForwardUp:       value(8, 12),
		FeedForwardUpBias:   &feedForwardUpBias,
		FeedForwardDown:     value(12, 8),
		FeedForwardDownBias: &feedForwardDownBias,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if graph.FeedForwardGate != nil ||
		graph.AttentionNormBias == nil ||
		graph.FeedForwardNormBias == nil ||
		graph.FeedForwardUpBias == nil ||
		graph.FeedForwardDownBias == nil ||
		len(feeds) != 13 {
		t.Fatalf("unexpected sequential FFN graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func hostTensorFixture(t *testing.T) []byte {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}
	_, _ = buffer.WriteString(gguf.Magic)
	write(uint32(gguf.CurrentVersion))
	write(uint64(1))
	write(uint64(0))
	writeString("weight")
	write(uint32(1))
	write(uint64(4))
	write(uint32(dtype.F32))
	write(uint64(0))
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	for value := float32(1); value <= 4; value++ {
		write(value)
	}
	_, _ = buffer.Write(make([]byte, 16))
	return buffer.Bytes()
}

func hostTableFixture(t *testing.T) []byte {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}
	_, _ = buffer.WriteString(gguf.Magic)
	write(uint32(gguf.CurrentVersion))
	write(uint64(1))
	write(uint64(0))
	writeString("table")
	write(uint32(2))
	write(uint64(2))
	write(uint64(3))
	write(uint32(dtype.F32))
	write(uint64(0))
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	for value := float32(1); value <= 6; value++ {
		write(value)
	}
	_, _ = buffer.Write(make([]byte, 8))
	return buffer.Bytes()
}
