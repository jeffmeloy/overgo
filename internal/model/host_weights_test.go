package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
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

func TestLoadHostRowsAndDotRows(t *testing.T) {
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
	scores, err := DotRows(context.Background(), file, file.Tensors[0], []float32{1, 1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantScores := []float32{3, 7, 11}
	for row, want := range wantScores {
		if scores[row] != want {
			t.Fatalf("score[%d] = %v, want %v", row, scores[row], want)
		}
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
	qB := value(4, 8)
	kNorm := value(4)
	ropeFactors := value(2)
	attentionQBias := value(8)
	attentionKBias := value(4)
	attentionVBias := value(4)
	attentionOutputBias := value(8)
	feedForwardGateBias := value(12)
	feedForwardUpBias := value(12)
	feedForwardDownBias := value(8)
	feedForwardActivationScale := value(12)
	layer := HostLayer{
		AttentionNorm:              value(8),
		AttentionQ:                 value(8, 8),
		AttentionQB:                &qB,
		AttentionK:                 value(8, 4),
		AttentionV:                 value(8, 4),
		AttentionOutput:            value(8, 8),
		AttentionQNorm:             &qNorm,
		AttentionKNorm:             &kNorm,
		RopeFactors:                &ropeFactors,
		AttentionQBias:             &attentionQBias,
		AttentionKBias:             &attentionKBias,
		AttentionVBias:             &attentionVBias,
		AttentionOutputBias:        &attentionOutputBias,
		FeedForwardNorm:            value(8),
		FeedForwardGate:            value(8, 12),
		FeedForwardUp:              value(8, 12),
		FeedForwardDown:            value(12, 8),
		FeedForwardGateBias:        &feedForwardGateBias,
		FeedForwardUpBias:          &feedForwardUpBias,
		FeedForwardDownBias:        &feedForwardDownBias,
		FeedForwardActivationScale: &feedForwardActivationScale,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 21 ||
		graph.AttentionQB == nil ||
		graph.AttentionQNorm == nil ||
		graph.RopeFactors == nil ||
		graph.AttentionOutputBias == nil ||
		graph.FeedForwardActivationScale == nil ||
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

func TestHostLayerGraphInputsPermitFusedBetaAlphaRecurrent(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	qkv := value(8, 8)
	conv := value(3, 8)
	dt := value(2)
	a := value(2)
	ba := value(8, 4)
	norm := value(2)
	output := value(4, 8)
	layer := HostLayer{
		AttentionNorm:   value(8),
		AttentionQKV:    &qkv,
		SSMConv1D:       &conv,
		SSMTimeStep:     &dt,
		SSMA:            &a,
		SSMBetaAlpha:    &ba,
		SSMNorm:         &norm,
		SSMOutput:       &output,
		FeedForwardNorm: value(8),
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 9 || graph.AttentionQKV == nil || graph.AttentionGate != nil ||
		graph.SSMBetaAlpha == nil || graph.SSMBeta != nil || graph.SSMAlpha != nil {
		t.Fatalf("unexpected recurrent graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitKimiKDA(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	q, k, v, output := value(8, 4), value(8, 4), value(8, 4), value(4, 8)
	queryConv, keyConv, valueConv := value(3, 1, 4, 1), value(3, 1, 4, 1), value(3, 1, 4, 1)
	forgetA, forgetB, beta := value(8, 2), value(2, 4), value(8, 2)
	a, dt, gateA, gateB, norm := value(1, 2, 1, 1), value(4), value(8, 2), value(2, 4), value(2)
	layer := HostLayer{
		AttentionNorm: value(8), AttentionQ: q, AttentionK: k, AttentionV: v, AttentionOutput: output,
		SSMQueryConv: &queryConv, SSMKeyConv: &keyConv, SSMValueConv: &valueConv,
		SSMForgetA: &forgetA, SSMForgetB: &forgetB, SSMBeta: &beta, SSMA: &a,
		SSMTimeStep: &dt, SSMOutputGateA: &gateA, SSMOutputGateB: &gateB, SSMNorm: &norm,
		FeedForwardNorm: value(8),
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 17 || graph.SSMQueryConv == nil || graph.SSMForgetA == nil ||
		graph.SSMOutputGateB == nil || graph.AttentionQ == nil || graph.AttentionOutput == nil {
		t.Fatalf("unexpected Kimi KDA graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitRWKV6Qwen2(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	w1, w2 := value(8, 15), value(3, 8, 5)
	lerpX, lerp := value(8, 1, 1), value(8, 1, 1, 5)
	decay, decayW1, decayW2 := value(8), value(8, 2), value(2, 8)
	key, val, receptance := value(8, 4), value(8, 4), value(8, 8)
	gate, output := value(8, 8), value(8, 8)
	layer := HostLayer{
		AttentionNorm: value(8), FeedForwardNorm: value(8),
		FeedForwardGate: value(8, 12), FeedForwardUp: value(8, 12), FeedForwardDown: value(12, 8),
		TimeMixW1: &w1, TimeMixW2: &w2, TimeMixLerpX: &lerpX, TimeMixLerpFused: &lerp,
		TimeMixDecay: &decay, TimeMixDecayW1: &decayW1, TimeMixDecayW2: &decayW2,
		TimeMixKey: &key, TimeMixValue: &val, TimeMixReceptance: &receptance,
		TimeMixGate: &gate, TimeMixOutput: &output,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 17 || graph.TimeMixW1 == nil || graph.TimeMixLerpFused == nil || graph.TimeMixOutput == nil {
		t.Fatalf("unexpected RWKV6-Qwen2 graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitRWKV6(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	attentionNorm, attentionBias := value(8), value(8)
	channelNorm, channelBias := value(8), value(8)
	w1, w2, lerpX, lerp := value(8, 15), value(3, 8, 5), value(8, 1, 1), value(8, 1, 1, 5)
	first, decay, decayW1, decayW2 := value(4, 2), value(8), value(8, 2), value(2, 8)
	key, val, receptance, gate := value(8, 8), value(8, 8), value(8, 8), value(8, 8)
	mixNorm, mixBias, output := value(8), value(8), value(8, 8)
	channelLerpK, channelLerpR := value(8, 1, 1), value(8, 1, 1)
	channelKey, channelValue, channelReceptance := value(8, 12), value(12, 8), value(8, 8)
	layer := HostLayer{
		AttentionNorm: attentionNorm, AttentionNormBias: &attentionBias,
		AttentionNorm2: &channelNorm, AttentionNorm2Bias: &channelBias,
		TimeMixW1: &w1, TimeMixW2: &w2, TimeMixLerpX: &lerpX, TimeMixLerpFused: &lerp,
		TimeMixFirst: &first, TimeMixDecay: &decay, TimeMixDecayW1: &decayW1, TimeMixDecayW2: &decayW2,
		TimeMixKey: &key, TimeMixValue: &val, TimeMixReceptance: &receptance, TimeMixGate: &gate,
		TimeMixLN: &mixNorm, TimeMixLNBias: &mixBias, TimeMixOutput: &output,
		ChannelMixLerpK: &channelLerpK, ChannelMixLerpR: &channelLerpR,
		ChannelMixKey: &channelKey, ChannelMixValue: &channelValue, ChannelMixReceptance: &channelReceptance,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 24 || graph.TimeMixFirst == nil || graph.TimeMixLN == nil || graph.ChannelMixReceptance == nil {
		t.Fatalf("unexpected RWKV6 graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitRWKV7(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	attentionNorm, attentionBias := value(8), value(8)
	channelNorm, channelBias := value(8), value(8)
	w0, w1, w2 := value(8), value(8, 3), value(3, 8)
	a0, a1, a2 := value(8), value(8, 2), value(2, 8)
	v0, v1, v2 := value(8), value(8, 2), value(2, 8)
	g1, g2 := value(8, 2), value(2, 8)
	kk, ka, rk := value(8), value(8), value(8)
	lerp := value(8, 1, 1, 6)
	key, val, receptance, output := value(8, 8), value(8, 8), value(8, 8), value(8, 8)
	mixNorm, mixBias := value(8), value(8)
	channelLerp, channelKey, channelValue := value(8, 1, 1), value(8, 12), value(12, 8)
	layer := HostLayer{
		AttentionNorm: attentionNorm, AttentionNormBias: &attentionBias,
		AttentionNorm2: &channelNorm, AttentionNorm2Bias: &channelBias,
		TimeMixW0: &w0, TimeMixW1: &w1, TimeMixW2: &w2,
		TimeMixA0: &a0, TimeMixA1: &a1, TimeMixA2: &a2,
		TimeMixV0: &v0, TimeMixV1: &v1, TimeMixV2: &v2,
		TimeMixG1: &g1, TimeMixG2: &g2, TimeMixKK: &kk, TimeMixKA: &ka, TimeMixRK: &rk,
		TimeMixLerpFused: &lerp, TimeMixKey: &key, TimeMixValue: &val,
		TimeMixReceptance: &receptance, TimeMixLN: &mixNorm, TimeMixLNBias: &mixBias,
		TimeMixOutput: &output, ChannelMixLerpK: &channelLerp,
		ChannelMixKey: &channelKey, ChannelMixValue: &channelValue,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 28 || graph.TimeMixW0 == nil || graph.TimeMixA2 == nil ||
		graph.TimeMixV2 == nil || graph.TimeMixG2 == nil || graph.TimeMixRK == nil ||
		graph.ChannelMixValue == nil {
		t.Fatalf("unexpected RWKV7 graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitMamba2(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	input := value(4, 28)
	conv := value(3, 16)
	convBias := value(16)
	dt := value(4)
	a := value(1, 4)
	d := value(1, 4)
	norm := value(4, 2)
	output := value(8, 4)
	layer := HostLayer{
		AttentionNorm: value(4), SSMInput: &input, SSMConv1D: &conv,
		SSMConv1DBias: &convBias, SSMTimeStep: &dt, SSMA: &a, SSMD: &d,
		SSMNorm: &norm, SSMOutput: &output,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 9 || graph.SSMInput == nil || graph.SSMNorm == nil ||
		graph.SSMX != nil || graph.SSMTimeStepWeight != nil {
		t.Fatalf("unexpected Mamba2 graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitFalconH1(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	ssmInput, conv, dt := value(4, 28), value(3, 16), value(4)
	a, d, ssmOutput := value(1, 4), value(1, 4), value(8, 4)
	layer := HostLayer{
		AttentionNorm: value(4), AttentionQ: value(4, 4), AttentionK: value(4, 2),
		AttentionV: value(4, 2), AttentionOutput: value(4, 4),
		FeedForwardNorm: value(4), FeedForwardGate: value(4, 6),
		FeedForwardUp: value(4, 6), FeedForwardDown: value(6, 4),
		SSMInput: &ssmInput, SSMConv1D: &conv, SSMTimeStep: &dt,
		SSMA: &a, SSMD: &d, SSMOutput: &ssmOutput,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 15 || graph.AttentionQ == nil || graph.AttentionOutput == nil ||
		graph.SSMInput == nil || graph.SSMConv1D == nil || graph.SSMOutput == nil ||
		graph.FeedForwardGate == nil || graph.FeedForwardDown == nil {
		t.Fatalf("unexpected Falcon-H1 graph inputs: graph=%+v feeds=%d", graph, len(feeds))
	}
}

func TestHostLayerGraphInputsPermitParallelDenseAndMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	value := func(shape ...uint64) reference.Value {
		tensorShape := tensor.MustShape(shape...)
		elements, _ := tensorShape.Elements()
		return reference.Value{Shape: tensorShape, Data: make([]float32, int(elements))}
	}
	expertNorm := value(8)
	router := value(8, 4)
	gateExperts := value(8, 12, 4)
	upExperts := value(8, 12, 4)
	downExperts := value(12, 8, 4)
	layer := HostLayer{
		AttentionNorm:          value(8),
		AttentionQ:             value(8, 8),
		AttentionK:             value(8, 4),
		AttentionV:             value(8, 4),
		AttentionOutput:        value(8, 8),
		FeedForwardNorm:        value(8),
		FeedForwardExpertNorm:  &expertNorm,
		FeedForwardGate:        value(8, 8),
		FeedForwardUp:          value(8, 8),
		FeedForwardDown:        value(8, 8),
		FeedForwardRouter:      &router,
		FeedForwardGateExperts: &gateExperts,
		FeedForwardUpExperts:   &upExperts,
		FeedForwardDownExperts: &downExperts,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if graph.FeedForwardGate == nil || graph.FeedForwardUp == nil || graph.FeedForwardDown == nil ||
		graph.FeedForwardExpertNorm == nil || graph.FeedForwardRouter == nil ||
		graph.FeedForwardGateExperts == nil || graph.FeedForwardUpExperts == nil ||
		graph.FeedForwardDownExperts == nil || len(feeds) != 14 {
		t.Fatalf("unexpected parallel dense/MoE graph inputs: graph=%+v feeds=%d", graph, len(feeds))
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
	attentionPostNormBias := value(8)
	feedForwardPostNorm := value(8)
	feedForwardPostNormBias := value(8)
	layer := HostLayer{
		AttentionQ:              value(8, 8),
		AttentionK:              value(8, 4),
		AttentionV:              value(8, 4),
		AttentionOutput:         value(8, 8),
		AttentionQNorm:          &qNorm,
		AttentionKNorm:          &kNorm,
		AttentionPostNorm:       &attentionPostNorm,
		AttentionPostNormBias:   &attentionPostNormBias,
		FeedForwardGate:         value(8, 12),
		FeedForwardUp:           value(8, 12),
		FeedForwardDown:         value(12, 8),
		FeedForwardPostNorm:     &feedForwardPostNorm,
		FeedForwardPostNormBias: &feedForwardPostNormBias,
	}
	graph, feeds, err := layer.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if graph.AttentionNorm != nil ||
		graph.FeedForwardNorm != nil ||
		graph.AttentionQNorm == nil ||
		graph.AttentionPostNorm == nil ||
		graph.AttentionPostNormBias == nil ||
		graph.FeedForwardPostNorm == nil ||
		graph.FeedForwardPostNormBias == nil ||
		len(feeds) != 13 {
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
