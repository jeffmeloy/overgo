package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

func TestQwen35MTPSessionStateRoundTrip(t *testing.T) {
	runner := qwen35MTPStateRunner()
	session := qwen35MTPStateFixture()
	data, err := runner.SaveQwen35MTPSession(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadQwen35MTPSession(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored Qwen3.5 MTP session differs:\n got %+v\nwant %+v", restored, session)
	}

	fresh := qwen35MTPStateFixture()
	fresh.Position = fresh.MTPStart
	fresh.Layer = LayerCache{}
	freshData, err := runner.SaveQwen35MTPSession(fresh)
	if err != nil {
		t.Fatal(err)
	}
	freshRestored, err := runner.LoadQwen35MTPSession(freshData)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(freshRestored, fresh) {
		t.Fatalf("fresh Qwen3.5 MTP session differs: got %+v want %+v", freshRestored, fresh)
	}
}

func TestQwen35MTPSessionStateRejectsCorruption(t *testing.T) {
	runner := qwen35MTPStateRunner()
	data, err := runner.SaveQwen35MTPSession(qwen35MTPStateFixture())
	if err != nil {
		t.Fatal(err)
	}
	for _, length := range []int{0, qwen35MTPStateHeader - 1, len(data) - 1} {
		if _, err := runner.LoadQwen35MTPSession(data[:length]); err == nil {
			t.Fatalf("truncated Qwen3.5 MTP state length %d was accepted", length)
		}
	}
	trailing := append(append([]byte(nil), data...), 0)
	if _, err := runner.LoadQwen35MTPSession(trailing); err == nil {
		t.Fatal("Qwen3.5 MTP state trailing data was accepted")
	}
	zeroTarget := append([]byte(nil), data...)
	clear(zeroTarget[40:72])
	if _, err := runner.LoadQwen35MTPSession(zeroTarget); err == nil {
		t.Fatal("Qwen3.5 MTP state without target binding was accepted")
	}
	badPosition := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badPosition[76:], binary.LittleEndian.Uint32(badPosition[72:]))
	if _, err := runner.LoadQwen35MTPSession(badPosition); err == nil {
		t.Fatal("Qwen3.5 MTP state with inconsistent layer position was accepted")
	}
	other := qwen35MTPStateRunner()
	other.spec.Name = "different"
	if _, err := other.LoadQwen35MTPSession(data); err == nil {
		t.Fatal("Qwen3.5 MTP state from another draft model was accepted")
	}
}

func qwen35MTPStateRunner() *Runner {
	return &Runner{
		spec: model.Spec{
			Architecture: "qwen35", Name: "state-test", BlockCount: 2,
			NextNPredictLayers: 1, ContextLength: 32, EmbeddingLength: 8,
			FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
			RopeSections: [4]int32{1, 1, 0, 0}, RMSNormEpsilon: 1e-6,
			SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2,
			SSMTimeStepRank: 2, SSMGroupCount: 1, FullAttentionInterval: 2,
		},
		weights: model.Weights{Qwen35MTP: &model.Qwen35MTPWeights{MTPOnly: true}},
	}
}

func qwen35MTPStateFixture() *Qwen35MTPSession {
	var targetModel [32]byte
	for index := range targetModel {
		targetModel[index] = byte(index + 1)
	}
	return &Qwen35MTPSession{
		TrunkCache: &KVCache{
			Tokens: 1, Position: 1,
			Layers: []LayerCache{
				{
					Key:   reference.Value{Shape: tensor.MustShape(2, 8), Data: make([]float32, 16)},
					Value: reference.Value{Shape: tensor.MustShape(2, 2, 2, 1), Data: make([]float32, 8)},
				},
				{
					Key:   reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: []float32{1, 2, 3, 4}},
					Value: reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: []float32{5, 6, 7, 8}},
				},
			},
		},
		Layer: LayerCache{
			Key:   reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: []float32{9, 10, 11, 12}},
			Value: reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: []float32{13, 14, 15, 16}},
		},
		PendingHidden: reference.Value{
			Shape: tensor.MustShape(8, 1), Data: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		MTPStart: 1, Position: 2, targetModel: targetModel,
	}
}
