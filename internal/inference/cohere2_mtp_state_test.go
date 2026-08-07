package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestCohere2MTPSessionStateRoundTrip(t *testing.T) {
	runner := cohere2MTPStateRunner()
	session := cohere2MTPStateFixture()
	data, err := runner.SaveCohere2MTPSession(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadCohere2MTPSession(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored Cohere2-MoE MTP session differs:\n got %+v\nwant %+v", restored, session)
	}
	fresh := cohere2MTPStateFixture()
	fresh.Position = fresh.MTPStart
	fresh.Layer = LayerCache{}
	freshData, err := runner.SaveCohere2MTPSession(fresh)
	if err != nil {
		t.Fatal(err)
	}
	freshRestored, err := runner.LoadCohere2MTPSession(freshData)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(freshRestored, fresh) {
		t.Fatalf("fresh Cohere2-MoE MTP session differs: got %+v want %+v", freshRestored, fresh)
	}
}

func TestCohere2MTPSessionStateRejectsCorruption(t *testing.T) {
	runner := cohere2MTPStateRunner()
	data, err := runner.SaveCohere2MTPSession(cohere2MTPStateFixture())
	if err != nil {
		t.Fatal(err)
	}
	for _, length := range []int{0, singleHeadMTPStateHeader - 1, len(data) - 1} {
		if _, err := runner.LoadCohere2MTPSession(data[:length]); err == nil {
			t.Fatalf("truncated Cohere2-MoE MTP state length %d was accepted", length)
		}
	}
	badPosition := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badPosition[76:], binary.LittleEndian.Uint32(badPosition[72:]))
	if _, err := runner.LoadCohere2MTPSession(badPosition); err == nil {
		t.Fatal("Cohere2-MoE MTP state with inconsistent position was accepted")
	}
	other := cohere2MTPStateRunner()
	other.spec.Name = "different"
	if _, err := other.LoadCohere2MTPSession(data); err == nil {
		t.Fatal("Cohere2-MoE MTP state from another model was accepted")
	}
}

func cohere2MTPStateRunner() *Runner {
	return &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "cohere2moe", Name: "state-test", BlockCount: 2,
		NextNPredictLayers: 1, ContextLength: 32, EmbeddingLength: 8,
		FeedForwardLength: 16,
		RMSNormEpsilon:    1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4},
	},
		weights: model.Weights{Cohere2MTP: &model.Cohere2MTPWeights{MTPOnly: true}}},
	}
}

func cohere2MTPStateFixture() *Cohere2MTPSession {
	var targetModel [32]byte
	for index := range targetModel {
		targetModel[index] = byte(index + 1)
	}
	layer := func(start float32) LayerCache {
		return LayerCache{
			Key:   reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: []float32{start, start + 1, start + 2, start + 3}},
			Value: reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: []float32{start + 4, start + 5, start + 6, start + 7}},
		}
	}
	return &Cohere2MTPSession{
		TrunkCache: &KVCache{Tokens: 1, Position: 1, Layers: []LayerCache{layer(1), layer(9)}},
		Layer:      layer(17),
		PendingHidden: reference.Value{
			Shape: tensor.MustShape(8, 1), Data: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		MTPStart: 1, Position: 2, targetModel: targetModel,
	}
}
