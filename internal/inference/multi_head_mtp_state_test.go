package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestMultiHeadMTPSessionStateRoundTrip(t *testing.T) {
	runner := multiHeadMTPStateRunner()
	session := multiHeadMTPStateFixture(t, runner, true)
	data, err := runner.SaveMultiHeadMTPSession(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadMultiHeadMTPSession(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored multi-head MTP session differs:\n got %+v\nwant %+v", restored, session)
	}
	fresh := multiHeadMTPStateFixture(t, runner, false)
	freshData, err := runner.SaveMultiHeadMTPSession(fresh)
	if err != nil {
		t.Fatal(err)
	}
	freshRestored, err := runner.LoadMultiHeadMTPSession(freshData)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(freshRestored, fresh) {
		t.Fatalf("fresh multi-head MTP session differs: got %+v want %+v", freshRestored, fresh)
	}
}

func TestAlternateMultiHeadMTPSessionStateRoundTrip(t *testing.T) {
	runner := multiHeadMTPStateRunner()
	runner.spec.Architecture = "hy_v3"
	runner.weights.HYV3MTP = runner.weights.Step35MTP
	runner.weights.Step35MTP = nil
	runner = attachFixtureProgram(runner)
	session := multiHeadMTPStateFixture(t, runner, true)
	data, err := runner.SaveMultiHeadMTPSession(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadMultiHeadMTPSession(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored alternate multi-head MTP session differs:\n got %+v\nwant %+v", restored, session)
	}
}

func TestMultiHeadMTPSessionStateRejectsCorruption(t *testing.T) {
	runner := multiHeadMTPStateRunner()
	data, err := runner.SaveMultiHeadMTPSession(multiHeadMTPStateFixture(t, runner, true))
	if err != nil {
		t.Fatal(err)
	}
	for _, length := range []int{0, multiHeadMTPStateHeader - 1, len(data) - 1} {
		if _, err := runner.LoadMultiHeadMTPSession(data[:length]); err == nil {
			t.Fatalf("truncated multi-head MTP state length %d was accepted", length)
		}
	}
	trailing := append(append([]byte(nil), data...), 0)
	if _, err := runner.LoadMultiHeadMTPSession(trailing); err == nil {
		t.Fatal("multi-head MTP state trailing data was accepted")
	}
	badHeads := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badHeads[80:], 1)
	if _, err := runner.LoadMultiHeadMTPSession(badHeads); err == nil {
		t.Fatal("multi-head MTP state with incomplete heads was accepted")
	}
	zeroTarget := append([]byte(nil), data...)
	clear(zeroTarget[40:72])
	if _, err := runner.LoadMultiHeadMTPSession(zeroTarget); err == nil {
		t.Fatal("multi-head MTP state without target binding was accepted")
	}
	other := multiHeadMTPStateRunner()
	other.spec.Name = "different"
	if _, err := other.LoadMultiHeadMTPSession(data); err == nil {
		t.Fatal("multi-head MTP state from another model was accepted")
	}
}

func multiHeadMTPStateRunner() *Runner {
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "step35", Name: "state-test", BlockCount: 1, NextNPredictLayers: 2,
		ContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		LayerHeadCounts: []uint32{2, 4, 2}, LayerKVHeadCounts: []uint32{1, 2, 1},
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		SlidingWindow: 16, SlidingLayers: []bool{false, true, false}},
	}
	return fixtureRunner(spec, model.Weights{
		Layers:    make([]model.LayerWeights, 1),
		Step35MTP: make([]model.Step35MTPWeights, 2),
	})
}

func multiHeadMTPStateFixture(t *testing.T, runner *Runner, active bool) *MultiHeadMTPSession {
	t.Helper()
	signature, err := runner.sessionModelSignature()
	if err != nil {
		t.Fatal(err)
	}
	cacheValue := func(shape tensor.Shape, start float32) reference.Value {
		elements, shapeErr := shape.Elements()
		if shapeErr != nil {
			t.Fatal(shapeErr)
		}
		data := make([]float32, int(elements))
		for index := range data {
			data[index] = start + float32(index)
		}
		return reference.Value{Shape: shape, Data: data}
	}
	trunk := LayerCache{
		Key:   cacheValue(tensor.MustShape(4, 1, 2), 1),
		Value: cacheValue(tensor.MustShape(4, 1, 2), 10),
	}
	head0Tokens := uint64(2)
	position := uint32(2)
	var draftTokens []tokenizer.TokenID
	var draftHidden []reference.Value
	if active {
		head0Tokens = 3
		position = 3
		draftTokens = []tokenizer.TokenID{1}
		draftHidden = []reference.Value{{
			Shape: tensor.MustShape(8, 1), Data: []float32{9, 10, 11, 12, 13, 14, 15, 16},
		}}
	}
	return &MultiHeadMTPSession{
		TrunkCache: &KVCache{Layers: []LayerCache{trunk}, Tokens: 2, Position: 2},
		Heads: []LayerCache{
			{
				Key:   cacheValue(tensor.MustShape(4, 2, head0Tokens), 20),
				Value: cacheValue(tensor.MustShape(4, 2, head0Tokens), 30),
			},
			{
				Key:   cacheValue(tensor.MustShape(4, 1, 2), 40),
				Value: cacheValue(tensor.MustShape(4, 1, 2), 50),
			},
		},
		PendingHidden: reference.Value{
			Shape: tensor.MustShape(8, 1), Data: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		DraftTokens: draftTokens, DraftHidden: draftHidden,
		MTPStart: 2, Position: position, targetModel: signature,
	}
}
