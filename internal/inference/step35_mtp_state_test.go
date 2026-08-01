package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func TestStep35MTPSessionStateRoundTrip(t *testing.T) {
	runner := step35MTPStateRunner()
	session := step35MTPStateFixture(t, runner, true)
	data, err := runner.SaveStep35MTPSession(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadStep35MTPSession(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored Step3.5 MTP session differs:\n got %+v\nwant %+v", restored, session)
	}
	fresh := step35MTPStateFixture(t, runner, false)
	freshData, err := runner.SaveStep35MTPSession(fresh)
	if err != nil {
		t.Fatal(err)
	}
	freshRestored, err := runner.LoadStep35MTPSession(freshData)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(freshRestored, fresh) {
		t.Fatalf("fresh Step3.5 MTP session differs: got %+v want %+v", freshRestored, fresh)
	}
}

func TestHYV3MTPSessionStateRoundTrip(t *testing.T) {
	runner := step35MTPStateRunner()
	runner.spec.Architecture = "hy_v3"
	runner.weights.HYV3MTP = runner.weights.Step35MTP
	runner.weights.Step35MTP = nil
	session := step35MTPStateFixture(t, runner, true)
	data, err := runner.SaveHYV3MTPSession(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadHYV3MTPSession(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored HY-V3 MTP session differs:\n got %+v\nwant %+v", restored, session)
	}
}

func TestStep35MTPSessionStateRejectsCorruption(t *testing.T) {
	runner := step35MTPStateRunner()
	data, err := runner.SaveStep35MTPSession(step35MTPStateFixture(t, runner, true))
	if err != nil {
		t.Fatal(err)
	}
	for _, length := range []int{0, step35MTPStateHeader - 1, len(data) - 1} {
		if _, err := runner.LoadStep35MTPSession(data[:length]); err == nil {
			t.Fatalf("truncated Step3.5 MTP state length %d was accepted", length)
		}
	}
	trailing := append(append([]byte(nil), data...), 0)
	if _, err := runner.LoadStep35MTPSession(trailing); err == nil {
		t.Fatal("Step3.5 MTP state trailing data was accepted")
	}
	badHeads := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badHeads[80:], 1)
	if _, err := runner.LoadStep35MTPSession(badHeads); err == nil {
		t.Fatal("Step3.5 MTP state with incomplete heads was accepted")
	}
	zeroTarget := append([]byte(nil), data...)
	clear(zeroTarget[40:72])
	if _, err := runner.LoadStep35MTPSession(zeroTarget); err == nil {
		t.Fatal("Step3.5 MTP state without target binding was accepted")
	}
	other := step35MTPStateRunner()
	other.spec.Name = "different"
	if _, err := other.LoadStep35MTPSession(data); err == nil {
		t.Fatal("Step3.5 MTP state from another model was accepted")
	}
}

func step35MTPStateRunner() *Runner {
	spec := model.Spec{
		Architecture: "step35", Name: "state-test", BlockCount: 1, NextNPredictLayers: 2,
		ContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 1,
		LayerHeadCounts: []uint32{2, 4, 2}, LayerKVHeadCounts: []uint32{1, 2, 1},
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		SlidingWindow: 16, SlidingLayers: []bool{false, true, false},
		RMSNormEpsilon: 1e-6,
	}
	return &Runner{
		spec: spec,
		weights: model.Weights{
			Layers:    make([]model.LayerWeights, 1),
			Step35MTP: make([]model.Step35MTPWeights, 2),
		},
	}
}

func step35MTPStateFixture(t *testing.T, runner *Runner, active bool) *Step35MTPSession {
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
	return &Step35MTPSession{
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
