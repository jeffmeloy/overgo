package inference

import (
	"reflect"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestT5SessionStateRoundTrip(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "t5", BlockCount: 1,
		ContextLength: 16, EmbeddingLength: 2}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 3, HeadCountKV: 1}, EncoderSpec: model.EncoderSpec{DecoderBlockCount: 1},
	}}}
	runner = attachFixtureProgram(runner)
	encoder, _ := reference.NewValue(tensor.MustShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8})
	key, _ := reference.NewValue(tensor.MustShape(2, 1, 2), make([]float32, 4))
	value, _ := reference.NewValue(tensor.MustShape(3, 1, 2), make([]float32, 6))
	crossKey, _ := reference.NewValue(tensor.MustShape(2, 1, 4), make([]float32, 8))
	crossValue, _ := reference.NewValue(tensor.MustShape(3, 1, 4), make([]float32, 12))
	session := &T5Session{Encoder: encoder, Cache: &KVCache{
		Layers: []LayerCache{{Key: key, Value: value, States: LayerStates{
			model.CacheStateCrossKey:   {Mode: CacheStateFixed, Value: crossKey},
			model.CacheStateCrossValue: {Mode: CacheStateFixed, Value: crossValue},
		}}}, Tokens: 2, Position: 2,
	}}
	payload, err := runner.SaveT5Session(session)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadT5Session(payload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored T5 session = %+v, want %+v", restored, session)
	}
	payload[len(payload)-1] ^= 0xff
	if _, err := runner.LoadT5Session(payload); err != nil {
		t.Fatalf("finite cache mutation should remain structurally valid: %v", err)
	}
}

func TestT5SessionStateRejectsBadPayload(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "t5", BlockCount: 1,
		ContextLength: 16, EmbeddingLength: 2}, AttentionSpec: model.AttentionSpec{KeyLength: 2, ValueLength: 3, HeadCountKV: 1}, EncoderSpec: model.EncoderSpec{DecoderBlockCount: 1},
	}}}
	runner = attachFixtureProgram(runner)
	encoder, _ := reference.NewValue(tensor.MustShape(2, 2), []float32{1, 2, 3, 4})
	payload, err := runner.SaveT5Session(&T5Session{Encoder: encoder})
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range [][]byte{payload[:10], append([]byte(nil), payload...)} {
		if len(data) == len(payload) {
			data[0] = 'X'
		}
		if _, err := runner.LoadT5Session(data); err == nil {
			t.Fatal("invalid T5 session payload was accepted")
		}
	}
}
