package inference

import (
	"encoding/binary"
	"reflect"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func TestSessionStateRoundTripRestoresSampler(t *testing.T) {
	runner := cacheTestRunner()
	runner.spec.Name = "session-fixture"
	runner.spec.ContextLength = 16
	runner.spec.VocabularySize = 32
	session := &Session{
		TokenIDs: []tokenizer.TokenID{1, 2, 3},
		Cache:    cacheTestValue(t),
	}
	session.Cache.Layers[0].States = map[string]LayerState{
		model.CacheStateIndexerKey: {Mode: CacheStateToken, Value: session.Cache.Layers[0].Key.Clone()},
	}
	config := sampling.Config{Temperature: 0.8, TopK: 3, Seed: 91}
	source, err := sampling.New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{0.1, 0.2, 0.3, 0.4}
	if _, err := source.Sample(logits); err != nil {
		t.Fatal(err)
	}
	data, err := runner.SaveSession(session, source)
	if err != nil {
		t.Fatal(err)
	}

	restoredSampler, err := sampling.New(config)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadSession(data, restoredSampler)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored session = %+v, want %+v", restored, session)
	}
	for index := range 20 {
		want, sampleErr := source.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		got, sampleErr := restoredSampler.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if got != want {
			t.Fatalf("resumed sample %d = %d, want %d", index, got, want)
		}
	}
}

func TestShiftedSessionRoundTripUsesAbsolutePosition(t *testing.T) {
	runner := cacheTestRunner()
	runner.spec.Name = "shifted-session-fixture"
	runner.spec.ContextLength = 2
	runner.spec.VocabularySize = 32
	cache := cacheTestValue(t)
	cache.Position = 5
	shifted, err := runner.ShiftCache(cache, 1)
	if err != nil {
		t.Fatal(err)
	}
	session := &Session{
		TokenIDs: []tokenizer.TokenID{1, 2, 3, 4, 5, 6},
		Cache:    shifted,
	}
	sampler, err := sampling.New(sampling.Config{Seed: 17})
	if err != nil {
		t.Fatal(err)
	}
	data, err := runner.SaveSession(session, sampler)
	if err != nil {
		t.Fatal(err)
	}
	restoredSampler, err := sampling.New(sampling.Config{Seed: 17})
	if err != nil {
		t.Fatal(err)
	}
	restored, err := runner.LoadSession(data, restoredSampler)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, session) {
		t.Fatalf("restored shifted session = %+v, want %+v", restored, session)
	}
}

func TestSessionStateRoundTripRestoresVariableGBNFState(t *testing.T) {
	runner := cacheTestRunner()
	runner.spec.Name = "gbnf-session-fixture"
	runner.spec.ContextLength = 16
	runner.spec.VocabularySize = 32
	session := &Session{
		TokenIDs: []tokenizer.TokenID{1, 2, 3},
		Cache:    cacheTestValue(t),
	}
	pieces := make([][]byte, 32)
	for index := range pieces {
		pieces[index] = []byte("x")
	}
	pieces[0], pieces[1], pieces[2] = []byte("a"), []byte("b"), nil
	grammar, err := sampling.NewGBNFGrammar(
		`root ::= "ab"`,
		"root",
		pieces,
		[]int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	config := sampling.Config{GBNF: grammar, Seed: 23}
	source, err := sampling.New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, 32)
	logits[0], logits[1], logits[2] = 10, 9, 100
	if token, sampleErr := source.Sample(logits); sampleErr != nil || token != 0 {
		t.Fatalf("first GBNF token = %d, %v", token, sampleErr)
	}
	data, err := runner.SaveSession(session, source)
	if err != nil {
		t.Fatal(err)
	}
	restoredSampler, err := sampling.New(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.LoadSession(data, restoredSampler); err != nil {
		t.Fatal(err)
	}
	want, err := source.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	got, err := restoredSampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || got != 1 {
		t.Fatalf("restored GBNF sample = %d, want %d", got, want)
	}
}

func TestSessionStateRejectsMalformedMismatchedAndInvalidData(t *testing.T) {
	runner := cacheTestRunner()
	runner.spec.Name = "source"
	runner.spec.ContextLength = 16
	runner.spec.VocabularySize = 32
	session := &Session{
		TokenIDs: []tokenizer.TokenID{1, 2, 3},
		Cache:    cacheTestValue(t),
	}
	config := sampling.Config{Temperature: 0.8, Seed: 7}
	sampler, err := sampling.New(config)
	if err != nil {
		t.Fatal(err)
	}
	data, err := runner.SaveSession(session, sampler)
	if err != nil {
		t.Fatal(err)
	}
	newSampler := func(t *testing.T) *sampling.Sampler {
		t.Helper()
		result, newErr := sampling.New(config)
		if newErr != nil {
			t.Fatal(newErr)
		}
		return result
	}

	for _, length := range []int{0, sessionHeaderSize - 1, sessionHeaderSize, len(data) - 1} {
		if _, loadErr := runner.LoadSession(data[:length], newSampler(t)); loadErr == nil {
			t.Fatalf("truncated length %d was accepted", length)
		}
	}
	withTrailing := append(append([]byte(nil), data...), 0)
	if _, err := runner.LoadSession(withTrailing, newSampler(t)); err == nil {
		t.Fatal("trailing session data was accepted")
	}
	badCount := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badCount[40:], maxSessionTokens+1)
	if _, err := runner.LoadSession(badCount, newSampler(t)); err == nil {
		t.Fatal("excessive token count was accepted")
	}
	badToken := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badToken[sessionHeaderSize:], 1<<31)
	if _, err := runner.LoadSession(badToken, newSampler(t)); err == nil {
		t.Fatal("out-of-range token ID was accepted")
	}
	other := cacheTestRunner()
	other.spec.Name = "different"
	other.spec.ContextLength = 16
	other.spec.VocabularySize = 32
	if _, err := other.LoadSession(data, newSampler(t)); err == nil {
		t.Fatal("session for a different model was accepted")
	}
	wrongConfig, err := sampling.New(sampling.Config{Temperature: 0.7, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.LoadSession(data, wrongConfig); err == nil {
		t.Fatal("session for a different sampler configuration was accepted")
	}

	invalidPending := &Session{
		TokenIDs: []tokenizer.TokenID{1, 2},
		Cache:    cacheTestValue(t),
	}
	if _, err := runner.SaveSession(invalidPending, sampler); err == nil {
		t.Fatal("session without exactly one pending token was accepted")
	}
}
