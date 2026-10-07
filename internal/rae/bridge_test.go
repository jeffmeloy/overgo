package rae

import (
	"os"
	"path"
	"path/filepath"
	"testing"
)

// TestABridgedPairNamesThePartsBesideIt holds a bridge to what it is on disk
// and to its arithmetic: saved into a directory it names the encoder and the
// collection beside that directory, and resolves to their paths; a directory
// with no bridge is no bridged pair; a bridge whose encoder is gone, or that
// names a part outside its parent, is refused; and applying it is a row of
// source weights and an offset for each target channel.
func TestABridgedPairNamesThePartsBesideIt(t *testing.T) {
	t.Parallel()
	sampler, err := servedSampler()
	if err != nil {
		t.Fatal(err)
	}
	// A models directory with the shape of a collection, its encoder's image
	// processor and another encoder: ResolveBridgedPair reads names, not weights.
	models := t.TempDir()
	touch := func(parts ...string) {
		t.Helper()
		file := filepath.Join(append([]string{models}, parts...)...)
		if err := os.MkdirAll(filepath.Dir(file), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	touch("collection", filepath.FromSlash(sampler.Checkpoint))
	touch(path.Base(sampler.Encoder), processorConfig)
	touch("other-encoder", "config.json")
	// Two target channels from three source channels.
	bridge := EncoderBridge{Weight: []float32{1, 0, 2, 0, -1, 0.5}, Bias: []float32{10, 20}}
	pairDirectory := filepath.Join(models, "pair")
	if _, held, err := ResolveBridgedPair(pairDirectory); held || err != nil {
		t.Fatalf("a directory with no bridge resolved as a pair (%t, %v)", held, err)
	}
	if err := SaveEncoderBridge(pairDirectory, bridge, "other-encoder", "collection"); err != nil {
		t.Fatal(err)
	}
	pair, held, err := ResolveBridgedPair(pairDirectory)
	if err != nil || !held {
		t.Fatalf("the saved pair did not resolve (%t, %v)", held, err)
	}
	if pair.Bridge != filepath.Join(pairDirectory, bridgeFile) || pair.SourceEncoder != filepath.Join(models, "other-encoder") ||
		pair.Collection.Checkpoint != filepath.Join(models, "collection", filepath.FromSlash(sampler.Checkpoint)) ||
		pair.Collection.EncoderDirectory() != filepath.Join(models, path.Base(sampler.Encoder)) {
		t.Fatalf("the pair resolved to %+v", pair)
	}
	mapped, err := bridge.Apply([]float32{1, 2, 3, 0, 0, 0})
	if err != nil || len(mapped) != 4 || mapped[0] != 17 || mapped[1] != 19.5 || mapped[2] != 10 || mapped[3] != 20 {
		t.Fatalf("two tokens mapped to %v, %v", mapped, err)
	}
	if _, err := bridge.Apply([]float32{1, 2}); err == nil {
		t.Fatal("a token of another width was mapped")
	}
	for name, save := range map[string]func() error{
		"an encoder outside its parent": func() error { return SaveEncoderBridge(pairDirectory, bridge, "../other-encoder", "collection") },
		"no collection":                 func() error { return SaveEncoderBridge(pairDirectory, bridge, "other-encoder", "") },
		"no weights":                    func() error { return SaveEncoderBridge(pairDirectory, EncoderBridge{}, "other-encoder", "collection") },
	} {
		if err := save(); err == nil {
			t.Errorf("a bridge with %s was saved", name)
		}
	}
	if err := os.RemoveAll(filepath.Join(models, "other-encoder")); err != nil {
		t.Fatal(err)
	}
	if _, held, err := ResolveBridgedPair(pairDirectory); !held || err == nil {
		t.Fatalf("a pair whose encoder is gone resolved (%t, %v)", held, err)
	}
}
