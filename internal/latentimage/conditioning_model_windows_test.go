//go:build windows

package latentimage

import (
	"context"
	"errors"
	"maps"
	"os"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"slices"
	"testing"
)

func TestMOSSConditioningModelGolden(t *testing.T) {
	cudatest.Require(t)
	const fixture = "model-small-v3-cuda.safetensors"
	mossFixtureBytes(t, fixture, "c96f1e0e709c9569840b10356bc55a47fd002585477fbe2e2692c37bb150786b")
	weights := mossFixtureBytes(t, "model-small-v3-weights.safetensors", "b5364f94c71b0134562493d8e09e8884d63859ce328cf4be4daa2a6440784e2e")
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", fixture))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	root := t.TempDir()
	if len(s.Metadata) != 1 || len(s.Tensors) != 28 {
		t.Fatal("native model fixture coverage")
	}
	for _, metadata := range s.Metadata {
		if metadata["torch"] != "2.12.0+cu126" || metadata["transformers"] != "5.14.1" {
			t.Fatal("native source identity")
		}
		if err := os.WriteFile(filepath.Join(root, "config.json"), []byte(metadata["config"]), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "model.safetensors"), weights, 0600); err != nil {
		t.Fatal(err)
	}
	repo, err := hfrepo.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer repo.Close()
	catalog, err := hfgguf.InspectLanguageModel(repo, hfgguf.LanguageHidden)
	if err != nil {
		t.Fatal(err)
	}
	session, err := graphruntime.NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.Background())
	for _, name := range []string{"t1", "t22", "t65", "t151"} {
		t.Run(name, func(t *testing.T) {
			hidden := refinementValues(t, s, name+".embedding")
			p, err := compileModelEncoder(catalog, len(hidden)/int(catalog.Spec.EmbeddingLength), 0, int(catalog.Spec.BlockCount))
			if err != nil {
				t.Fatal(err)
			}
			names := slices.Sorted(maps.Keys(p.boundaries))
			var outputs []*tensor.Tensor
			for _, label := range names {
				outputs = append(outputs, p.boundaries[label])
			}
			program, err := p.resident(t.Context(), session, catalog, root, outputs...)
			if err != nil {
				t.Fatal(err)
			}
			defer session.Release(context.Background(), program)
			p.feeds[p.input] = reference.Value{Shape: p.input.Shape, Data: hidden}
			var previous map[*tensor.Tensor]reference.Value
			for pass := range 2 {
				actual, err := session.Execute(t.Context(), program, p.feeds)
				if err != nil {
					t.Fatal(err)
				}
				for _, label := range names {
					got := actual[p.boundaries[label]].Data
					want := refinementValues(t, s, name+"."+label)
					failed, unequal, worst := compareMOSSValues(got, want, mossStageTolerance{})
					t.Logf("%s pass=%d values=%d failed=%d unequal=%d max_abs=%g", label, pass, len(want), failed, unequal, worst)
					if failed != 0 || unequal != 0 {
						t.Error("native model differs", label)
					}
					if pass != 0 {
						_, unequal, _ := compareMOSSValues(got, previous[p.boundaries[label]].Data, mossStageTolerance{})
						if unequal != 0 {
							t.Error("repeat differs", label)
						}
					}
				}
				previous = actual
			}
			conditioner := modelConditioner{catalog: catalog, root: root}
			text := textInput{IDs: make([]int, len(hidden)/int(catalog.Spec.EmbeddingLength))}
			text.Mask = make([]bool, len(text.IDs))
			for i := range text.Mask {
				text.Mask[i] = true
			}
			before := session.StaticBytes()
			final, err := conditioner.encode(t.Context(), session, text, hidden, nil)
			if err != nil {
				t.Fatal(err)
			}
			failed, unequal, _ := compareMOSSValues(final, refinementValues(t, s, name+".final_norm"), mossStageTolerance{})
			if failed != 0 || unequal != 0 {
				t.Fatal("final-output-only graph differs from native")
			}
			if session.StaticBytes() != before {
				t.Fatal("conditioner retained temporary weights")
			}
			stop := errors.New("observer stopped")
			if _, err := conditioner.encode(t.Context(), session, text, hidden, func(string, []float32) error { return stop }); !errors.Is(err, stop) {
				t.Fatal("observer failure lost", err)
			}
			if session.StaticBytes() != before {
				t.Fatal("observer failure retained temporary weights")
			}
			cancelled, cancel := context.WithCancel(t.Context())
			cancel()
			if _, err := conditioner.encode(cancelled, session, text, hidden, nil); !errors.Is(err, context.Canceled) {
				t.Fatal("cancelled conditioning accepted", err)
			}
			text.Mask[0] = false
			if _, err := conditioner.encode(t.Context(), session, text, hidden, nil); err == nil {
				t.Fatal("padded conditioner input accepted")
			}
		})
	}
	for _, bounds := range [][3]int{{0, 0, 4}, {1, -1, 4}, {1, 0, 5}, {1, 2, 1}, {1, 1, 1}} {
		if _, err := compileModelEncoder(catalog, bounds[0], bounds[1], bounds[2]); err == nil {
			t.Fatal("invalid encoder range accepted", bounds)
		}
	}
}
