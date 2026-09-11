package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
)

func TestStoredRecordComparison(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	r := guardResult(t)
	publish := func(result longform.Result) string {
		t.Helper()
		id, err := longform.Publish(t.Context(), store, result.Inputs.Model, result)
		if err != nil {
			t.Fatal(err)
		}
		return id.String()
	}
	reference := publish(r)
	r.WallNS++
	secondReference := publish(r)
	r.WallNS++
	candidate := publish(r)
	r.Shape.WarmupOutputTokens = longform.WarmupOutputTokens
	initialized := publish(r)
	rateFailure := r
	rateFailure.Shape.Measure.PromptTokensPerSecond *= r.Floors.RateRegressionFraction / 2
	initializedSlow := publish(rateFailure)
	r.Shape.NLL += r.Floors.NLLTolerance * 2
	regression := publish(r)
	r.Inputs.Model, _, err = artifact.Identify(artifact.KindModel, strings.NewReader("other weights"))
	if err != nil {
		t.Fatal(err)
	}
	other := publish(r)
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name                   string
		references, candidates []string
		want                   string
		fails                  bool
	}{
		{"all references", []string{reference, secondReference}, []string{candidate}, "comparisons=2 failures=0", false},
		{"initialization disclosure", []string{reference}, []string{initialized}, "no like-for-like timing improvement claim", false},
		{"initialized rate regression", []string{reference}, []string{initializedSlow}, "short: prompt", true},
		{"regression", []string{reference, secondReference}, []string{regression}, "comparisons=2 failures=2", true},
		{"self comparison", []string{reference}, []string{reference}, "self-comparison", true},
		{"duplicate reference", []string{reference, reference}, []string{candidate}, "duplicate", true},
		{"duplicate candidate", []string{reference}, []string{candidate, candidate}, "duplicate", true},
		{"wrong model", []string{reference}, []string{other}, "no matching reference", true},
		{"unused reference", []string{reference, other}, []string{candidate}, "unused reference", true},
		{"invalid ID", []string{reference}, []string{"invalid"}, "", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			args := []string{"-repo", repository, "-root", filepath.Join(t.TempDir(), "absent"), "-budget", "1m"}
			for _, id := range test.references {
				args = append(args, "-baseline", id)
			}
			for _, id := range test.candidates {
				args = append(args, "-compare-record", id)
			}
			var output strings.Builder
			err := run(args, &output)
			if (err != nil) != test.fails {
				t.Fatalf("error=%v output=%s", err, &output)
			}
			text := output.String()
			if err != nil {
				text += err.Error()
			}
			if !strings.Contains(text, test.want) {
				t.Fatalf("missing %q in %s", test.want, text)
			}
		})
	}
	store, err = overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if after, count := store.Head(); after != head || count != sequence {
		t.Fatal("stored comparisons changed the store")
	}
}

func TestStoredRecordComparisonOptions(t *testing.T) {
	base := []string{"-repo", t.TempDir(), "-budget", "1m", "-baseline", "reference", "-compare-record", "candidate"}
	for _, flags := range [][]string{{"-all"}, {"-publish=false"}, {"-guard"}, {"-check"}, {"-validate-baselines"}, {"-guard-coverage"}, {"-guard-select"}, {"-corpus", "corpus"}, {"-paths-file", "paths"}, {"-model-budget", "1m"}, {"-export-corpus", "out", "-corpus-bytes", "100"}, {"model.gguf"}} {
		if _, err := parseOptions(append(slices.Clone(base), flags...)); err == nil {
			t.Errorf("accepted conflicting flags %v", flags)
		}
	}
	if _, err := parseOptions([]string{"-repo", t.TempDir(), "-budget", "1m", "-compare-record", "candidate"}); err == nil {
		t.Fatal("accepted no references")
	}
}
