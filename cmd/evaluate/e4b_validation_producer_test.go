package main

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestE4BValidationProducer(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, value string) artifact.ID { return testutil.ArtifactID(t, kind, value) }
	source := id(artifact.KindDataset, "registered validation")
	inventory := id(artifact.KindDatasetShard, "source inventory")
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "source", Artifacts: []artifact.Descriptor{{ID: source}, {ID: inventory}}, Lineage: []artifact.Lineage{{Child: source, Parent: inventory, Relation: artifact.RelationContains}}}); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "bad inventory split", Lineage: artifact.DependencyLineage(inventory, source)}); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("source inventory cycle was not reproduced: %v", err)
	}
	if after, n := store.Head(); after != head || n != sequence {
		t.Fatal("refused cycle changed the store")
	}
	encoded := []byte("pinned encoded payload; this host fixture checks identity, not audio quality")
	audio, err := artifact.IdentifyBytes(artifact.KindFile, encoded)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "payload")
	if err := os.WriteFile(path, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	origin := dataset.AudioPayloadOrigin{Container: audio}
	input := evaluation.TranscriptionResourceInput{Name: "case", Reference: dataset.AudioPayloadReference{Path: path, Audio: audio, Origin: origin}, Policy: dataset.AudioInspectionPolicy{MaximumEncodedBytes: uint64(len(encoded))}}
	suite := evaluation.TranscriptionSuite{Kind: evaluation.TranscriptionKind, Schema: "fixture/transcription", Source: "frozen host source", Dataset: source,
		Prompt: "transcribe", MaxTokens: 16, DecodeRecipe: id(artifact.KindRecipe, "decode"), Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase},
		Cases: []evaluation.TranscriptionCase{{Name: "case", Group: "validation", Source: recipecontract.AudioReference{Audio: audio, Profile: id(artifact.KindProfile, "format")}, Reference: "target", SampleCount: 1, SampleRate: 16000}}}
	for _, failure := range []struct {
		name   string
		change func(*evaluation.TranscriptionSuite, *[]evaluation.TranscriptionResourceInput)
	}{
		{"inventory split", func(s *evaluation.TranscriptionSuite, _ *[]evaluation.TranscriptionResourceInput) {
			s.Split = inventory
		}},
		{"missing source", func(_ *evaluation.TranscriptionSuite, in *[]evaluation.TranscriptionResourceInput) {
			(*in)[0].Reference.Path += ".absent"
		}},
		{"wrong payload", func(_ *evaluation.TranscriptionSuite, in *[]evaluation.TranscriptionResourceInput) {
			(*in)[0].Reference.Audio = id(artifact.KindFile, "wrong")
		}},
		{"wrong container", func(_ *evaluation.TranscriptionSuite, in *[]evaluation.TranscriptionResourceInput) {
			(*in)[0].Reference.Origin.Container = id(artifact.KindFile, "wrong")
		}},
		{"missing case", func(_ *evaluation.TranscriptionSuite, in *[]evaluation.TranscriptionResourceInput) { *in = nil }},
		{"duplicate case", func(s *evaluation.TranscriptionSuite, in *[]evaluation.TranscriptionResourceInput) {
			s.Cases = append(s.Cases, s.Cases[0])
			*in = append(*in, (*in)[0])
		}},
		{"invalid budget", func(s *evaluation.TranscriptionSuite, _ *[]evaluation.TranscriptionResourceInput) { s.MaxTokens = 0 }},
	} {
		t.Run(failure.name, func(t *testing.T) {
			candidate := suite
			candidate.Cases = slices.Clone(suite.Cases)
			inputs := []evaluation.TranscriptionResourceInput{input}
			failure.change(&candidate, &inputs)
			if err := prepareTranscriptionSelection(t.Context(), store, &candidate, inputs, uint64(len(encoded))); err == nil {
				t.Fatal("invalid preparation passed")
			}
			if after, n := store.Head(); after != head || n != sequence {
				t.Fatal("invalid preparation published evidence")
			}
		})
	}
	if err := prepareTranscriptionSelection(t.Context(), store, &suite, []evaluation.TranscriptionResourceInput{input}, uint64(len(encoded))); err != nil {
		t.Fatal(err)
	}
	if suite.Split == inventory || suite.Split.Kind() != artifact.KindDatasetShard {
		t.Fatal("selection reused inventory")
	}
	parents, err := store.Parents(t.Context(), suite.Split)
	if err != nil || len(parents) != 1 || parents[0].Parent != source {
		t.Fatalf("selection provenance: %v %v", parents, err)
	}
	head, sequence = store.Head()
	if err := prepareTranscriptionSelection(t.Context(), store, &suite, []evaluation.TranscriptionResourceInput{input}, uint64(len(encoded))); err != nil {
		t.Fatal(err)
	}
	if after, n := store.Head(); after != head || n != sequence {
		t.Fatal("exact replay appended evidence")
	}
	before, err := evaluation.CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	suite.Cases[0].Reference = "different target"
	if err := prepareTranscriptionSelection(t.Context(), store, &suite, []evaluation.TranscriptionResourceInput{input}, uint64(len(encoded))); err != nil {
		t.Fatal(err)
	}
	after, err := evaluation.CompileTranscription(suite)
	if err != nil || reflect.DeepEqual(before, after) {
		t.Fatalf("changed target did not rebind the compiled suite: %v", err)
	}
	if after, n := store.Head(); after != head || n != sequence {
		t.Fatal("target change altered target-free source selection")
	}
	t.Log("inventory cycle reproduced; seven invalid preparations refused before publication; exact selection replay stable; model execution=0")
}
