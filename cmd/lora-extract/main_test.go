package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/modelmerge"
	"overgo/internal/overgodb"
	"overgo/internal/peft"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

const queryModule = "model.layers.0.self_attn.q_proj"

// writeRequest writes an extraction request for one two-row, three-column
// weight named tensorName whose fine-tune adds the rank-one update u·v, and a
// store holding the artifacts an extraction of it descends from.
func writeRequest(t *testing.T, tensorName string) (path, store string, base, tuned []float32) {
	t.Helper()
	base = []float32{1, 2, 3, 4, 5, 6}
	u, v := []float32{1, 2}, []float32{1, -1, 0.5}
	tuned = make([]float32, len(base))
	for row := range u {
		for column := range v {
			tuned[row*len(v)+column] = base[row*len(v)+column] + u[row]*v[column]
		}
	}
	layout := tensor.MustShape(3, 2)
	compiler := modelmerge.Compiler{}
	definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "lora-extract definition")
	baseSnapshot, err := compiler.Seal(definition, artifact.ID{}, map[string]modelmerge.Weight{tensorName: {Layout: layout, Values: base}})
	if err != nil {
		t.Fatal(err)
	}
	tunedSnapshot, err := compiler.Seal(definition, baseSnapshot.ID, map[string]modelmerge.Weight{tensorName: {Layout: layout, Values: tuned}})
	if err != nil {
		t.Fatal(err)
	}
	request := extractionRequest{
		Base:  extractionSnapshot{ID: baseSnapshot.ID, Definition: definition, Tensors: baseSnapshot.Tensors},
		Tuned: extractionSnapshot{ID: tunedSnapshot.ID, Definition: definition, Base: baseSnapshot.ID, Tensors: tunedSnapshot.Tensors},
		Policy: modelmerge.LoRAExtractionPolicy{
			MaximumRelativeResidual: 1e-5, Rationale: "keep the least rank that reconstructs the fine-tune", ReopenTrigger: "a different base, tuned weights or budget",
		},
	}
	path = filepath.Join(t.TempDir(), "request.json")
	if err := jsonfile.Write(path, request, 0o600); err != nil {
		t.Fatal(err)
	}
	store = t.TempDir()
	opened, err := overgodb.Open(store)
	if err != nil {
		t.Fatal(err)
	}
	parents := artifact.Batch{Key: "lora-extract/parents", Artifacts: []artifact.Descriptor{{ID: definition}, {ID: baseSnapshot.ID}, {ID: tunedSnapshot.ID}}}
	if _, err := opened.Commit(t.Context(), parents); err != nil {
		t.Fatal(err)
	}
	if err := opened.Close(); err != nil {
		t.Fatal(err)
	}
	return path, store, base, tuned
}

// TestExtractWritesThePEFTAdapter: with -peft the extracted adapter is also
// written as a Hugging Face PEFT directory whose update reproduces the
// fine-tune.
func TestExtractWritesThePEFTAdapter(t *testing.T) {
	request, store, base, tuned := writeRequest(t, queryModule+".weight")
	directory := filepath.Join(t.TempDir(), "adapter")
	if err := extract([]string{"-request", request, "-record", store, "-peft", directory}); err != nil {
		t.Fatal(err)
	}
	adapter, err := peft.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	module, found := adapter.Modules[queryModule]
	if !found || len(adapter.Modules) != 1 || module.Rank != 1 || module.In != 3 || module.Out != 2 || adapter.Config.Rank != 1 || adapter.Config.Alpha != 1 {
		t.Fatalf("adapter = %+v, want one rank-one module of two rows and three columns whose scale is one", adapter)
	}
	delta, err := adapter.Delta(queryModule)
	if err != nil {
		t.Fatal(err)
	}
	for index := range delta {
		if math.Abs(float64(delta[index]-(tuned[index]-base[index]))) > 1e-4 {
			t.Fatalf("update[%d] = %v, want the fine-tune's %v", index, delta[index], tuned[index]-base[index])
		}
	}
}

// TestExtractRefusesAnAdapterPEFTCannotName: a tensor that is not a model's
// .weight has no module PEFT could attach the factors to.
func TestExtractRefusesAnAdapterPEFTCannotName(t *testing.T) {
	request, store, _, _ := writeRequest(t, "weight")
	directory := filepath.Join(t.TempDir(), "adapter")
	err := extract([]string{"-request", request, "-record", store, "-peft", directory})
	if err == nil || !strings.Contains(err.Error(), ".weight") {
		t.Fatalf("err = %v, want a refusal naming .weight", err)
	}
	if _, statErr := os.Stat(directory); statErr == nil {
		t.Fatal("a refused adapter was written")
	}
}
