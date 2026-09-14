package main

import (
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestQwenFourNativeReferenceAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for retained Qwen4 native evidence")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(id artifact.ID, target any) {
		t.Helper()
		content, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(content.Data, target); err != nil {
			t.Fatal(err)
		}
	}
	id, err := artifact.ParseID("evidence:sha256:0a4c2403d85cda517e4145d295552303b3a3a0414ff8f72abdb180d907132b2c")
	if err != nil {
		t.Fatal(err)
	}
	var provenance struct {
		Schema, Model, Recipe string
		GoCommit              string                 `json:"go_commit"`
		NativeCommit          string                 `json:"native_commit"`
		Inputs                map[string]artifact.ID `json:"inputs"`
	}
	read(id, &provenance)
	if provenance.Schema != "overgo/native-reference-provenance/v1" ||
		provenance.Model != "model:sha256:887f41243686e101a314013a168b6bad5df19f8bee5df3e45d942ff79455413b" ||
		provenance.Recipe != "recipe:sha256:49d1ed7c41e45abcc9c10ff13ec247ab775f7054123d5507caa1f63c0a19f803" ||
		provenance.GoCommit != "3abc5e1c2bd6e46e4d350bda88698fc2e1013b26" ||
		provenance.NativeCommit != "42fc243060709331ff9b158a9ed2cbe37219ae83" {
		t.Fatal("native reference producer or identity differs")
	}
	for _, name := range []string{"go_capture", "native_capture", "native_input", "go_source", "native_source", "native_build", "metadata_source", "metadata_verification", "widen_source", "widen_verification", "native_runner", "attempt", "native_receipt", "metadata_failure", "dtype_failure", "historical_fixture"} {
		if _, err := artifact.RequireTypedContent(t.Context(), store, provenance.Inputs[name]); err != nil {
			t.Fatalf("reference input %s: %v", name, err)
		}
	}
	var captured struct {
		nativeTextCapture
		Producer    string
		Identity    modelrecipe.ProgramIdentity
		Environment runrecord.Environment
	}
	var native, submitted nativeTextCapture
	read(provenance.Inputs["go_capture"], &captured)
	read(provenance.Inputs["native_capture"], &native)
	read(provenance.Inputs["native_input"], &submitted)
	if captured.Producer != provenance.GoCommit || captured.Identity.Model.String() != provenance.Model || captured.Identity.Recipe.String() != provenance.Recipe || captured.Environment.Backend != "cuda" {
		t.Fatal("executed producer, model, recipe or backend differs")
	}
	if _, err := runrecord.NewEnvironment(captured.Environment); err != nil {
		t.Fatal(err)
	}
	active, found, err := modelrecipe.ActiveRecord(t.Context(), store, captured.Identity.Model, recipe.TaskInference)
	if err != nil || !found || active.Definition.ID != captured.Identity.Recipe {
		t.Fatalf("active recipe differs: found=%t err=%v", found, err)
	}
	for role, expected := range map[recipe.DependencyRole]artifact.ID{
		recipe.DependencyModel: captured.Identity.Model, recipe.DependencyProfile: captured.Identity.Profile, recipe.DependencyDefinition: captured.Identity.Definition,
	} {
		if actual, found := active.Definition.PrimaryDependency(role); !found || actual != expected {
			t.Fatalf("recipe dependency %s differs", role)
		}
	}
	if !reflect.DeepEqual(captured.nativeTextCapture, submitted) {
		t.Fatal("native process received different cases or prompting")
	}
	if err := checkNativeTextCaptures(captured.nativeTextCapture, native, 42, 26); err != nil {
		t.Fatal(err)
	}
	for index, row := range captured.Cases {
		if row.Chat != (index == 3) || row.MaxTokens != []int{8, 8, 8, 16}[index] {
			t.Fatal("declared raw/chat protocol or token cap differs")
		}
	}
	var metadata struct {
		SourceSHA256           string `json:"source_sha256"`
		DestinationSHA256      string `json:"destination_sha256"`
		ByteIdenticalTensors   int    `json:"byte_identical_tensors"`
		ChangedMetadataEntries int    `json:"changed_metadata_entries"`
	}
	var widening struct {
		SourceSHA256, DestinationSHA256      string
		WidenedVectors, ByteIdenticalTensors int
	}
	var historical struct {
		ModelSHA256 string `json:"model_sha256"`
	}
	read(provenance.Inputs["metadata_verification"], &metadata)
	read(provenance.Inputs["widen_verification"], &widening)
	read(provenance.Inputs["historical_fixture"], &historical)
	if metadata.SourceSHA256 != historical.ModelSHA256 || metadata.SourceSHA256 != "c4e8dc8f885d590f146d96919562960a5d040ccc366623e986df8656539419ec" ||
		metadata.ChangedMetadataEntries != 1 || metadata.ByteIdenticalTensors != 441 || metadata.DestinationSHA256 != widening.SourceSHA256 ||
		widening.WidenedVectors != 24 || widening.ByteIdenticalTensors != 417 || widening.DestinationSHA256 != "aa7d198450dd45279ec3d5fb797913feaccac5f9e3a371e98b6b8ac9402e7e6a" {
		t.Fatal("reference conversion lost its exact weight binding")
	}
	for _, mutate := range []func(*nativeTextCapture){
		func(c *nativeTextCapture) { c.Cases = c.Cases[1:] },
		func(c *nativeTextCapture) { c.Cases[1] = c.Cases[0] },
		func(c *nativeTextCapture) { c.Cases[0].Text += "changed" },
		func(c *nativeTextCapture) { c.Cases[0].Output = []int32{0} },
	} {
		corrupt := nativeTextCapture{Cases: slices.Clone(native.Cases)}
		mutate(&corrupt)
		if checkNativeTextCaptures(captured.nativeTextCapture, corrupt, 42, 26) == nil {
			t.Fatal("accepted corrupt native comparison")
		}
	}
	t.Log("four native cases match: 42 input and 26 generated tokens; 441 tensors preserved or exactly widened; no model loads")
}
