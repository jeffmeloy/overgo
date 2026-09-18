package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestImageVideoReportContract(t *testing.T) {
	t.Run("focused update refuses before store access", func(t *testing.T) {
		savedFlags, savedArgs := flag.CommandLine, os.Args
		defer func() { flag.CommandLine, os.Args = savedFlags, savedArgs }()
		flag.CommandLine = flag.NewFlagSet("compatibility", flag.ContinueOnError)
		os.Args = []string{"compatibility", "-update-media", "-media-scope", "image-video", "-models-repo", filepath.Join(t.TempDir(), "absent")}
		if err := run(); err == nil || !strings.Contains(err.Error(), "always writes the complete docs/MEDIA_REPORT.md") {
			t.Fatalf("focused update must preserve complete report: %v", err)
		}
	})
	ctx := t.Context()
	root := t.TempDir()
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	writeTestManifest(t, root, []claim{}, testModels())
	writeEmptyAcceptedCoverage(t, root)
	register := func(task recipe.Task) recipe.Definition {
		t.Helper()
		payload := []byte(task)
		component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
		manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: component}})
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(root, string(task)+".bin")
		if err := os.WriteFile(path, payload, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, artifact.Batch{Key: "report/model/" + string(task), Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}}, Manifests: []artifact.Manifest{manifest}, Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: component, Kind: artifact.LocationFile, Value: path}, Action: artifact.LocationAdd}}}); err != nil {
			t.Fatal(err)
		}
		var definition recipe.Definition
		if task == recipe.TaskImageGen {
			definition, err = modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, manifest.ID, artifact.ID{})
		} else {
			definition, err = modelrecipe.CapabilityDefinition(task, manifest.ID)
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := modelrecipetest.PublishActivation(ctx, store, "report/active/"+string(task), definition); err != nil {
			t.Fatal(err)
		}
		return definition
	}
	definition := register(recipe.TaskImageGen)
	register(recipe.TaskSpeech)
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, image.NewRGBA(image.Rect(0, 0, 2, 3))); err != nil {
		t.Fatal(err)
	}
	outputID, err := artifact.IdentifyBytes(artifact.KindOutput, buffer.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	output, err := (artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: "image/png", Schema: "overgo.report-fixture-image.v1"}).Content(outputID, buffer.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	input, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.report-fixture-input.v1"), map[string]any{"seed": 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{Key: "report/media", Contents: []artifact.Content{input, output}}); err != nil {
		t.Fatal(err)
	}
	var successful runrecord.Run
	for _, outcome := range []runrecord.Outcome{runrecord.OutcomeSucceeded, runrecord.OutcomeFailed, runrecord.OutcomeCancelled} {
		var outputs []artifact.ID
		failure := ""
		if outcome == runrecord.OutcomeSucceeded {
			outputs = []artifact.ID{outputID}
		}
		if outcome == runrecord.OutcomeFailed {
			failure = "fixture-failure"
		}
		run, err := runrecord.NewRun(definition.ID, outcome, []artifact.ID{input.Descriptor.ID}, outputs, failure)
		if err != nil {
			t.Fatal(err)
		}
		content, err := run.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("report/run/"+string(outcome), []artifact.Content{content}, run.Lineage(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		if outcome == runrecord.OutcomeSucceeded {
			successful = run
		}
	}
	scope, err := resolveMediaReportScope("image-video")
	if err != nil {
		t.Fatal(err)
	}
	if !scope.FrozenRequests || len(scope.Tasks) != 2 {
		t.Fatal("focused scope differs")
	}
	if _, err := resolveMediaReportScope("image"); err == nil {
		t.Fatal("unknown report scope accepted")
	}
	model, ok := definition.Dependency(recipe.DependencyModel, 0)
	if !ok {
		t.Fatal("fixture has no model dependency")
	}
	protocol := imageVideoProtocol{Version: 1, RequiredChecks: []string{"lineage"}, Cases: []imageVideoCase{{ID: "fixture", Model: model, Task: recipe.TaskImageGen, Recipe: definition.ID, Run: successful.ID, Inputs: successful.Inputs, Outputs: successful.Outputs}}}
	writeProtocol := func(value imageVideoProtocol) {
		t.Helper()
		if err := os.MkdirAll(filepath.Join(root, "docs"), 0o700); err != nil {
			t.Fatal(err)
		}
		value.Census = testutil.ArtifactID(t, artifact.KindEvidence, "report fixture census")
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "docs", "image_video_protocol.json"), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeProtocol(protocol)
	if err := exportMediaSamples(root, storePath, scope); err != nil {
		t.Fatal(err)
	}
	first, err := generateMediaReport(root, storePath)
	if err != nil {
		t.Fatal(err)
	}
	second, err := generateMediaReport(root, storePath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Markdown, second.Markdown) || !bytes.Equal(first.JSON, second.JSON) {
		t.Fatal("report rendering is not deterministic")
	}
	var projected mediaProjection
	if err := json.Unmarshal(first.JSON, &projected); err != nil {
		t.Fatal(err)
	}
	if projected.Coverage == nil || projected.Coverage.Activations != 2 || projected.Coverage.AcceptedActivations != 0 {
		t.Fatal("fixture activation or generated sample gained acceptance credit")
	}
	text := string(first.Markdown)
	for _, required := range []string{"# Media capability report", "## Current validation checkpoint", "(image_video_protocol.json)", "failed: 1; cancelled: 1", "fixture-failure", successful.ID.String(), outputID.String(), "Source: unavailable", "Environment: unavailable", "](media_samples/", "`speech`"} {
		if !strings.Contains(text, required) {
			t.Fatalf("report omits %q:\n%s", required, text)
		}
	}
	for _, defect := range []string{"duplicate", "omitted", "lineage", "criteria"} {
		bad := protocol
		bad.Cases = append([]imageVideoCase(nil), protocol.Cases...)
		switch defect {
		case "duplicate":
			bad.Cases = append(bad.Cases, bad.Cases[0])
		case "omitted":
			bad.Cases = nil
		case "lineage":
			bad.Cases[0].Inputs = nil
		case "criteria":
			bad.RequiredChecks = nil
		}
		writeProtocol(bad)
		if err := exportMediaSamples(root, storePath, scope); err == nil {
			t.Fatalf("%s protocol accepted", defect)
		}
	}
	writeProtocol(protocol)
	selection := mediaSampleSelection{definition.ID: {successful.Inputs}}
	changed := successful
	changed.Inputs = []artifact.ID{outputID}
	if !selection.includes(successful) || selection.includes(changed) {
		t.Fatal("sample selection ignores the fixed request")
	}
	name := hashName(buffer.Bytes(), "png")
	path := filepath.Join(root, mediaSamplesDir, name)
	if err := os.WriteFile(path, []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := exportMediaSamples(root, storePath, scope); err == nil {
		t.Fatal("damaged existing export accepted")
	}
	damaged, err := generateMediaReport(root, storePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(damaged.Markdown), "Unavailable or invalid export") || strings.Contains(string(damaged.Markdown), "![") {
		t.Fatal("damaged sample hidden or embedded")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	missing, err := generateMediaReport(root, storePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(missing.Markdown), "can be exported on demand") || !strings.Contains(string(missing.Markdown), outputID.String()) {
		t.Fatal("retained output lacks an export disposition")
	}
	if bytes.Contains(missing.Markdown, []byte(root)) {
		t.Fatal("report embeds the local checkout path")
	}
	invalid := []byte("not a PNG")
	if validateMediaSampleContent(hashName(invalid, "png"), invalid) == nil {
		t.Fatal("hash-correct undecodable image accepted")
	}
	foreign := testutil.ArtifactID(t, artifact.KindRecipe, "foreign recipe")
	testutil.PublishArtifact(t, store, foreign)
	if _, err := store.Commit(ctx, artifact.Batch{Key: "report/stale-lineage", Lineage: []artifact.Lineage{{Child: successful.ID, Parent: foreign, Relation: artifact.RelationDependsOn}}}); err != nil {
		t.Fatal(err)
	}
	if err := visitRecipeRuns(ctx, store, foreign, func(runrecord.Run) error { return nil }); err == nil {
		t.Fatal("substituted recipe lineage accepted")
	}
}

func writeEmptyAcceptedCoverage(t *testing.T, root string) {
	t.Helper()
	// This isolated store has no campaign acceptance. Declare that explicitly;
	// activation and generated samples must not acquire protocol credit.
	coverageRoot := filepath.Join(root, "docs", "verification")
	if err := os.MkdirAll(coverageRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	emptyCoverage, err := json.Marshal(acceptedCoverage{Version: artifact.InitialDocumentVersion, Scope: "Fixture with no accepted campaign protocols."})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"text-vision-coverage.json", "image-video-coverage.json", "specialized-coverage.json"} {
		if err := os.WriteFile(filepath.Join(coverageRoot, name), emptyCoverage, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}
