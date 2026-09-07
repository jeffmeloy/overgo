package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type validationFixtureRuntime struct{}

func (validationFixtureRuntime) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if prompt != "fixture input" || options.MaxNewTokens != 2 {
		return nil, "", errors.New("host acceptance fixture received a different workload")
	}
	ids := []tokenizer.TokenID{11, 12}
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(ids)})
	for index, piece := range []string{"o", "k"} {
		id := tokenizer.TokenID(index)
		ids = append(ids, id)
		if err := options.OnToken(inference.TokenEvent{ID: id, Piece: piece}); err != nil {
			return nil, "", err
		}
	}
	return ids, "", nil
}

func (validationFixtureRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("host acceptance fixture does not implement likelihood scoring")
}

func TestE4BValidationAcceptanceContract(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, value string) artifact.ID { return testutil.ArtifactID(t, kind, value) }
	model, projector, definition := id(artifact.KindModel, "model"), id(artifact.KindProjector, "projector"), id(artifact.KindModelDefinition, "definition")
	projection, err := modelrecipe.ProjectionDefinition(model, projector, artifact.ID{}, recipe.DataImage)
	if err != nil {
		t.Fatal(err)
	}
	content, err := projection.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "fixture/models", Artifacts: []artifact.Descriptor{{ID: model}, {ID: projector}, {ID: definition}}, Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	spec := modelValidationSpecification{Model: model, Projector: projector, CodeCommit: strings.Repeat("a", 40)}
	campaign, err := evaluation.NewIsolatedCampaign(store, validationFixtureRuntime{}, modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: projection.ID}, environment, spec.CodeCommit)
	if err != nil {
		t.Fatal(err)
	}
	// These are host evidence fixtures, not claims of real multimodal quality.
	required := []string{"quality/video/native", "quality/image/native"}
	for _, name := range required {
		data, err := json.Marshal(evaluation.ExactSuite{Schema: "fixture/exact", Source: name, Cases: []evaluation.ExactCase{{Name: "one", Prompt: "fixture input", MaxTokens: 2, Text: "ok", PromptTokens: 2, GeneratedTokens: 2}}})
		if err != nil {
			t.Fatal(err)
		}
		suite, err := evaluation.CompileSuite(data, campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, err := campaign.Evaluate(t.Context(), suite)
		if err != nil {
			t.Fatal(err)
		}
		evidence, err := evaluation.RequireEvaluationEvidence(t.Context(), store, result.Evidence)
		if err != nil {
			t.Fatal(err)
		}
		metric := evidence.Metrics[0]
		spec.Cells = append(spec.Cells, modelValidationCell{Name: name, Evidence: evidence.ID, Plan: evidence.Plan, ModelDefinition: definition, Recipe: projection.ID, Environment: environment.ID, Dataset: evidence.Dataset, Split: evidence.Split, Shards: evidence.Shards,
			Bounds: []modelValidationBound{{Metric: metric.Name, Unit: metric.Unit, Direction: metric.Direction, Minimum: metric.Value, Maximum: metric.Value}}})
	}
	head, sequence := store.Head()
	if err := checkModelValidation(t.Context(), store, spec, required); err != nil {
		t.Fatal(err)
	}
	if after, n := store.Head(); after != head || n != sequence {
		t.Fatal("read-only acceptance wrote evidence")
	}
	for _, failure := range []struct {
		name   string
		change func(*modelValidationSpecification)
	}{
		{"missing modality", func(s *modelValidationSpecification) { s.Cells = s.Cells[:1] }},
		{"reused evidence", func(s *modelValidationSpecification) { s.Cells[1].Evidence = s.Cells[0].Evidence }},
		{"reused plan", func(s *modelValidationSpecification) { s.Cells[1].Plan = s.Cells[0].Plan }},
		{"wrong mode", func(s *modelValidationSpecification) { s.Cells[0].Name = "protocol/video/native" }},
		{"stale source", func(s *modelValidationSpecification) { s.CodeCommit = strings.Repeat("b", 40) }},
		{"wrong environment", func(s *modelValidationSpecification) {
			s.Cells[0].Environment = id(artifact.KindEvidence, "other environment")
		}},
		{"wrong model", func(s *modelValidationSpecification) { s.Model = id(artifact.KindModel, "other model") }},
		{"wrong projector", func(s *modelValidationSpecification) { s.Projector = id(artifact.KindProjector, "other projector") }},
		{"missing evidence", func(s *modelValidationSpecification) { s.Cells[0].Evidence = id(artifact.KindEvidence, "missing") }},
		{"missing cases", func(s *modelValidationSpecification) { s.Cells[0].Shards = nil }},
		{"duplicate cases", func(s *modelValidationSpecification) {
			s.Cells[0].Shards = append(s.Cells[0].Shards, s.Cells[0].Shards[0])
		}},
		{"wrong dataset", func(s *modelValidationSpecification) { s.Cells[0].Dataset = id(artifact.KindDataset, "other data") }},
		{"no threshold", func(s *modelValidationSpecification) { s.Cells[0].Bounds = nil }},
		{"failed threshold", func(s *modelValidationSpecification) {
			s.Cells[0].Bounds[0].Maximum = 0
			s.Cells[0].Bounds[0].Minimum = 0
		}},
		{"nonfinite threshold", func(s *modelValidationSpecification) { s.Cells[0].Bounds[0].Maximum = math.NaN() }},
	} {
		t.Run(failure.name, func(t *testing.T) {
			candidate := spec
			candidate.Cells = slices.Clone(spec.Cells)
			for i := range candidate.Cells {
				candidate.Cells[i].Shards = slices.Clone(spec.Cells[i].Shards)
				candidate.Cells[i].Bounds = slices.Clone(spec.Cells[i].Bounds)
			}
			failure.change(&candidate)
			if err := checkModelValidation(t.Context(), store, candidate, required); err == nil {
				t.Fatal("invalid evidence accepted")
			}
		})
	}
	if cells := e4bValidationCells(); len(cells) != 30 {
		t.Fatalf("E4B denominator=%d want 18 protocol + 6 quality + 6 resource cells", len(cells))
	}
	if err := checkModelValidation(t.Context(), store, spec, e4bValidationCells()); err == nil {
		t.Fatal("partial host fixture claimed complete E4B")
	}
	t.Run("one executed protocol bundle covers eighteen cells", func(t *testing.T) {
		oracle := strings.Repeat("c", 64)
		record, err := runrecord.NewGateRecord(projection.ID, environment.ID, spec.CodeCommit, runrecord.OutcomeSucceeded, "", 1,
			[]runrecord.GateStep{{Name: "candidate-execution", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1,
				Evidence: fmt.Sprintf("contract=e4b-http-modalities;oracle_sha256=%s;model_definition=%s;cases=18", oracle, definition)}})
		if err != nil {
			t.Fatal(err)
		}
		batch, err := record.Batch("fixture/protocol-gate")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		protocols := spec
		protocols.Cells = nil
		var names []string
		for _, name := range e4bValidationCells() {
			if strings.HasPrefix(name, "protocol/") {
				names = append(names, name)
				protocols.Cells = append(protocols.Cells, modelValidationCell{Name: name, Evidence: record.Result.ID, Run: record.Run.ID,
					OracleSHA256: oracle, ModelDefinition: definition, Recipe: projection.ID, Environment: environment.ID})
			}
		}
		if err := checkModelValidation(t.Context(), store, protocols, names); err != nil {
			t.Fatal(err)
		}
		for _, mutate := range []func(*modelValidationCell){
			func(c *modelValidationCell) { c.OracleSHA256 = strings.Repeat("d", 64) },
			func(c *modelValidationCell) { c.Run = id(artifact.KindRun, "missing") },
			func(c *modelValidationCell) { c.ModelDefinition = id(artifact.KindModelDefinition, "other") },
			func(c *modelValidationCell) { c.Environment = id(artifact.KindEvidence, "other") },
			func(c *modelValidationCell) { *c = spec.Cells[0]; c.Name = names[0] },
		} {
			bad := protocols
			bad.Cells = slices.Clone(protocols.Cells)
			mutate(&bad.Cells[0])
			if err := checkModelValidation(t.Context(), store, bad, names); err == nil {
				t.Fatal("altered protocol binding or renamed native evaluation accepted")
			}
		}
	})
	t.Run("observed media resource contract", func(t *testing.T) {
		testMediaResourceAcceptance(t, store, spec, projection.ID, environment.ID, definition)
	})
	t.Run("exact output cannot substitute for transcription quality", func(t *testing.T) {
		candidate := spec
		candidate.Cells = slices.Clone(spec.Cells[:1])
		candidate.Cells[0].Name = "quality/audio/native"
		if err := checkModelValidation(t.Context(), store, candidate, []string{candidate.Cells[0].Name}); err == nil {
			t.Fatal("exact-output report bypassed WER and CER acceptance")
		}
	})
	t.Run("existing transcription owner", func(t *testing.T) {
		dataset, split := id(artifact.KindDataset, "audio data"), id(artifact.KindDatasetShard, "audio selection")
		source := recipecontract.AudioReference{Audio: id(artifact.KindFile, "audio"), Profile: id(artifact.KindProfile, "audio format")}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "fixture/audio-source", Artifacts: []artifact.Descriptor{{ID: dataset}, {ID: split}, {ID: source.Audio}, {ID: source.Profile}}}); err != nil {
			t.Fatal(err)
		}
		compiled, err := evaluation.CompileTranscription(evaluation.TranscriptionSuite{
			Kind: evaluation.TranscriptionKind, Schema: "fixture/transcription", Source: "host-only stored-output fixture",
			Dataset: dataset, Split: split, Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase},
			Cases: []evaluation.TranscriptionCase{{Name: "one", Group: "fixture", Source: source, Reference: "hello", SampleCount: 16000, SampleRate: 16000}},
		})
		if err != nil {
			t.Fatal(err)
		}
		plan, err := evaluation.BindTranscription(compiled, evaluation.ExactAuthorities{ModelDefinition: definition, RuntimeRecipe: projection.ID, CodeCommit: spec.CodeCommit,
			Environment: environment.ID, Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident}})
		if err != nil {
			t.Fatal(err)
		}
		output, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema), recipecontract.Transcription{Source: source, Text: "hello"})
		if err != nil {
			t.Fatal(err)
		}
		run, err := runrecord.NewBoundRun(projection.ID, runrecord.OutcomeSucceeded, []artifact.ID{dataset, split, source.Audio, source.Profile}, []artifact.ID{output.Descriptor.ID}, "",
			spec.CodeCommit, environment.ID, 1, []runrecord.PhaseMetric{{Phase: runrecord.PhaseGenerate, DurationNS: 1}})
		if err != nil {
			t.Fatal(err)
		}
		runContent, err := run.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("fixture/audio-output", []artifact.Content{output, runContent}, run.Lineage(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		report, err := evaluation.EvaluateTranscription(t.Context(), store, compiled, plan, []evaluation.TranscriptionPrediction{{Name: "one", Run: run.ID}})
		if err != nil {
			t.Fatal(err)
		}
		cell := modelValidationCell{Name: "quality/audio/native", Evidence: report.ID, Plan: plan.Identity(), ModelDefinition: definition, Recipe: projection.ID,
			Environment: environment.ID, Dataset: dataset, Split: split, Cases: []string{"one"}, Bounds: []modelValidationBound{
				{Metric: "wer", Unit: "ratio", Direction: runrecord.DirectionMinimize}, {Metric: "cer", Unit: "ratio", Direction: runrecord.DirectionMinimize},
			}}
		audioSpec := spec
		audioSpec.Cells = []modelValidationCell{cell}
		if err := checkModelValidation(t.Context(), store, audioSpec, []string{cell.Name}); err != nil {
			t.Fatal(err)
		}
		for _, bad := range []modelValidationCell{
			func() modelValidationCell { v := cell; v.Cases = nil; return v }(),
			func() modelValidationCell { v := cell; v.Cases = []string{"one", "one"}; return v }(),
			func() modelValidationCell { v := cell; v.Bounds = v.Bounds[:1]; return v }(),
			func() modelValidationCell { v := cell; v.Name = "resources/audio/native"; return v }(),
		} {
			audioSpec.Cells = []modelValidationCell{bad}
			if err := checkModelValidation(t.Context(), store, audioSpec, []string{bad.Name}); err == nil {
				t.Fatal("incomplete audio acceptance or resource substitution passed")
			}
		}
	})
	resource := spec.Cells[0]
	resource.Name = "resources/image/native"
	value, err := evaluation.RequireEvaluationEvidence(t.Context(), store, resource.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkModelValidationCell(spec, resource, value); err == nil {
		t.Fatal("unmeasured GPU peak accepted")
	}
	t.Log("two canonical host evidence records accepted read-only; 15 invalid joins refused; incomplete E4B and absent GPU measurements refused; model executions=0")
}
