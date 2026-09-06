package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

// TestE4BProjectedTranscription exercises real Parquet selection, projection,
// autoregressive decoding, persisted run binding and the common WER/CER scorer.
// Its store is temporary: fixture activation and dirty-source test runs never
// become canonical model verification or isolated performance evidence.
func TestE4BProjectedTranscription(t *testing.T) {
	cudatest.Require(t)
	ctx, cancel := context.WithTimeoutCause(t.Context(), 5*time.Minute, errors.New("E4B transcription fixture budget exhausted"))
	defer cancel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer canonical.Close()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(value string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	location := func(value string, kind artifact.LocationKind) string {
		t.Helper()
		path, err := artifact.AvailablePath(ctx, canonical, parse(value), kind)
		if err != nil {
			t.Fatal(err)
		}
		return path
	}
	modelPath := location("tensor-set:sha256:cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501", artifact.LocationFile)
	projectorPath := location("tensor-set:sha256:185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e", artifact.LocationFile)
	datasetID := parse("dataset:sha256:39b4aa4872600afff4be0633dc066adfe5ae36ac17ade4a6ac4a16e26852b2e3")
	corpus := filepath.Join(location(datasetID.String(), artifact.LocationDirectory), "0000.parquet")
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, canonical, modelPath, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = modelrecipe.PublishResolvedModelDefinition(ctx, store, candidate.Inventory, candidate.Resolved); err != nil {
		t.Fatal(err)
	}
	if err = modelrecipetest.PublishActivation(ctx, store, "transcription-fixture/inference", candidate.Definition); err != nil {
		t.Fatal(err)
	}
	projection, err := modelintake.PrepareProjectionCandidate(ctx, modelPath, projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	if err = modelintake.RegisterProjectionCandidate(ctx, store, projection); err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Prompt       string `json:"prompt"`
		MaxTokens    int    `json:"max_tokens"`
		ContainerSHA string `json:"container_sha256"`
		Cases        []struct {
			Name      string   `json:"name"`
			Row       uint64   `json:"row"`
			Reference string   `json:"reference"`
			Samples   uint64   `json:"samples"`
			WAVSHA    string   `json:"wav_sha256"`
			Outputs   []string `json:"accepted_outputs"`
		} `json:"cases"`
	}
	data, err := os.ReadFile(testutil.FixturePath(t, "e4b_audio_validation_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if len(oracle.Cases) != 8 {
		t.Fatal("native validation denominator differs")
	}
	container := parse("file:sha256:" + oracle.ContainerSHA)
	// The Parquet container is the exact selected validation shard. Its bytes
	// stay external; this isolated test only needs the registered identities.
	split := testutil.ArtifactID(t, artifact.KindDatasetShard, "e4b native validation eight/"+container.String())
	environment, err := runrecord.CurrentEnvironment("cuda:0", "hybrid-native")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := environment.Batch("transcription-fixture/environment")
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: datasetID}, artifact.Descriptor{ID: split}, artifact.Descriptor{ID: container})
	if _, err = artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	var maxSamples uint64
	for _, row := range oracle.Cases {
		maxSamples = max(maxSamples, row.Samples)
	}
	// A PCM16 mono RIFF encoding bounds this lossless corpus selection. The
	// selected values are still admitted by the shared signal-quality policy.
	policy := dataset.AudioInspectionPolicy{MaximumEncodedBytes: maxSamples*2 + 44, MaximumSamples: maxSamples, ClipThreshold: 1,
		Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 1.0 / 32768, MaximumAbsoluteDCOffset: 1}}
	// Match the CPU audio acceptance's explicit host decode-workspace budget.
	// This is an admission bound, not a measured host or device peak.
	const sourceMemoryBytes = 4 << 30
	rows, err := dataset.OpenParquetRows(ctx, corpus, []string{"bytes"}, sourceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var inputs []evaluation.TranscriptionResourceInput
	for _, row := range oracle.Cases {
		values, err := rows.Read(ctx, row.Row)
		if err != nil || values["bytes"] == nil {
			t.Fatalf("source row %d unavailable: %v", row.Row, err)
		}
		audio, err := artifact.IdentifyBytes(artifact.KindFile, []byte(*values["bytes"]))
		if err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, evaluation.TranscriptionResourceInput{Name: row.Name, Policy: policy,
			Reference: dataset.AudioPayloadReference{Path: corpus, Audio: audio,
				Origin: dataset.AudioPayloadOrigin{Container: container, Column: "bytes", Row: new(row.Row)}}})
	}
	if err := rows.Close(); err != nil {
		t.Fatal(err)
	}
	source, err := dataset.NewAudioPayloadReader(sourceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	suite := evaluation.TranscriptionSuite{Kind: evaluation.TranscriptionKind, Schema: "overgo/e4b-audio-validation/v1",
		Source:  "Eight pinned LibriSpeech validation cases; native FP32/BF16 SDPA full-transcript oracle; held-out excluded",
		Dataset: datasetID, Split: split, Prompt: oracle.Prompt, MaxTokens: oracle.MaxTokens, DecodeRecipe: candidate.Definition.ID,
		Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase, evaluation.TranscriptionStripPunctuation, evaluation.TranscriptionCollapseWhitespace}}
	expected := make(map[string][]string)
	for index, input := range inputs {
		row := oracle.Cases[index]
		encoded, err := source.Read(ctx, input.Reference, input.Policy.MaximumEncodedBytes)
		if err != nil {
			t.Fatal(err)
		}
		inspection, err := dataset.InspectAudio(ctx, store, encoded, input.Reference.Origin, input.Policy)
		if err != nil {
			t.Fatal(err)
		}
		if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted || uint64(len(inspection.Samples)) != row.Samples {
			t.Fatalf("source admission differs for %s: %+v", row.Name, inspection.Decision)
		}
		wav, err := media.EncodeWAVPCM16(inspection.Samples, int(inspection.Signal.Format.SampleRate))
		if err != nil {
			t.Fatal(err)
		}
		wavID, err := artifact.IdentifyBytes(artifact.KindFile, wav)
		if err != nil || wavID.DigestHex() != row.WAVSHA {
			t.Fatalf("native waveform differs for %s: %s %v", row.Name, wavID, err)
		}
		suite.Cases = append(suite.Cases, evaluation.TranscriptionCase{Name: row.Name, Group: "validation", Source: inspection.Signal.Source,
			Reference: row.Reference, SampleCount: row.Samples, SampleRate: inspection.Signal.Format.SampleRate})
		expected[row.Name] = row.Outputs
	}
	compiled, err := evaluation.CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	authorities := evaluation.ExactAuthorities{ModelDefinition: candidate.Resolved.Document.ID, RuntimeRecipe: projection.Definition.ID,
		CodeCommit: "0123456789abcdef0123456789abcdef01234567", Environment: environment.ID,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident, Prompting: evaluation.PromptingChatTemplate}}
	plan, err := evaluation.BindTranscription(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	report, err := evaluation.EvaluateTranscriptionResources(ctx, store, compiled, plan, candidate.Inventory.Manifest.ID, inputs, evaluation.TranscriptionResourceOptions{TimedRuns: 2}, sourceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.Runs != 16 || report.Summary.AdmissionFailures != 0 || report.Summary.InferenceFailures != 0 || report.QualityScore.Utterances != 8 {
		t.Fatalf("incorrect execution denominator: %+v quality=%+v", report.Summary, report.QualityScore)
	}
	for _, execution := range report.Executions {
		output, err := speechrecognition.RequireTranscription(ctx, store, execution.Output)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(expected[execution.Name], output.Text) {
			t.Fatalf("%s transcript differs from both native precisions: %q", execution.Name, output.Text)
		}
	}
	t.Logf("full transcripts=16/16 native matches; scored utterances=8/8 WER=%g CER=%g; held-out=0; timings diagnostic only", report.QualityScore.WordErrorRate, report.QualityScore.CharacterErrorRate)
	// The same source must become a scored inference failure when its frozen
	// token budget cannot complete; an emitted opener is not a transcript.
	suite.Cases = suite.Cases[:1]
	suite.MaxTokens = 1
	compiled, err = evaluation.CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err = evaluation.BindTranscription(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	truncated, err := evaluation.EvaluateTranscriptionResources(ctx, store, compiled, plan, candidate.Inventory.Manifest.ID, inputs[:1], evaluation.TranscriptionResourceOptions{TimedRuns: 1}, sourceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	if truncated.Summary.InferenceFailures != 1 || truncated.QualityScore.InferenceFailures != 1 || truncated.QualityScore.WordErrorRate != 1 {
		t.Fatalf("truncated transcript escaped the failure denominator: %+v", truncated)
	}
}
