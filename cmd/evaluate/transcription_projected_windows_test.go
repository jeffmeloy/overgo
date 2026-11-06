package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestE4BProjectedTranscription exercises real Parquet selection, projection,
// autoregressive decoding, persisted run binding and the common WER/CER scorer.
// Its store is temporary: fixture activation and dirty-source test runs never
// become canonical model verification or isolated performance evidence.
func TestE4BProjectedTranscription(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	testcheck.NoError(t, err)
	canonical := overgodbtest.OpenReadOnly(t, roots.Store)
	store := overgodbtest.Open(t, t.TempDir())
	parse := func(value string) artifact.ID { return testutil.ParseID(t, value) }
	location := func(value string, kind artifact.LocationKind) string {
		t.Helper()
		path := testcheck.Must(artifact.AvailablePath(ctx, canonical, parse(value), kind))(t)
		return path
	}
	modelPath := location("tensor-set:sha256:cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501", artifact.LocationFile)
	projectorPath := location("tensor-set:sha256:185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e", artifact.LocationFile)
	datasetID := parse("dataset:sha256:39b4aa4872600afff4be0633dc066adfe5ae36ac17ade4a6ac4a16e26852b2e3")
	corpus := filepath.Join(location(datasetID.String(), artifact.LocationDirectory), "0000.parquet")
	candidate := testcheck.Must(modelintake.PrepareInferenceCandidate(ctx, canonical, modelPath, modelintake.SessionOverride{}, recipe.ResidencyHybridNative))(t)
	if err = modelrecipe.PublishResolvedModelDefinition(ctx, store, candidate.Inventory, candidate.Resolved); err != nil {
		t.Fatal(err)
	}
	if err = modelrecipetest.PublishActivation(ctx, store, "transcription-fixture/inference", candidate.Definition); err != nil {
		t.Fatal(err)
	}
	projection := testcheck.Must(modelintake.PrepareProjectionCandidate(ctx, canonical, modelPath, projectorPath))(t)
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
	data := testcheck.Must(os.ReadFile(testutil.FixturePath(t, "e4b_audio_validation_golden.json")))(t)
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if len(oracle.Cases) != 8 {
		t.Fatal("native validation denominator differs")
	}
	container := parse("file:sha256:" + oracle.ContainerSHA)
	// Source bytes stay external; production preparation derives an independent
	// selected membership rather than treating this container as the split.
	environment := testcheck.Must(runrecord.CurrentEnvironment("cuda:0", "hybrid-native"))(t)
	batch := testcheck.Must(environment.Batch("transcription-fixture/environment"))(t)
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: datasetID}, artifact.Descriptor{ID: container})
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
	rows := testcheck.Must(dataset.OpenParquetRows(ctx, corpus, []string{"bytes"}, sourceMemoryBytes))(t)
	defer rows.Close()
	var inputs []evaluation.TranscriptionResourceInput
	for _, row := range oracle.Cases {
		values, err := rows.Read(ctx, row.Row)
		if err != nil || values["bytes"] == nil {
			t.Fatalf("source row %d unavailable: %v", row.Row, err)
		}
		audio := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte(*values["bytes"])))(t)
		inputs = append(inputs, evaluation.TranscriptionResourceInput{Name: row.Name, Policy: policy,
			Reference: dataset.AudioPayloadReference{Path: corpus, Audio: audio,
				Origin: dataset.AudioPayloadOrigin{Container: container, Column: "bytes", Row: new(row.Row)}}})
	}
	testcheck.NoError(t, rows.Close())
	source := testcheck.Must(dataset.NewAudioPayloadReader(sourceMemoryBytes))(t)
	defer source.Close()
	suite := evaluation.TranscriptionSuite{Kind: evaluation.TranscriptionKind, Schema: "overgo/e4b-audio-validation/v1",
		Source:  "Eight pinned LibriSpeech validation cases; native FP32/BF16 SDPA full-transcript oracle; held-out excluded",
		Dataset: datasetID, Prompt: oracle.Prompt, MaxTokens: oracle.MaxTokens, DecodeRecipe: candidate.Definition.ID,
		Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase, evaluation.TranscriptionStripPunctuation, evaluation.TranscriptionCollapseWhitespace}}
	expected := make(map[string][]string)
	for index, input := range inputs {
		row := oracle.Cases[index]
		encoded := testcheck.Must(source.Read(ctx, input.Reference, input.Policy.MaximumEncodedBytes))(t)
		inspection := testcheck.Must(dataset.InspectAudio(ctx, store, encoded, input.Reference.Origin, input.Policy))(t)
		if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted || uint64(len(inspection.Samples)) != row.Samples {
			t.Fatalf("source admission differs for %s: %+v", row.Name, inspection.Decision)
		}
		wav := testcheck.Must(media.EncodeWAVPCM16(inspection.Samples, int(inspection.Signal.Format.SampleRate)))(t)
		wavID, err := artifact.IdentifyBytes(artifact.KindFile, wav)
		if err != nil || wavID.DigestHex() != row.WAVSHA {
			t.Fatalf("native waveform differs for %s: %s %v", row.Name, wavID, err)
		}
		suite.Cases = append(suite.Cases, evaluation.TranscriptionCase{Name: row.Name, Group: "validation", Source: inspection.Signal.Source,
			Reference: row.Reference, SampleCount: row.Samples, SampleRate: inspection.Signal.Format.SampleRate})
		expected[row.Name] = row.Outputs
	}
	testcheck.NoError(t, prepareTranscriptionSelection(ctx, store, &suite, inputs, sourceMemoryBytes))
	compiled := testcheck.Must(evaluation.CompileTranscription(suite))(t)
	authorities := evaluation.ExactAuthorities{ModelDefinition: candidate.Resolved.Document.ID, RuntimeRecipe: projection.Definition.ID,
		CodeCommit: "0123456789abcdef0123456789abcdef01234567", Environment: environment.ID,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident, Prompting: evaluation.PromptingChatTemplate}}
	plan := testcheck.Must(evaluation.BindTranscription(compiled, authorities))(t)
	testcheck.NoError(t, source.Close())
	report := testcheck.Must(evaluation.EvaluateTranscriptionResources(ctx, store, compiled, plan, candidate.Inventory.Manifest.ID, inputs, evaluation.TranscriptionResourceOptions{TimedRuns: 2}, sourceMemoryBytes))(t)
	if report.Summary.Runs != 16 || report.Summary.AdmissionFailures != 0 || report.Summary.InferenceFailures != 0 || report.QualityScore.Utterances != 8 {
		t.Fatalf("incorrect execution denominator: %+v quality=%+v", report.Summary, report.QualityScore)
	}
	for _, execution := range report.Executions {
		output := testcheck.Must(speechrecognition.RequireTranscription(ctx, store, execution.Output))(t)
		if !slices.Contains(expected[execution.Name], output.Text) {
			t.Fatalf("%s transcript differs from both native precisions: %q", execution.Name, output.Text)
		}
	}
	t.Logf("full transcripts=16/16 native matches; scored utterances=8/8 WER=%g CER=%g; held-out=0; timings diagnostic only", report.QualityScore.WordErrorRate, report.QualityScore.CharacterErrorRate)
	// The same source must become a scored inference failure when its frozen
	// token budget cannot complete; an emitted opener is not a transcript.
	suite.Cases = suite.Cases[:1]
	suite.MaxTokens = 1
	compiled = testcheck.Must(evaluation.CompileTranscription(suite))(t)
	plan = testcheck.Must(evaluation.BindTranscription(compiled, authorities))(t)
	truncated := testcheck.Must(evaluation.EvaluateTranscriptionResources(ctx, store, compiled, plan, candidate.Inventory.Manifest.ID, inputs[:1], evaluation.TranscriptionResourceOptions{TimedRuns: 1}, sourceMemoryBytes))(t)
	if truncated.Summary.InferenceFailures != 1 || truncated.QualityScore.InferenceFailures != 1 || truncated.QualityScore.WordErrorRate != 1 {
		t.Fatalf("truncated transcript escaped the failure denominator: %+v", truncated)
	}
}
