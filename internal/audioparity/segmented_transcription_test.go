package audioparity

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechactivity"
	"overgo/internal/speechrecognition"
)

// The complete capture is retained in testdata, including source/runtime/file
// identities and native tensor identity. This acceptance reads transcript/span
// boundaries only; it does not claim fresh logit or performance comparison.
type segmentedReference struct {
	Schema              string            `json:"schema"`
	Source              SourceIdentity    `json:"source"`
	Model               artifact.Manifest `json:"model"`
	VADSource           string            `json:"vad_source"`
	VADCaptureSHA256    string            `json:"vad_capture_sha256"`
	VADCheckpointSHA256 string            `json:"vad_checkpoint_sha256"`
	Corpus              struct {
		AudioSHA256 string `json:"audio_sha256"`
		ShardSHA256 string `json:"shard_sha256"`
		Samples     uint64 `json:"samples"`
	} `json:"corpus"`
	Policies []struct {
		Name   string                       `json:"name"`
		Policy speechactivity.OfflineConfig `json:"policy"`
		Spans  [][2]uint64                  `json:"frame_exact_spans"`
		Pieces []struct {
			Text   string `json:"text"`
			SHA256 string `json:"sample_sha256"`
		} `json:"pieces"`
		Text string `json:"assembled_text"`
	} `json:"policies"`
}

type segmentedFixture struct {
	*vadLifecycle
	clip       ctcTrainingFixture
	definition recipe.Definition
	policy     dataset.AudioInspectionPolicy
	origin     dataset.AudioPayloadOrigin
	binding    speechrecognition.RunBinding
	want       segmentedReference
}

func newSegmentedFixture(t *testing.T) *segmentedFixture {
	t.Helper()
	reference, store, _ := loadVADReference(t)
	l := newVADLifecycle(t, store, reference.Models[0], false)
	f := &segmentedFixture{vadLifecycle: l, clip: loadCTCTrainingFixture(t)}
	data, err := os.ReadFile("testdata/segmented_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &f.want); err != nil {
		t.Fatal(err)
	}
	if f.want.VADCheckpointSHA256 != reference.Models[0].SHA256 || f.want.VADSource != reference.Source {
		t.Fatal("activity capture source differs")
	}
	data, err = os.ReadFile("testdata/vad_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	captureID, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil || captureID.DigestHex() != f.want.VADCaptureSHA256 {
		t.Fatal("activity probability capture differs")
	}
	if f.want.Schema != "overgo/segmented-asr-reference/v1" || f.want.Source != f.clip.election.ModelSource || f.want.Model.ID != f.clip.election.Model.ID ||
		f.want.Corpus.AudioSHA256 != f.clip.record.AudioSHA256 ||
		f.want.Corpus.ShardSHA256 != f.clip.record.ShardSHA256 || f.want.Corpus.Samples != f.clip.record.Samples || len(f.want.Policies) != 2 ||
		f.want.Policies[1].Name != "accepted-vad" || f.want.Policies[1].Policy != *l.profile.Offline {
		t.Fatal("segmented reference authority differs")
	}
	base, _ := l.publishTranscriptionBase(t, f.clip)
	activity, err := recipe.RequireDefinition(t.Context(), l.store, l.recipe)
	if err != nil {
		t.Fatal(err)
	}
	f.definition, err = modelrecipe.SegmentedTranscriptionDefinition(base, activity)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), l.store, "segmented/recipe", f.definition); err != nil {
		t.Fatal(err)
	}
	shard, err := artifact.ParseID("file:sha256:" + f.clip.record.ShardSHA256)
	if err != nil {
		t.Fatal(err)
	}
	row := f.clip.record.Row
	f.origin = dataset.AudioPayloadOrigin{Container: shard, Column: "audio.bytes", Row: &row}
	l.commit(t, artifact.Batch{Key: "segmented/shard", Artifacts: []artifact.Descriptor{{ID: shard, Size: f.clip.record.ShardBytes}}}, nil)
	data, err = os.ReadFile("../dataset/testdata/audio_inspection_policy.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &f.policy); err != nil {
		t.Fatal(err)
	}
	f.policy.MaximumEncodedBytes, f.policy.MaximumSamples = uint64(len(f.clip.payload)), f.clip.record.Samples
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := environment.Batch("segmented/environment")
	l.commit(t, batch, err)
	f.binding = speechrecognition.RunBinding{Key: "segmented/dataset", CodeCommit: strings.TrimSpace(baselineCommand(t, f.clip.root, "git", "rev-parse", "HEAD")), Environment: environment.ID}
	return f
}

func (f *segmentedFixture) run(t *testing.T) (recipecontract.Transcription, runrecord.Run) {
	t.Helper()
	session, err := speechrecognition.LoadSession(t.Context(), f.store, f.definition.ID, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	}()
	lease, err := session.Lease(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	result, run, err := lease.Transcribe(t.Context(), f.clip.payload, f.origin, f.policy, f.binding)
	if err != nil {
		t.Fatal(err)
	}
	want := f.want.Policies[1]
	if result.Text != want.Text || run.Recipe != f.definition.ID || run.Outcome != runrecord.OutcomeSucceeded || len(run.Outputs) != 1 ||
		!slices.Contains(run.Inputs, f.profile.Model) || !slices.Contains(run.Inputs, f.definition.Model) {
		t.Fatalf("segmented result text=%q recipe=%s outputs=%d", result.Text, run.Recipe, len(run.Outputs))
	}
	var compositionID artifact.ID
	for _, id := range run.Inputs {
		descriptor, found, err := f.store.Artifact(t.Context(), id)
		if err != nil {
			t.Fatal(err)
		}
		if found && descriptor.Schema == "overgo/segmented-transcription/v1" {
			if compositionID.Valid() {
				t.Fatal("multiple composition evidence records")
			}
			compositionID = id
		}
	}
	content, err := artifact.RequireTypedContent(t.Context(), f.store, compositionID)
	if err != nil || content.Descriptor.Schema != "overgo/segmented-transcription/v1" {
		t.Fatalf("composition output: %v", err)
	}
	var composition struct {
		Source   recipecontract.AudioReference `json:"source"`
		Activity artifact.ID                   `json:"activity"`
		Pieces   []struct {
			Span   recipecontract.SampleSpan `json:"span"`
			SHA256 string                    `json:"samples_sha256"`
			Text   string                    `json:"text"`
		} `json:"pieces"`
	}
	if err := json.Unmarshal(content.Data, &composition); err != nil {
		t.Fatal(err)
	}
	if composition.Source != result.Source || len(composition.Pieces) != len(want.Pieces) || len(want.Spans) != len(want.Pieces) {
		t.Fatal("composition source or piece count differs")
	}
	activity, err := speechactivity.RequireActivity(t.Context(), f.store, composition.Activity)
	if err != nil || len(activity.Segments) != len(want.Pieces) {
		t.Fatalf("activity output: %v", err)
	}
	for i, piece := range composition.Pieces {
		if piece.Span != (recipecontract.SampleSpan{Start: want.Spans[i][0], End: want.Spans[i][1]}) || piece.Span != activity.Segments[i].Span || piece.SHA256 != want.Pieces[i].SHA256 || piece.Text != want.Pieces[i].Text {
			t.Fatalf("piece %d differs: %+v", i, piece)
		}
	}
	parents, err := f.store.Parents(t.Context(), compositionID)
	if err != nil {
		t.Fatal(err)
	}
	for _, parent := range []artifact.ID{f.definition.ID, composition.Activity, result.Source.Audio, result.Source.Profile} {
		if !slices.Contains(parents, artifact.Lineage{Child: compositionID, Parent: parent, Relation: artifact.RelationDependsOn}) {
			t.Fatalf("missing composition parent %s", parent)
		}
	}
	// Dataset intake stores a source descriptor, not a copied encoded recording.
	if _, found, err := artifact.ReadContent(t.Context(), f.store, result.Source.Audio); err != nil || found {
		t.Fatalf("dataset bytes copied to store: found=%t err=%v", found, err)
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	if snapshot := session.Snapshot(); snapshot.Loads != 2 || snapshot.Active != 0 {
		t.Fatalf("component residency differs: %+v", snapshot)
	}
	if _, _, err := lease.Transcribe(t.Context(), f.clip.payload, f.origin, f.policy, f.binding); err == nil {
		t.Fatal("released lease executed")
	}
	t.Logf("CPU composed %d native-reference spans and transcripts on %d samples; exact PCM identities and lineage; no held-out quality, word alignment, streaming, logit or GPU claim", len(want.Pieces), f.clip.record.Samples)
	return result, run
}

func TestSegmentedTranscriptionAcceptance(t *testing.T) {
	f := newSegmentedFixture(t)
	f.run(t)
}

func TestValidatedTranscriptionConsumers(t *testing.T) {
	f := newSegmentedFixture(t)
	silence, err := media.EncodeWAVPCM16(make([]float32, f.clip.record.Samples), int(f.clip.record.SampleRate))
	if err != nil {
		t.Fatal(err)
	}
	f.policy.MaximumEncodedBytes = max(f.policy.MaximumEncodedBytes, uint64(len(silence)))
	result, _ := f.run(t)
	f.evaluate(t, result)
	handler, _ := newAudioHTTPFixture(t, f.store, f.definition, f.policy, f.binding.CodeCommit)
	for _, test := range []struct {
		name       string
		data       []byte
		authorized bool
		status     int
	}{
		{"unauthorized", f.clip.payload, false, http.StatusUnauthorized},
		{"speech", f.clip.payload, true, http.StatusOK},
		{"silence", silence, true, http.StatusBadRequest},
		{"corrupt", []byte("invalid audio"), true, http.StatusBadRequest},
		{"empty", nil, true, http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := audioHTTPRequest(t, f.definition, test.data, test.authorized)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.status {
				t.Fatalf("HTTP %d want %d: %s", response.Code, test.status, response.Body)
			}
			if test.status == http.StatusOK {
				var got struct {
					Text string `json:"text"`
				}
				if err := json.Unmarshal(response.Body.Bytes(), &got); err != nil || got.Text != result.Text {
					t.Fatalf("HTTP transcript %q differs: %v", got.Text, err)
				}
			}
		})
	}
	t.Logf("same real segmented recipe %s through source-addressed dataset input and authenticated HTTP; 1 speech and 4 refusal controls; no production activation", f.definition.ID)
}

func (f *segmentedFixture) evaluate(t *testing.T, expected recipecontract.Transcription) {
	t.Helper()
	metadata, err := modelrecipe.PublishTaskModelDefinition(t.Context(), f.store, f.definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	selection := struct {
		Origin dataset.AudioPayloadOrigin `json:"origin"`
		Audio  artifact.ID                `json:"audio"`
	}{f.origin, expected.Source.Audio}
	data, err := artifact.JSONContent(artifact.JSONContract(artifact.KindDataset, "overgo/audio-selected-corpus/v1"), selection)
	if err != nil {
		t.Fatal(err)
	}
	split, err := artifact.JSONContent(artifact.JSONContract(artifact.KindDatasetShard, "overgo/audio-selected-split/v1"), selection)
	if err != nil {
		t.Fatal(err)
	}
	f.commit(t, artifact.Batch{Key: "segmented/evaluation-data", Contents: []artifact.Content{data, split}, Lineage: append(artifact.DependencyLineage(data.Descriptor.ID, f.origin.Container), artifact.DependencyLineage(split.Descriptor.ID, data.Descriptor.ID)...)}, nil)
	compiled, err := evaluation.CompileTranscription(evaluation.TranscriptionSuite{
		Kind: evaluation.TranscriptionKind, Schema: "overgo/segmented-consumer-fixture/v1", Source: "Pinned LibriSpeech train.100 row; consumer integration only, not held-out quality",
		Dataset: data.Descriptor.ID, Split: split.Descriptor.ID,
		Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase, evaluation.TranscriptionCollapseWhitespace},
		Cases:         []evaluation.TranscriptionCase{{Name: f.clip.record.RowID, Group: f.clip.record.Split, Source: expected.Source, Reference: f.clip.record.Text, SampleCount: f.clip.record.Samples, SampleRate: f.clip.record.SampleRate}},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := evaluation.BindTranscription(compiled, evaluation.ExactAuthorities{ModelDefinition: metadata, RuntimeRecipe: f.definition.ID,
		CodeCommit: f.binding.CodeCommit, Environment: f.binding.Environment, Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident}})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(filepath.Dir(f.clip.storeRoot), "datasets", "librispeech_asr-clean-xet", "clean", f.clip.record.Split, f.clip.record.Shard)
	report, err := evaluation.EvaluateTranscriptionResources(t.Context(), f.store, compiled, plan, f.definition.Model,
		[]evaluation.TranscriptionResourceInput{{Name: f.clip.record.RowID, Reference: dataset.AudioPayloadReference{Path: path, Audio: expected.Source.Audio, Origin: f.origin}, Policy: f.policy}},
		evaluation.TranscriptionResourceOptions{TimedRuns: 1}, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.Runs != 1 || report.QualityScore.Utterances != 1 || len(report.Executions) != 1 || report.Summary.InferenceFailures != 0 {
		t.Fatalf("dataset consumer denominator differs: %+v", report.Summary)
	}
	run, err := runrecord.RequireExactRun(t.Context(), f.store, report.Executions[0].Run)
	if err != nil || run.Recipe != f.definition.ID || !slices.Contains(run.Inputs, data.Descriptor.ID) || !slices.Contains(run.Inputs, split.Descriptor.ID) {
		t.Fatalf("dataset run authority differs: %v", err)
	}
	got, err := speechrecognition.RequireTranscription(t.Context(), f.store, report.Executions[0].Output)
	if err != nil || got != expected {
		t.Fatalf("dataset consumer transcript differs: %v", err)
	}
	t.Logf("native evaluation executed the same recipe on 1 source-addressed training row; WER=%g CER=%g are integration observations, not held-out quality or promotion", report.QualityScore.WordErrorRate, report.QualityScore.CharacterErrorRate)
}
