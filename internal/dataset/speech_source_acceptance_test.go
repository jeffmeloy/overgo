// Package dataset_test verifies speech materialization through public dataset
// and universal training-stream contracts on pinned source artifacts.
package dataset_test

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

// Fixed goldens are compile-time test inputs.
//
//go:embed testdata/audio_inspection_policy.json
var speechInspectionPolicy []byte

//go:embed testdata/nullable_parquet.json
var speechNullableGolden []byte

// TestSpeechMaterializationContracts holds speech materialization to its
// contracts on embedded fixtures and scratch stores: persistent references,
// rejected-row accounting, signal admission and failure rollback. It reads
// no data root, so the landing runs it; the pinned corpus is the cycle's.
func TestSpeechMaterializationContracts(t *testing.T) {
	t.Run("persistent-references-and-rejected-row-accounting", testSpeechReferenceRejections)
	t.Run("signal-admission-and-failure-rollback", testSpeechSignalAdmission)
}

func testSpeechReferenceRejections(t *testing.T) {
	var fixture struct {
		Parquet string `json:"parquet_base64"`
	}
	if err := json.Unmarshal(speechNullableGolden, &fixture); err != nil {
		t.Fatal(err)
	}
	encoded, err := base64.StdEncoding.DecodeString(fixture.Parquet)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	path := filepath.Join(root, "nullable.parquet")
	if err := os.WriteFile(path, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	registered, err := dataset.RegisterDirectoryDataset(t.Context(), store, "nullable", root)
	if err != nil {
		t.Fatal(err)
	}
	spec := dataset.SpeechMaterializationSpec{Source: registered.Dataset, AudioColumn: "audio.bytes", TargetColumn: "text"}
	result, err := dataset.MaterializeSpeechDataset(t.Context(), store, root, spec, uint64(len(encoded)))
	if err != nil || result.Rows != 6 || result.Records != 3 || result.Rejected != 3 || result.Shards != 1 {
		t.Fatalf("null/empty rows must retain physical denominators: %+v %v", result, err)
	}
	version, found, err := dataset.Load(t.Context(), store, result.Dataset)
	if err != nil || !found || len(version.Assets) != 1 {
		t.Fatal("derived reference asset absent", err)
	}
	content, found, err := artifact.ReadContent(t.Context(), store, version.Assets[0].Artifact)
	if err != nil || !found {
		t.Fatal("inline reference asset absent", err)
	}
	lines := strings.Split(strings.TrimSpace(string(content.Data)), "\n")
	for index, want := range []uint64{0, 4, 5} {
		var record dataset.SpeechRecord
		if err := json.Unmarshal([]byte(lines[index]), &record); err != nil || record.Origin.Row == nil || *record.Origin.Row != want {
			t.Fatalf("row %d was renumbered: %+v %v", want, record, err)
		}
	}
	before, _ := store.Head()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := dataset.MaterializeSpeechDataset(ctx, store, root, spec, uint64(len(encoded))); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	if _, err := dataset.MaterializeSpeechDataset(t.Context(), store, root, spec, 1); err == nil {
		t.Fatal("container budget bypassed")
	}
	encoded[len("PAR1")] ^= 1
	if err := os.WriteFile(path, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := dataset.MaterializeSpeechDataset(t.Context(), store, root, spec, uint64(len(encoded))); err == nil {
		t.Fatal("changed source admitted under old inventory")
	}
	after, _ := store.Head()
	if before != after {
		t.Fatal("failed or cancelled materialization published partial authority")
	}
}

func testSpeechSignalAdmission(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var policy dataset.AudioInspectionPolicy
	if err := json.Unmarshal(speechInspectionPolicy, &policy); err != nil {
		t.Fatal(err)
	}
	source, err := dataset.NewAudioPayloadReader(policy.MaximumEncodedBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	processor, err := trainingdata.AudioTextProcessor(store, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	accepted, rejected := 0, 0
	for _, tc := range []struct {
		name   string
		data   []byte
		accept bool
	}{
		{"valid", testutil.MonoPCM16WAV(16000, []int16{8192, -8192}), true},
		{"silent", testutil.MonoPCM16WAV(16000, []int16{0, 0}), false},
		{"corrupt", []byte("RIFF truncated payload"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id, _ := artifact.IdentifyBytes(artifact.KindFile, tc.data)
			path := filepath.Join(t.TempDir(), tc.name+".wav")
			if err := os.WriteFile(path, tc.data, 0600); err != nil {
				t.Fatal(err)
			}
			batch := artifact.Batch{Key: "signal/" + tc.name, Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(tc.data))}}, Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: id, Kind: artifact.LocationFile, Value: path}, Action: artifact.LocationAdd}}}
			if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
				t.Fatal(err)
			}
			origin := dataset.AudioPayloadOrigin{Container: id}
			data, err := json.Marshal(dataset.SpeechRecord{Audio: id, Origin: origin, Target: "isolated target"})
			if err != nil {
				t.Fatal(err)
			}
			profile, _ := artifact.JSONID(artifact.KindProfile, policy)
			authority := trainingdata.Authority{Dataset: testutil.ArtifactID(t, artifact.KindDataset, tc.name), Split: testutil.ArtifactID(t, artifact.KindDatasetShard, tc.name), Processors: []artifact.ID{profile}, Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityAudio}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}}}
			materialized, err := trainingdata.MaterializeRecords(authority, profile, []trainingdata.RawRecord{{ID: tc.name, Data: data}}, trainingdata.ProcessorBinding{Artifact: profile, Modalities: []recipecontract.Modality{recipecontract.ModalityAudio, recipecontract.ModalityText}, Process: processor})
			if err != nil {
				t.Fatal(err)
			}
			defer materialized.Close()
			stream, err := trainingdata.NewStream(materialized, nil)
			if err != nil {
				t.Fatal(err)
			}
			batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, DecodeWorkers: 1})
			if err != nil {
				t.Fatal(err)
			}
			before := stream.Snapshot()
			result, err := batcher.Next(t.Context())
			if tc.accept {
				if err != nil || len(result.Examples) != 1 {
					t.Fatalf("valid signal refused: %v", err)
				}
				accepted++
			} else {
				if err == nil || stream.Snapshot() != before {
					t.Fatalf("refusal advanced stream: %v", err)
				}
				rejected++
			}
			inspection, inspectErr := dataset.InspectAudio(t.Context(), store, tc.data, origin, policy)
			if inspectErr != nil || !inspection.DecisionID.Valid() {
				t.Fatal("durable admission decision absent", inspectErr)
			}
			if !tc.accept && !strings.Contains(err.Error(), inspection.DecisionID.String()) {
				t.Fatalf("refusal lost decision identity: %v", err)
			}
		})
	}
	if accepted != 1 || rejected != 2 {
		t.Fatalf("signal denominators %d/%d", accepted, rejected)
	}
}
