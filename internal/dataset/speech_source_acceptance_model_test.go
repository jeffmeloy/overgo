//go:build modeltest

package dataset_test

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

// The decode goldens are compile-time test inputs; only the corpus itself is
// opened from the configured external data root.
//
//go:embed testdata/librispeech_decode_expectations.json
var speechDecodeGolden []byte

// The real corpus exercises source traversal and the universal stream, not
// model inference or parameter training.
func TestSpeechMaterializationAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	if reference := os.Getenv("OVERGO_AUDIO_REFERENCE_STORE"); reference != "" {
		roots.Datasets = filepath.Join(filepath.Dir(reference), "datasets")
	}
	path := filepath.Join(roots.Datasets, "librispeech_asr-clean-xet", "clean", "validation", "0000.parquet")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE pinned speech corpus: %v", err)
	}
	var golden struct {
		Container    string `json:"container_sha256"`
		Observations []struct {
			Encoded string `json:"encoded_sha256"`
			Decoded string `json:"decoded_sha256"`
			Frames  uint64 `json:"frames"`
		} `json:"observations"`
	}
	if err := json.Unmarshal(speechDecodeGolden, &golden); err != nil {
		t.Fatal(err)
	}
	var policy dataset.AudioInspectionPolicy
	if err := json.Unmarshal(speechInspectionPolicy, &policy); err != nil {
		t.Fatal(err)
	}
	// This fixture's explicit CPU workspace ceiling is 64 MiB, independently
	// of per-sample limits. It is not a process-peak measurement or default.
	const workspaceBytes = 64 << 20
	rows, err := dataset.OpenParquetRows(t.Context(), path, []string{"audio.bytes", "text", "id"}, workspaceBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if rows.Rows() != 2703 || len(golden.Observations) != 4 {
		t.Fatal("pinned corpus cardinality differs")
	}
	container, err := artifact.ParseID("dataset-shard:sha256:" + golden.Container)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	wantTexts := []string{
		"HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT HIS WIFE'S ACTION THREATENED TO CAST UPON HIS ENTIRE FUTURE",
		"HE WOULD HAVE TO PAY HER THE MONEY WHICH SHE WOULD NOW REGULARLY DEMAND OR THERE WOULD BE TROUBLE IT DID NOT MATTER WHAT HE DID",
		"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF POINTS OF HIS SITUATION",
		"HE ALSO THOUGHT OF HIS MANAGERIAL POSITION",
	}
	for index, observation := range golden.Observations {
		values, err := rows.Read(t.Context(), uint64(index))
		if err != nil {
			t.Fatal(err)
		}
		if values["audio.bytes"] == nil || values["text"] == nil || *values["text"] != wantTexts[index] || values["id"] == nil || *values["id"] != fmt.Sprintf("2277-149896-%04d", index) {
			t.Fatalf("source row %d differs", index)
		}
		digest := sha256.Sum256([]byte(*values["audio.bytes"]))
		if hex.EncodeToString(digest[:]) != observation.Encoded {
			t.Fatalf("encoded row %d differs", index)
		}
	}
	registered, err := dataset.RegisterDirectoryDataset(t.Context(), store, "pinned-validation", filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if registered.Dataset.String() != "dataset:sha256:39b4aa4872600afff4be0633dc066adfe5ae36ac17ade4a6ac4a16e26852b2e3" {
		t.Fatalf("master validation inventory identity differs: %s", registered.Dataset)
	}
	spec := dataset.SpeechMaterializationSpec{Source: registered.Dataset, AudioColumn: "audio.bytes", TargetColumn: "text", Limit: uint64(len(golden.Observations))}
	existing := artifact.Descriptor{ID: container, Size: uint64(info.Size()), MediaType: "application/vnd.apache.parquet", Schema: "fixture/existing-container/v1"}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "speech/existing-descriptor", Artifacts: []artifact.Descriptor{existing}}); err != nil {
		t.Fatal(err)
	}
	publication, err := dataset.MaterializeSpeechDataset(t.Context(), store, filepath.Dir(path), spec, workspaceBytes)
	if err != nil {
		t.Fatal(err)
	}
	physical, found, err := store.Artifact(t.Context(), container)
	if err != nil || !found || physical != existing {
		t.Fatalf("physical shard descriptor: %v", err)
	}
	if publication.Rows != spec.Limit || publication.Records != spec.Limit || publication.Rejected != 0 || publication.Shards != 1 {
		t.Fatalf("materialization denominator: %+v", publication)
	}
	head, _ := store.Head()
	replay, err := dataset.MaterializeSpeechDataset(t.Context(), store, filepath.Dir(path), spec, workspaceBytes)
	if err != nil || replay != publication {
		t.Fatalf("materialization replay: %+v %v", replay, err)
	}
	after, _ := store.Head()
	if head != after {
		t.Fatal("exact materialization replay appended duplicate publication")
	}
	version, found, err := dataset.Load(t.Context(), store, publication.Dataset)
	if err != nil || !found || len(version.Assets) != 1 {
		t.Fatalf("row-addressed dataset: %v", err)
	}
	records := make([]dataset.Record, len(golden.Observations))
	for index := range records {
		records[index] = dataset.Record{ID: fmt.Sprintf("%s/%s/%d", version.ID, version.Assets[0].Name, index), Group: "2277-149896"}
	}
	split, err := dataset.BuildGroupSplit(version.ID, records, 0, []dataset.SplitPartition{{Name: "fixture", Weight: 1}, {Name: "reserved", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	var membership dataset.Membership
	for _, candidate := range split.Memberships {
		if len(candidate.Records) != 0 {
			membership = candidate
		}
	}
	batch, err := split.PublicationBatch("speech/source/fixture", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	source, err := dataset.NewAudioPayloadReader(workspaceBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	processor, err := trainingdata.AudioTextProcessor(store, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	processorID, err := artifact.JSONID(artifact.KindProfile, policy)
	if err != nil {
		t.Fatal(err)
	}
	materialized, err := trainingdata.Materialize(t.Context(), store, trainingdata.Authority{
		Dataset: version.ID, Split: membership.ID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityAudio}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
	}, []trainingdata.ProcessorBinding{{Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityAudio, recipecontract.ModalityText}, Process: processor}})
	if err != nil {
		t.Fatal(err)
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		t.Fatal(err)
	}
	batchPolicy := trainingdata.BatchPolicy{Examples: 1, DecodeWorkers: 1, MaxBytes: workspaceBytes}
	batcher, err := trainingdata.NewBatcher(stream, batchPolicy)
	if err != nil {
		t.Fatal(err)
	}
	var resume trainingdata.StreamState
	var expected trainingdata.Batch
	for index, observation := range golden.Observations {
		result, err := batcher.Next(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		example := result.Examples[0]
		if len(example.Values) != 2 || example.Values[0].Role != trainingdata.RoleInput || example.Values[1].Role != trainingdata.RoleTarget || string(example.Values[1].Data) != wantTexts[index] {
			t.Fatalf("target isolation row %d", index)
		}
		samples, rate, err := trainingdata.Audio(example.Values[0])
		if err != nil || rate != 16000 || uint64(len(samples)) != observation.Frames {
			t.Fatalf("PCM row %d: %v", index, err)
		}
		digest := sha256.Sum256(example.Values[0].Data)
		if hex.EncodeToString(digest[:]) != observation.Decoded {
			t.Fatalf("decoded row %d differs", index)
		}
		if index == 1 {
			resume = result.State
		}
		if index == 2 {
			expected = result
		}
	}
	resumed, err := trainingdata.NewStream(materialized, &resume)
	if err != nil {
		t.Fatal(err)
	}
	resumedBatcher, err := trainingdata.NewBatcher(resumed, batchPolicy)
	if err != nil {
		t.Fatal(err)
	}
	got, err := resumedBatcher.Next(t.Context())
	if err != nil || !reflect.DeepEqual(got, expected) {
		t.Fatalf("exact next-record resume differs: %v", err)
	}
	before := resumed.Snapshot()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := resumedBatcher.Next(ctx); !errors.Is(err, context.Canceled) || resumed.Snapshot() != before {
		t.Fatalf("cancellation advanced stream: %v", err)
	}
	t.Logf("CPU source: %d/%d pinned payloads and decoded hashes; exact next-record resume and target isolation passed; corpus rows=%d, source workspace cap=%d; no model inference, gradients, GPU or process-peak measurement", len(golden.Observations), len(golden.Observations), rows.Rows(), workspaceBytes)
}
