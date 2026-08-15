package trainingdata

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/repodb"
)

func TestMaterializeIndexesMembershipAndDeduplicates(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	root := t.TempDir()
	firstPath := filepath.Join(root, "first.jsonl")
	secondPath := filepath.Join(root, "second.jsonl")
	if err := os.WriteFile(firstPath, []byte("alpha\nbeta\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(secondPath, []byte("alpha\ngamma\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	firstID, firstDescriptor := identifyFile(t, firstPath)
	secondID, secondDescriptor := identifyFile(t, secondPath)
	version, err := dataset.NewVersion([]dataset.Asset{
		{Name: "first", Artifact: firstID, Records: 2},
		{Name: "second", Artifact: secondID, Records: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	records := []dataset.Record{
		{ID: recordID(version.ID, "first", 0), Group: "all"},
		{ID: recordID(version.ID, "first", 1), Group: "all"},
		{ID: recordID(version.ID, "second", 0), Group: "all"},
		{ID: recordID(version.ID, "second", 1), Group: "all"},
	}
	plan, err := dataset.BuildGroupSplit(version.ID, records, 7, []dataset.SplitPartition{{Name: "train", Weight: 4}, {Name: "validation", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	var membership dataset.Membership
	for _, candidate := range plan.Memberships {
		if len(candidate.Records) != 0 {
			membership = candidate
		}
	}
	if !membership.ID.Valid() {
		t.Fatal("nonempty membership absent")
	}
	store, err := repodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := plan.PublicationBatch("split", nil)
	if err != nil {
		t.Fatal(err)
	}
	content, err := version.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, version.Lineage()...)
	batch.Artifacts = append(batch.Artifacts, firstDescriptor, secondDescriptor)
	batch.Locations = append(batch.Locations,
		artifact.LocationEvent{Location: artifact.Location{Artifact: firstID, Kind: artifact.LocationFile, Value: firstPath}, Action: artifact.LocationAdd},
		artifact.LocationEvent{Location: artifact.Location{Artifact: secondID, Kind: artifact.LocationFile, Value: secondPath}, Action: artifact.LocationAdd},
	)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	processor := profileID(t, "text-lines")
	materialized, err := Materialize(ctx, store, Authority{
		Dataset: version.ID, Split: membership.ID, Processors: []artifact.ID{processor}, Seed: 11,
		Signature: textSignature(),
	}, []ProcessorBinding{{Artifact: processor, Modalities: []recipecontract.Modality{recipecontract.ModalityText}, Process: Passthrough(RoleInput, recipecontract.ModalityText, "utf-8")}})
	if err != nil {
		t.Fatal(err)
	}
	defer materialized.Close()
	if materialized.Records() != 3 {
		t.Fatalf("records=%d, want 3 after exact dedup", materialized.Records())
	}
	stream, err := NewStream(materialized, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := NewBatcher(stream, BatchPolicy{Examples: 3, MicrobatchExamples: 2, DecodeWorkers: 3})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := batcher.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, len(decoded.Examples))
	for index, example := range decoded.Examples {
		got[index] = string(example.Values[0].Data)
	}
	wantSet := map[string]bool{"alpha": true, "beta": true, "gamma": true}
	for _, value := range got {
		if !wantSet[value] {
			t.Fatalf("unexpected decoded record %q", value)
		}
		delete(wantSet, value)
	}
	if len(wantSet) != 0 {
		t.Fatalf("missing decoded records: %v", wantSet)
	}
	micro := decoded.Microbatches()
	if len(micro) != 2 || len(micro[0]) != 2 || len(micro[1]) != 1 {
		t.Fatalf("microbatch geometry = %v", []int{len(micro[0]), len(micro[1])})
	}
}

func TestWeightedStreamResumeAndPacking(t *testing.T) {
	t.Parallel()
	processor := profileID(t, "processor")
	datasetID := profileKindID(t, artifact.KindDataset, "dataset")
	splitID := profileKindID(t, artifact.KindDatasetShard, "split")
	authority := Authority{Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processor}, Seed: 5, Signature: textSignature()}
	identity, err := identifyAuthority(authority)
	if err != nil {
		t.Fatal(err)
	}
	makeRecord := func(id, data string) recordRef {
		raw := []byte(data)
		return recordRef{id: id, group: id, processor: processor, length: int64(len(raw)), digest: sha256Sum(raw), inline: raw}
	}
	d := &Dataset{
		authority: authority, identity: identity,
		processors: map[artifact.ID]compiledProcessor{processor: {
			process:    Passthrough(RoleInput, recipecontract.ModalityText, "utf-8"),
			modalities: map[recipecontract.Modality]bool{recipecontract.ModalityText: true},
		}},
		members: []member{
			{identity: "heavy", weight: 3, records: []recordRef{makeRecord("h0", "aaaa"), makeRecord("h1", "bbbb")}},
			{identity: "light", weight: 1, records: []recordRef{makeRecord("l0", "cc")}},
		},
	}
	stream, err := NewStream(d, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := NewBatcher(stream, BatchPolicy{Examples: 4, MicrobatchExamples: 2, DecodeWorkers: 2, MaxBytes: 8})
	if err != nil {
		t.Fatal(err)
	}
	first, err := batcher.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Examples) != 2 {
		t.Fatalf("packed examples=%d, want 2", len(first.Examples))
	}
	state := first.State
	want, err := batcher.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := NewStream(d, &state)
	if err != nil {
		t.Fatal(err)
	}
	resumedBatcher, err := NewBatcher(resumed, BatchPolicy{Examples: 4, MicrobatchExamples: 2, DecodeWorkers: 1, MaxBytes: 8})
	if err != nil {
		t.Fatal(err)
	}
	got, err := resumedBatcher.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Examples, want.Examples) || got.State != want.State {
		t.Fatalf("resumed batch differs\ngot=%+v\nwant=%+v", got, want)
	}
}

func TestProcessorBindingsCoverTypedRunSignature(t *testing.T) {
	t.Parallel()
	image := profileID(t, "image-processor")
	text := profileID(t, "text-processor")
	authority := Authority{
		Dataset:    profileKindID(t, artifact.KindDataset, "multimodal"),
		Split:      profileKindID(t, artifact.KindDatasetShard, "multimodal-split"),
		Processors: []artifact.ID{image, text},
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{recipecontract.ModalityImage}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
	}
	bindings := []ProcessorBinding{
		{Artifact: image, Assets: []string{"image"}, Modalities: []recipecontract.Modality{recipecontract.ModalityImage}, Process: Passthrough(RoleInput, recipecontract.ModalityImage, "png")},
		{Artifact: text, Assets: []string{"caption"}, Modalities: []recipecontract.Modality{recipecontract.ModalityText}, Process: Passthrough(RoleTarget, recipecontract.ModalityText, "utf-8")},
	}
	if _, _, _, err := compileBindings(authority, bindings); err != nil {
		t.Fatal(err)
	}
	bindings[0].Modalities = []recipecontract.Modality{recipecontract.ModalityAudio}
	if _, _, _, err := compileBindings(authority, bindings); err == nil {
		t.Fatal("run signature accepted without an image processor")
	}
}

func TestBatcherRestoresStreamAfterDecodeFailure(t *testing.T) {
	t.Parallel()
	processor := profileID(t, "failing-processor")
	authority := Authority{
		Dataset:    profileKindID(t, artifact.KindDataset, "failing-dataset"),
		Split:      profileKindID(t, artifact.KindDatasetShard, "failing-split"),
		Processors: []artifact.ID{processor},
		Signature:  textSignature(),
	}
	d, err := MaterializeDocuments(authority, processor, []string{"first", "second"}, ProcessorBinding{
		Artifact:   processor,
		Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: func(context.Context, RawRecord) (Example, error) {
			return Example{}, errors.New("decode failed")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := NewStream(d, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := NewBatcher(stream, BatchPolicy{Examples: 2, DecodeWorkers: 2})
	if err != nil {
		t.Fatal(err)
	}
	before := stream.Snapshot()
	if _, err := batcher.Next(context.Background()); err == nil {
		t.Fatal("expected decode failure")
	}
	if after := stream.Snapshot(); after != before {
		t.Fatalf("stream advanced after failed batch: got %+v want %+v", after, before)
	}
}

func identifyFile(t *testing.T, path string) (artifact.ID, artifact.Descriptor) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	id, size, err := artifact.Identify(artifact.KindFile, file)
	if err != nil {
		t.Fatal(err)
	}
	return id, artifact.Descriptor{ID: id, Size: size, MediaType: "application/octet-stream"}
}

func profileID(t *testing.T, value string) artifact.ID {
	t.Helper()
	return profileKindID(t, artifact.KindProfile, value)
}

func profileKindID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func recordID(datasetID artifact.ID, asset string, index int) string {
	return fmt.Sprintf("%s/%s/%d", datasetID, asset, index)
}

func sha256Sum(data []byte) [sha256.Size]byte { return sha256.Sum256(data) }

func textSignature() recipecontract.ModalitySignature {
	return recipecontract.ModalitySignature{
		Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
	}
}
