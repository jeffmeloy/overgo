package trainingworkflow

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// writeTrainedFixtureBase writes a base holding an f16 tensor training
// leaves as it was, an f32 tensor training changes and a tensor training
// never holds, and returns its path.
func writeTrainedFixtureBase(t *testing.T, root string) string {
	t.Helper()
	base := filepath.Join(root, "base.gguf")
	half := []byte{0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44} // 1, 2, 3, 4
	if err := gguf.WriteFileExclusive(base, []gguf.Metadata{testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "fixture")}, []gguf.TensorData{
		testutil.GGUFTensorF32("untrained.weight", []uint64{2}, 2),
		{Name: "unchanged.weight", Shape: []uint64{4}, Type: gguf.DTypeF16, Data: bytes.NewReader(half)},
		testutil.GGUFTensorF32("changed.weight", []uint64{2}, 1),
	}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return base
}

// lookupTrained is the trained lookup of a run that trained the named
// tensors to the given values.
func lookupTrained(trained map[string][]float32) func(string) ([]float32, bool) {
	return func(name string) ([]float32, bool) { values, ok := trained[name]; return values, ok }
}

// readStored reads a tensor's directory entry and stored bytes from file.
func readStored(t *testing.T, file *gguf.File, name string) (gguf.TensorInfo, []byte) {
	t.Helper()
	info, ok := file.Tensor(name)
	data := make([]byte, info.Size)
	if !ok || file.ReadTensorData(info, data) != nil {
		t.Fatalf("tensor %s is absent", name)
	}
	return info, data
}

// TestTrainedModelKeepsUnchangedTensorsAsStored writes a trained model over
// a base holding an f16 tensor training left as it was, an f32 tensor
// training changed and a tensor training never held: the model opens as
// one split in two, its first shard holding the base's metadata and the
// tensors training held, its second the tensor training never held. Only
// the changed tensor is rewritten, in f32; the others keep the base's
// stored bytes and type. A trained tensor set the base does not hold
// refuses.
func TestTrainedModelKeepsUnchangedTensorsAsStored(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	base := writeTrainedFixtureBase(t, root)
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	trained := map[string][]float32{"unchanged.weight": {1, 2, 3, 4}, "changed.weight": {5, 6}}
	directory := filepath.Join(root, "trained")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteTrainedModel(t.Context(), store, base, directory, len(trained), lookupTrained(trained)); err != nil {
		t.Fatal(err)
	}
	original, err := gguf.Open(base)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	written, err := gguf.Open(filepath.Join(directory, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	defer written.Close()
	if paths := written.SourcePaths(); len(paths) != 2 || filepath.Base(paths[1]) != trainingprogram.CheckpointFrozenShard {
		t.Fatalf("trained model shards = %v, want %s then %s", paths, trainingprogram.CheckpointModel, trainingprogram.CheckpointFrozenShard)
	}
	if architecture, ok := written.MetadataValue("general.architecture"); !ok || architecture.Data != "fixture" {
		t.Fatalf("trained model architecture = %v, want the base's", architecture.Data)
	}
	for name, shard := range map[string]uint16{"unchanged.weight": 0, "untrained.weight": 1} {
		baseInfo, baseData := readStored(t, original, name)
		writtenInfo, writtenData := readStored(t, written, name)
		if writtenInfo.Type != baseInfo.Type || !bytes.Equal(writtenData, baseData) || writtenInfo.Shard != shard {
			t.Fatalf("%s = %s %x in shard %d, want the base's %s %x in shard %d", name, writtenInfo.Type, writtenData, writtenInfo.Shard, baseInfo.Type, baseData, shard)
		}
	}
	changed, changedData := readStored(t, written, "changed.weight")
	if changed.Type != gguf.DTypeF32 || changed.Shard != 0 || !slices.Equal(changedData, []byte{0, 0, 0xa0, 0x40, 0, 0, 0xc0, 0x40}) {
		t.Fatalf("changed.weight = %s %x in shard %d, want f32 5 and 6 in the first shard", changed.Type, changedData, changed.Shard)
	}
	trained["absent.weight"] = []float32{1}
	refused := filepath.Join(root, "refused")
	if err := os.Mkdir(refused, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteTrainedModel(t.Context(), store, base, refused, len(trained), lookupTrained(trained)); err == nil {
		t.Fatal("a trained tensor the base does not hold was accepted")
	}
}

// TestTrainedModelSharesTheFrozenShard publishes three trained models over
// one base: the first writes the shard of tensors training never held, the
// second, whose store records the first as a registered model, links that
// shard instead of writing it, and the third, whose store records the
// shard's identity at a file holding other bytes, writes its own. Every
// run writes its own trained shard.
func TestTrainedModelSharesTheFrozenShard(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	base := writeTrainedFixtureBase(t, root)
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publish := func(name string, store artifact.Reader, changed float32) string {
		directory := filepath.Join(root, name)
		if err := os.Mkdir(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		trained := map[string][]float32{"unchanged.weight": {1, 2, 3, 4}, "changed.weight": {changed, changed}}
		if err := WriteTrainedModel(ctx, store, base, directory, len(trained), lookupTrained(trained)); err != nil {
			t.Fatal(err)
		}
		return directory
	}
	same := func(left, right string) bool {
		leftInfo, leftErr := os.Stat(left)
		rightInfo, rightErr := os.Stat(right)
		if leftErr != nil || rightErr != nil {
			t.Fatal(leftErr, rightErr)
		}
		return os.SameFile(leftInfo, rightInfo)
	}

	first := publish("first", store, 5)
	file, err := gguf.Open(filepath.Join(first, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if closeErr := file.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	batch, err := inventory.Batch("fixture/first-trained-model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	second := publish("second", store, 7)
	if !same(filepath.Join(first, trainingprogram.CheckpointFrozenShard), filepath.Join(second, trainingprogram.CheckpointFrozenShard)) {
		t.Fatal("the second run wrote the frozen shard the store records, want it linked")
	}
	if same(filepath.Join(first, trainingprogram.CheckpointModel), filepath.Join(second, trainingprogram.CheckpointModel)) {
		t.Fatal("the second run linked the first run's trained shard")
	}

	// A store whose only copy of the frozen shard is a file holding other
	// bytes records no copy.
	frozen := inventory.Components[1]
	stale, err := overgodb.Open(filepath.Join(root, "stale"))
	if err != nil {
		t.Fatal(err)
	}
	defer stale.Close()
	other, err := artifact.CanonicalLocalLocation(frozen.ID, artifact.LocationFile, filepath.Join(first, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, stale, artifact.Batch{
		Key: "fixture/stale-frozen-shard", Artifacts: []artifact.Descriptor{frozen},
		Locations: []artifact.LocationEvent{{Location: other, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	third := publish("third", stale, 9)
	if same(other.Value, filepath.Join(third, trainingprogram.CheckpointFrozenShard)) {
		t.Fatal("the third run linked a recorded copy that holds other bytes")
	}
	want, err := os.ReadFile(filepath.Join(first, trainingprogram.CheckpointFrozenShard))
	if err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(third, trainingprogram.CheckpointFrozenShard)); err != nil || !bytes.Equal(got, want) {
		t.Fatalf("the third run's frozen shard = (%d bytes, %v), want the first run's %d bytes", len(got), err, len(want))
	}
}
