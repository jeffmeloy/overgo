package trainingworkflow

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unsafe"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/densecausal"
	"overgo/internal/gguf"
	"overgo/internal/modelintake"
	"overgo/internal/quant"
	"overgo/internal/recipe"
	"overgo/internal/tensorcatalog"
	"overgo/internal/trainingprogram"
)

// WriteTrainedModel writes in directory the model base's GGUF becomes once
// trained, split in two so the server opens it as it opens base. The first
// shard, trainingprogram.CheckpointModel, holds base's metadata and, in
// base's order, every tensor training held: in f32 at base's extents when
// training changed it, else as base's own stored bytes. The second,
// trainingprogram.CheckpointFrozenShard, holds every tensor training never
// held as base's stored bytes: the same bytes for every run over base, so
// when store records a live copy of them that copy is linked rather than
// written again. trained looks a tensor up by base's name for it and must
// find tensors of base's; one whose trained values equal base's dequantized
// values is unchanged.
func WriteTrainedModel(ctx context.Context, store artifact.Reader, base, directory string, tensors int, trained func(name string) ([]float32, bool)) error {
	file, err := gguf.Open(base)
	if err != nil {
		return err
	}
	defer file.Close()
	asStored := func(info gguf.TensorInfo) gguf.TensorData {
		return gguf.TensorData{Name: info.Name, Shape: info.Extents(), Type: info.Type, Data: io.NewSectionReader(storedTensor{file, info}, 0, int64(info.Size))}
	}
	var held []gguf.TensorData
	var frozen []gguf.TensorInfo
	for _, info := range file.Tensors {
		values, ok := trained(info.Name)
		if !ok {
			frozen = append(frozen, info)
			continue
		}
		data := asStored(info)
		elements, err := info.ElementCount()
		if err != nil || elements != uint64(len(values)) {
			return fmt.Errorf("training workflow: trained %s differs from %s's shape: %w", info.Name, base, err)
		}
		stored := make([]byte, info.Size)
		if err := file.ReadTensorData(info, stored); err != nil {
			return err
		}
		before, err := quant.Dequantize(info.Type, stored, elements)
		if err != nil {
			return err
		}
		if !slices.Equal(before, values) {
			data.Type = gguf.DTypeF32
			data.Data = bytes.NewReader(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*binaryschema.Uint32Bytes))
		}
		held = append(held, data)
	}
	if len(held) != tensors {
		return fmt.Errorf("training workflow: %s names %d of the %d tensors training changed", base, len(held), tensors)
	}
	// The frozen shard streams from fresh readers each time it is read:
	// once to identify it, and again when it is written.
	frozenData := func() []gguf.TensorData {
		data := make([]gguf.TensorData, len(frozen))
		for index, info := range frozen {
			data[index] = asStored(info)
		}
		return data
	}
	options := gguf.WriteOptions{Version: file.Version, Alignment: file.Alignment}
	total := len(file.Tensors)
	if err := writeShared(ctx, store, filepath.Join(directory, trainingprogram.CheckpointFrozenShard),
		shardMetadata(nil, frozenShard, total), frozenData, options); err != nil {
		return err
	}
	return gguf.WriteFileExclusive(filepath.Join(directory, trainingprogram.CheckpointModel),
		shardMetadata(file.Metadata, trainedShard, total), held, options)
}

// trainedGGUFTensors is the lookup WriteTrainedModel takes for a model trained over
// a GGUF base: the weight the model holds under the HF name of the base's tensor, in
// the layout the base stores it (a llama query or key projection is reordered at load
// and put back here, or the published file would read back reordered twice).
func trainedGGUFTensors(model *densecausal.Model) func(name string) ([]float32, bool) {
	return func(name string) ([]float32, bool) {
		trained, named := tensorcatalog.HFDenseTensorName(name)
		values, found := model.Weights[trained]
		return model.StoredRows(trained, values), named && found
	}
}

// trainedShard and frozenShard index the shards of a trained model, which
// is split in trainedModelShards: trainingprogram.CheckpointModel, then
// trainingprogram.CheckpointFrozenShard.
const (
	trainedShard uint16 = iota
	frozenShard
	trainedModelShards
)

// The split bookkeeping every shard of a split GGUF carries, under the keys
// the GGUF split convention names and gguf.Open reads: the shard's index,
// how many shards there are, and how many tensors they hold together.
const (
	splitIndexKey   = "split.no"
	splitCountKey   = "split.count"
	splitTensorsKey = "split.tensors.count"
)

// shardMetadata is the metadata shard index of a trained model over tensors
// carries: metadata, which only the first shard carries, without the split
// bookkeeping a split base held, then the bookkeeping that names the shard.
func shardMetadata(metadata []gguf.Metadata, index uint16, tensors int) []gguf.Metadata {
	shard := slices.DeleteFunc(slices.Clone(metadata), func(entry gguf.Metadata) bool { return strings.HasPrefix(entry.Key, "split.") })
	return append(shard,
		gguf.Metadata{Key: splitIndexKey, Value: gguf.Value{Type: gguf.ValueTypeUint16, Data: index}},
		gguf.Metadata{Key: splitCountKey, Value: gguf.Value{Type: gguf.ValueTypeUint16, Data: trainedModelShards}},
		gguf.Metadata{Key: splitTensorsKey, Value: gguf.Value{Type: gguf.ValueTypeInt32, Data: int32(tensors)}},
	)
}

// writeShared writes at path the GGUF shard of metadata and the tensors
// data streams, unless store records a live copy of the same bytes: the
// shard is identified first, by the digest its registration records, and
// a recorded copy that still holds that digest is hard-linked at path. A
// copy that is gone, on another volume or changed is no copy, and the
// shard is written.
func writeShared(ctx context.Context, store artifact.Reader, path string, metadata []gguf.Metadata, data func() []gguf.TensorData, options gguf.WriteOptions) error {
	reader, writer := io.Pipe()
	go func() { writer.CloseWithError(gguf.Write(writer, metadata, data(), options)) }()
	id, _, err := artifact.Identify(artifact.KindTensorSet, reader)
	// Closing the read end releases the writer if hashing stopped early.
	reader.CloseWithError(err)
	if err != nil {
		return err
	}
	if recorded, err := artifact.AvailablePath(ctx, store, id, artifact.LocationFile); err == nil && os.Link(recorded, path) == nil {
		linked, err := os.Open(path)
		if err != nil {
			return err
		}
		held, _, identifyErr := artifact.Identify(artifact.KindTensorSet, linked)
		if err := errors.Join(identifyErr, linked.Close()); err != nil || held == id {
			return err
		}
		if err := os.Remove(path); err != nil {
			return err
		}
	}
	return gguf.WriteFileExclusive(path, metadata, data(), options)
}

// storedTensor reads one base tensor's stored bytes in place.
type storedTensor struct {
	file *gguf.File
	info gguf.TensorInfo
}

// ReadAt reads the tensor's stored bytes from offset into data.
func (t storedTensor) ReadAt(data []byte, offset int64) (int, error) {
	if err := t.file.ReadTensorRange(t.info, uint64(offset), data); err != nil {
		return 0, err
	}
	return len(data), nil
}

// RecordCheckpoint records the checkpoint a run published at directory as
// the run's training evidence: its document, resume sidecar, weights and
// locations, beside each parent its lineage names that the store does not
// hold yet. A checkpoint holding a trained model, as a GGUF base's run
// publishes, also registers that model as an inactive inference candidate,
// the model the server, validate and the scorecard open, which returns; a
// checkpoint of trained tensors returns no candidate.
func RecordCheckpoint(ctx context.Context, store artifact.Repository, directory string, checkpoint trainingprogram.Checkpoint) (modelintake.Candidate, error) {
	directory, err := filepath.Abs(directory)
	if err != nil {
		return modelintake.Candidate{}, err
	}
	batch, err := checkpoint.Batch("training/evidence/checkpoint/"+checkpoint.ID().String()+"/"+directory, directory)
	if err != nil {
		return modelintake.Candidate{}, err
	}
	if err := publishWithParents(ctx, store, batch); err != nil {
		return modelintake.Candidate{}, err
	}
	weights, err := trainingprogram.CheckpointWeightsPath(directory)
	if err != nil {
		return modelintake.Candidate{}, err
	}
	if !isGGUFModelInput(weights) {
		return modelintake.Candidate{}, nil
	}
	return publishTrainedModel(ctx, store, weights, checkpoint)
}

// publishWithParents publishes a checkpoint's batch as training evidence,
// declaring first each parent its lineage names that neither the batch nor
// the store holds: the store refuses an edge to an artifact it does not know.
func publishWithParents(ctx context.Context, store artifact.Repository, batch artifact.Batch) error {
	for _, edge := range batch.Lineage {
		if slices.ContainsFunc(batch.Artifacts, func(declared artifact.Descriptor) bool { return declared.ID == edge.Parent }) ||
			slices.ContainsFunc(batch.Contents, func(content artifact.Content) bool { return content.Descriptor.ID == edge.Parent }) ||
			slices.ContainsFunc(batch.Manifests, func(manifest artifact.Manifest) bool { return manifest.ID == edge.Parent }) {
			continue
		}
		if _, held, err := store.Artifact(ctx, edge.Parent); err != nil {
			return err
		} else if !held {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: edge.Parent})
		}
	}
	return PublishTrainingEvidence(ctx, store, TrainingEvidencePublication{Batch: batch})
}

// RecordCheckpointDocument records a checkpoint whose files do not stay: an
// intermediate product of a larger run, removed once the run's own record is
// kept, or a proof's, which keeps no file at all. The store is given the
// checkpoint's document and lineage, which a session's record, a verification
// and a verdict name, and no location, weights, trained model or candidate:
// a catalog must not list files that will be gone, and what is read back
// from an intermediate checkpoint is read from its directory while it stands.
//
// A checkpoint whose document the store holds is recorded already, with its
// lineage, and nothing is published: the same checkpoint trained again, as a
// run begun again after its checkpoints were removed trains it, would name
// the same key with other content, because the parents the first batch had to
// declare are held by then.
func RecordCheckpointDocument(ctx context.Context, store artifact.Repository, checkpoint trainingprogram.Checkpoint) error {
	if _, held, err := artifact.ReadContent(ctx, store, checkpoint.ID()); err != nil || held {
		return err
	}
	content, err := checkpoint.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch("training/evidence/checkpoint/"+checkpoint.ID().String()+"/throwaway", []artifact.Content{content}, checkpoint.ArtifactLineage(), nil)
	if err != nil {
		return err
	}
	return publishWithParents(ctx, store, batch)
}

// publishTrainedModel registers the trained model a recorded checkpoint
// holds at directory as an inactive inference candidate and records its
// lineage: the base model it trained from, the dataset, program and run
// plan the checkpoint names, and the checkpoint itself.
func publishTrainedModel(ctx context.Context, store artifact.Repository, weights string, checkpoint trainingprogram.Checkpoint) (modelintake.Candidate, error) {
	directory := filepath.Dir(weights)
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, weights, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		return modelintake.Candidate{}, err
	}
	model := candidate.Inventory.Manifest.ID
	batch := artifact.Batch{
		Key:     "training/model/" + checkpoint.ID().String() + "/" + directory,
		Lineage: []artifact.Lineage{{Child: model, Parent: checkpoint.ID(), Relation: artifact.RelationProducedBy}},
	}
	// A run that trained bytes the store registered from another run's
	// directory adds where this run's copy is.
	_, registered, err := store.Artifact(ctx, candidate.Resolved.Document.ID)
	if err != nil {
		return modelintake.Candidate{}, err
	}
	if registered {
		for _, location := range candidate.Inventory.Locations {
			batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
	} else if err := modelintake.RegisterCandidate(ctx, store, candidate); err != nil {
		return modelintake.Candidate{}, err
	}
	for _, parent := range checkpoint.Lineage {
		batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: model, Parent: parent.Artifact, Relation: parent.Relation})
	}
	_, err = artifact.Publish(ctx, store, batch)
	return candidate, err
}
