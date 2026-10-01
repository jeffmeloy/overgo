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
	"overgo/internal/gguf"
	"overgo/internal/modelintake"
	"overgo/internal/quant"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// WriteTrainedModel writes at path the model base's GGUF becomes once
// trained: base's metadata and tensor order, every tensor training changed
// in f32 at base's extents, and every other tensor as base's own stored
// bytes, so the server opens it as it opens base. trained looks a tensor up
// by base's name for it and must find tensors of base's; one whose trained
// values equal base's dequantized values is unchanged and stays as stored.
func WriteTrainedModel(base, path string, tensors int, trained func(name string) ([]float32, bool)) error {
	file, err := gguf.Open(base)
	if err != nil {
		return err
	}
	defer file.Close()
	data := make([]gguf.TensorData, len(file.Tensors))
	found := 0
	for index, info := range file.Tensors {
		data[index] = gguf.TensorData{Name: info.Name, Shape: info.Extents(), Type: info.Type, Data: io.NewSectionReader(storedTensor{file, info}, 0, int64(info.Size))}
		values, ok := trained(info.Name)
		if !ok {
			continue
		}
		found++
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
			data[index].Type = gguf.DTypeF32
			data[index].Data = bytes.NewReader(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*binaryschema.Uint32Bytes))
		}
	}
	if found != tensors {
		return fmt.Errorf("training workflow: %s names %d of the %d tensors training changed", base, found, tensors)
	}
	// A split base's shard bookkeeping does not describe the one file
	// written.
	metadata := slices.DeleteFunc(slices.Clone(file.Metadata), func(entry gguf.Metadata) bool { return strings.HasPrefix(entry.Key, "split.") })
	return gguf.WriteFileExclusive(path, metadata, data, gguf.WriteOptions{Version: file.Version, Alignment: file.Alignment})
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
	for _, edge := range batch.Lineage {
		if slices.ContainsFunc(batch.Artifacts, func(declared artifact.Descriptor) bool { return declared.ID == edge.Parent }) ||
			slices.ContainsFunc(batch.Contents, func(content artifact.Content) bool { return content.Descriptor.ID == edge.Parent }) {
			continue
		}
		if _, held, err := store.Artifact(ctx, edge.Parent); err != nil {
			return modelintake.Candidate{}, err
		} else if !held {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: edge.Parent})
		}
	}
	if err := PublishTrainingEvidence(ctx, store, TrainingEvidencePublication{Batch: batch}); err != nil {
		return modelintake.Candidate{}, err
	}
	if _, err := os.Stat(filepath.Join(directory, trainingprogram.CheckpointModel)); errors.Is(err, os.ErrNotExist) {
		return modelintake.Candidate{}, nil
	}
	return publishTrainedModel(ctx, store, directory, checkpoint)
}

// publishTrainedModel registers the trained model a recorded checkpoint
// holds at directory as an inactive inference candidate and records its
// lineage: the base model it trained from, the dataset, program and run
// plan the checkpoint names, and the checkpoint itself.
func publishTrainedModel(ctx context.Context, store artifact.Repository, directory string, checkpoint trainingprogram.Checkpoint) (modelintake.Candidate, error) {
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, filepath.Join(directory, trainingprogram.CheckpointModel), modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
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
