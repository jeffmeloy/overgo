package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/trainingdata"
)

type trainingDataPreviewManifest struct {
	Dataset     artifact.ID                   `json:"dataset"`
	Split       artifact.ID                   `json:"split"`
	Position    uint64                        `json:"position"`
	Seed        uint64                        `json:"seed"`
	Shuffle     bool                          `json:"shuffle"`
	MemoryBytes uint64                        `json:"memory_bytes"`
	Policy      dataset.AudioInspectionPolicy `json:"policy"`
}

// previewTrainingData executes the same source, processor and sampler used by
// the trainer, before a model or optimizer is needed. Output contains shapes,
// rates and target text, not PCM arrays. Signal admission evidence is persisted.
func previewTrainingData(ctx context.Context, repository artifact.Repository, path string, output io.Writer) error {
	data, err := artifact.ReadContentFile(path)
	if err != nil {
		return err
	}
	var request trainingDataPreviewManifest
	if err := strictjson.DecodeBytes(data, &request); err != nil {
		return err
	}
	if err := request.Policy.Validate(); err != nil {
		return err
	}
	if request.MemoryBytes == 0 || request.Policy.MaximumEncodedBytes > request.MemoryBytes || request.Policy.MaximumSamples > request.MemoryBytes/uint64(binary.Size(float32(0))) {
		return errors.New("train: preview sample bounds exceed declared memory")
	}
	source, err := dataset.NewAudioPayloadReader(request.MemoryBytes)
	if err != nil {
		return err
	}
	defer source.Close()
	processor, err := trainingdata.AudioTextProcessor(repository, source, request.Policy)
	if err != nil {
		return err
	}
	profile, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/audio-text-processor/v1"), request.Policy)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, repository, artifact.Batch{Key: "training/data-processor/" + profile.Descriptor.ID.String(), Contents: []artifact.Content{profile}}); err != nil {
		return err
	}
	authority := trainingdata.Authority{Dataset: request.Dataset, Split: request.Split, Seed: request.Seed, Shuffle: request.Shuffle, Processors: []artifact.ID{profile.Descriptor.ID}, Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityAudio}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}}}
	materialized, err := trainingdata.Materialize(ctx, repository, authority, []trainingdata.ProcessorBinding{{Artifact: profile.Descriptor.ID, Modalities: []recipecontract.Modality{recipecontract.ModalityAudio, recipecontract.ModalityText}, Process: processor}})
	if err != nil {
		return err
	}
	defer materialized.Close()
	preview, err := trainingdata.Preview(ctx, materialized, request.Position, 1, request.MemoryBytes)
	if err != nil {
		return err
	}
	return json.NewEncoder(output).Encode(struct {
		Preview           trainingdata.PreviewResult `json:"preview"`
		ModelLoaded       bool                       `json:"model_loaded"`
		ParametersUpdated bool                       `json:"parameters_updated"`
	}{Preview: preview})
}
