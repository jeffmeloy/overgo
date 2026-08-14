package scratchmodel

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

func (c Construction) documentBatcher(documents []string) (*trainingdata.Dataset, *trainingdata.Batcher, error) {
	if len(documents) == 0 || c.processor.Kind() != artifact.KindProfile {
		return nil, nil, errors.New("scratch model: training documents absent")
	}
	authority := trainingdata.Authority{
		Dataset: c.dataset, Split: c.splitID, Processors: []artifact.ID{c.processor}, Seed: uint64(c.seed),
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
	}
	materialized, err := trainingdata.MaterializeDocuments(authority, c.processor, documents, trainingdata.ProcessorBinding{
		Artifact:   c.processor,
		Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process:    trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
	})
	if err != nil {
		return nil, nil, err
	}
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		materialized.Close()
		return nil, nil, err
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{
		Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1,
	})
	if err != nil {
		materialized.Close()
		return nil, nil, err
	}
	return materialized, batcher, nil
}

func nextDocument(ctx context.Context, batcher *trainingdata.Batcher) (string, error) {
	batch, err := batcher.Next(ctx)
	if err != nil {
		return "", err
	}
	if len(batch.Examples) != 1 || len(batch.Examples[0].Values) != 1 {
		return "", errors.New("scratch model: text processor emitted invalid batch")
	}
	return string(batch.Examples[0].Values[0].Data), nil
}
