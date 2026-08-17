package trainingworkflow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

type batchAuthority struct {
	Dataset, Split, Processor artifact.ID
}

func tokenBatchesResume(ctx context.Context, raw []byte, steps, maximumSequence int, encode func(string) ([]int, error), resume *trainingdata.StreamState) ([][]int, trainingdata.StreamState, batchAuthority, error) {
	authority, materialized, err := materializeText(raw)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, resume)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	batcher, err := trainingdata.NewBatcher(stream, unitBatchPolicy)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	batches := make([][]int, steps)
	for step := range batches {
		batch, err := batcher.Next(ctx)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, err
		}
		tokens, err := encode(string(batch.Examples[0].Values[0].Data))
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, fmt.Errorf("training workflow: encode dataset: %w", err)
		}
		if maximumSequence > 0 && len(tokens) > maximumSequence {
			tokens = tokens[:maximumSequence]
		}
		if len(tokens) < 2 {
			return nil, trainingdata.StreamState{}, batchAuthority{}, fmt.Errorf("training workflow: record has %d tokens", len(tokens))
		}
		batches[step] = tokens
	}
	return batches, stream.Snapshot(), authority, nil
}

func materializeText(raw []byte) (batchAuthority, *trainingdata.Dataset, error) {
	if len(raw) == 0 {
		return batchAuthority{}, nil, errors.New("training workflow: empty text dataset")
	}
	authority, processor, err := datasetAuthority(raw, "text-utf8")
	if err != nil {
		return batchAuthority{}, nil, err
	}
	materialized, err := trainingdata.MaterializeDocuments(trainingdata.Authority{
		Dataset: authority.Dataset, Split: authority.Split, Processors: []artifact.ID{processor}, Signature: textSignature,
	}, processor, []string{string(raw)}, trainingdata.ProcessorBinding{
		Artifact: processor, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
	})
	return authority, materialized, err
}

type preferenceDocument struct {
	ID, Group, Prompt, Chosen, Rejected string
}

type encodedPreference struct {
	Chosen, Rejected []int
}

type preparedPreference struct {
	id, group string
	encoded   encodedPreference
}

func preferenceBatchesResume(ctx context.Context, raw []byte, steps int, encode func(string) ([]int, error), resume *trainingdata.StreamState) ([]trainingdata.PreferenceBatch, trainingdata.StreamState, batchAuthority, error) {
	documents, err := parsePreferences(raw, encode)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	authority, processor, err := datasetAuthority(raw, "preference-token-pair")
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	records := make([]trainingdata.RawRecord, len(documents))
	for index, document := range documents {
		data, err := json.Marshal(document.encoded)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, err
		}
		records[index] = trainingdata.RawRecord{ID: document.id, Group: document.group, Data: data}
	}
	materialized, err := trainingdata.MaterializeRecords(trainingdata.Authority{
		Dataset: authority.Dataset, Split: authority.Split, Processors: []artifact.ID{processor}, Signature: textSignature,
	}, processor, records, trainingdata.ProcessorBinding{
		Artifact: processor, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: func(_ context.Context, record trainingdata.RawRecord) (trainingdata.Example, error) {
			var encoded encodedPreference
			if err := json.Unmarshal(record.Data, &encoded); err != nil {
				return trainingdata.Example{}, err
			}
			chosen, rejected, err := trainingdata.PreferenceTokenValues(encoded.Chosen, encoded.Rejected)
			return trainingdata.Example{ID: record.ID, Group: record.Group, Values: []trainingdata.Value{chosen, rejected}}, err
		},
	})
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, resume)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	batcher, err := trainingdata.NewBatcher(stream, unitBatchPolicy)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	batches := make([]trainingdata.PreferenceBatch, steps)
	for step := range batches {
		batch, err := batcher.Next(ctx)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, err
		}
		batches[step], err = trainingdata.CompilePreferenceBatch(batch, trainingdata.TokenIDs)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, err
		}
	}
	return batches, stream.Snapshot(), authority, nil
}

func parsePreferences(raw []byte, encode func(string) ([]int, error)) ([]preparedPreference, error) {
	if len(raw) == 0 || encode == nil {
		return nil, errors.New("training workflow: invalid preference dataset")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var result []preparedPreference
	for {
		var document preferenceDocument
		if err := decoder.Decode(&document); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("training workflow: decode preference: %w", err)
		}
		if document.ID == "" || document.Prompt == "" || document.Chosen == "" || document.Rejected == "" {
			return nil, errors.New("training workflow: preference requires id, prompt, chosen, and rejected")
		}
		chosen, err := encode(document.Prompt + document.Chosen)
		if err != nil {
			return nil, err
		}
		rejected, err := encode(document.Prompt + document.Rejected)
		if err != nil {
			return nil, err
		}
		prefix := trainingdata.CommonTokenPrefix(chosen, rejected)
		if prefix == 0 || prefix >= len(chosen) || prefix >= len(rejected) {
			return nil, errors.New("training workflow: preference responses need exact shared prefix and distinct suffixes")
		}
		result = append(result, preparedPreference{id: document.ID, group: document.Group, encoded: encodedPreference{Chosen: chosen, Rejected: rejected}})
	}
	if len(result) == 0 {
		return nil, errors.New("training workflow: preference dataset empty")
	}
	return result, nil
}

func datasetAuthority(raw []byte, processorName string) (batchAuthority, artifact.ID, error) {
	dataset, err := artifact.IdentifyBytes(artifact.KindDataset, raw)
	if err != nil {
		return batchAuthority{}, artifact.ID{}, err
	}
	split, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	if err != nil {
		return batchAuthority{}, artifact.ID{}, err
	}
	processor, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/"+processorName+"/v1"))
	return batchAuthority{Dataset: dataset, Split: split, Processor: processor}, processor, err
}

var (
	unitBatchPolicy = trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1}
	textSignature   = recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}}
)
