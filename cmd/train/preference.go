package main

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

type preferenceDocument struct {
	ID       string `json:"id"`
	Group    string `json:"group,omitempty"`
	Prompt   string `json:"prompt"`
	Chosen   string `json:"chosen"`
	Rejected string `json:"rejected"`
}

type encodedPreference struct {
	Chosen   []int `json:"chosen"`
	Rejected []int `json:"rejected"`
}

func preferenceBatchesResume(
	ctx context.Context,
	raw []byte,
	steps int,
	encode func(string) ([]int, error),
	resume *trainingdata.StreamState,
) ([]trainingdata.PreferenceBatch, trainingdata.StreamState, batchAuthority, error) {
	if len(raw) == 0 || steps <= 0 || encode == nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, errors.New("train: invalid preference stream input")
	}
	documents, err := parsePreferenceDocuments(raw, encode)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	datasetID, err := artifact.IdentifyBytes(artifact.KindDataset, raw)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/preference-token-pair/v1"))
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
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
	}
	materialized, err := trainingdata.MaterializeRecords(authority, processorID, records, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: func(_ context.Context, record trainingdata.RawRecord) (trainingdata.Example, error) {
			var encoded encodedPreference
			if err := json.Unmarshal(record.Data, &encoded); err != nil {
				return trainingdata.Example{}, err
			}
			chosen, rejected, err := trainingdata.PreferenceTokenValues(encoded.Chosen, encoded.Rejected)
			if err != nil {
				return trainingdata.Example{}, err
			}
			return trainingdata.Example{ID: record.ID, Group: record.Group, Values: []trainingdata.Value{chosen, rejected}}, nil
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
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
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
	return batches, stream.Snapshot(), batchAuthority{Dataset: datasetID, Split: splitID, Processor: processorID}, nil
}

type preparedPreference struct {
	id      string
	group   string
	encoded encodedPreference
}

func parsePreferenceDocuments(raw []byte, encode func(string) ([]int, error)) ([]preparedPreference, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var result []preparedPreference
	for {
		var document preferenceDocument
		if err := decoder.Decode(&document); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("train: decode preference record: %w", err)
		}
		if document.ID == "" || document.Prompt == "" || document.Chosen == "" || document.Rejected == "" {
			return nil, errors.New("train: preference record requires id, prompt, chosen, and rejected")
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
			return nil, errors.New("train: preference responses require an exact shared prefix and distinct suffixes")
		}
		result = append(result, preparedPreference{
			id: document.ID, group: document.Group,
			encoded: encodedPreference{Chosen: chosen, Rejected: rejected},
		})
	}
	if len(result) == 0 {
		return nil, errors.New("train: preference dataset is empty")
	}
	return result, nil
}
