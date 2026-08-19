package trainingworkflow

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"

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

type rolloutDocument struct {
	ID, Group, Prompt, Completion, Evaluator string
	Reward                                   float64
}

type encodedRollout struct {
	ID     string  `json:"id"`
	Tokens []int   `json:"tokens"`
	Reward float64 `json:"reward"`
}

type encodedRolloutGroup struct {
	ID        string           `json:"id"`
	Evaluator artifact.ID      `json:"evaluator"`
	Prefix    int              `json:"prefix"`
	Rollouts  []encodedRollout `json:"rollouts"`
}

func groupedRolloutBatchesResume(ctx context.Context, raw []byte, steps int, encode func(string) ([]int, error), resume *trainingdata.StreamState) ([]trainingdata.RolloutGroup, trainingdata.StreamState, batchAuthority, []artifact.ID, error) {
	groups, evaluators, err := parseRollouts(raw, encode)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
	}
	authority, processor, err := datasetAuthority(raw, "grouped-rollout")
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
	}
	records := make([]trainingdata.RawRecord, len(groups))
	for index, group := range groups {
		data, err := json.Marshal(group)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
		}
		records[index] = trainingdata.RawRecord{ID: group.ID, Group: group.ID, Data: data}
	}
	materialized, err := trainingdata.MaterializeRecords(trainingdata.Authority{
		Dataset: authority.Dataset, Split: authority.Split, Processors: []artifact.ID{processor}, Signature: textSignature,
	}, processor, records, trainingdata.ProcessorBinding{
		Artifact: processor, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "grouped-rollout-json"),
	})
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, resume)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
	}
	batcher, err := trainingdata.NewBatcher(stream, unitBatchPolicy)
	if err != nil {
		return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
	}
	result := make([]trainingdata.RolloutGroup, steps)
	for step := range result {
		batch, err := batcher.Next(ctx)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
		}
		var encoded encodedRolloutGroup
		if err := json.Unmarshal(batch.Examples[0].Values[0].Data, &encoded); err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
		}
		rollouts := make([]trainingdata.Rollout, len(encoded.Rollouts))
		for index, rollout := range encoded.Rollouts {
			completion := make([]bool, len(rollout.Tokens))
			for token := encoded.Prefix; token < len(completion); token++ {
				completion[token] = true
			}
			rollouts[index] = trainingdata.Rollout{ID: rollout.ID, Reward: rollout.Reward,
				Sequence: trainingdata.PreferenceSequence{Tokens: rollout.Tokens, Completion: completion}}
		}
		result[step], err = trainingdata.NewRolloutGroup(encoded.ID, encoded.Evaluator, batch.State, rollouts)
		if err != nil {
			return nil, trainingdata.StreamState{}, batchAuthority{}, nil, err
		}
	}
	return result, stream.Snapshot(), authority, evaluators, nil
}

func parseRollouts(raw []byte, encode func(string) ([]int, error)) ([]encodedRolloutGroup, []artifact.ID, error) {
	if len(raw) == 0 || encode == nil {
		return nil, nil, errors.New("training workflow: invalid rollout dataset")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var order []string
	groups := make(map[string]*encodedRolloutGroup)
	for {
		var document rolloutDocument
		if err := decoder.Decode(&document); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, nil, fmt.Errorf("training workflow: decode rollout: %w", err)
		}
		evaluator, err := artifact.ParseID(document.Evaluator)
		if document.ID == "" || document.Group == "" || document.Prompt == "" || document.Completion == "" ||
			err != nil || evaluator.Kind() != artifact.KindEvidence || math.IsNaN(document.Reward) || math.IsInf(document.Reward, 0) {
			return nil, nil, errors.New("training workflow: rollout facts are incomplete")
		}
		tokens, err := encode(document.Prompt + document.Completion)
		if err != nil {
			return nil, nil, err
		}
		group := groups[document.Group]
		if group == nil {
			group = &encodedRolloutGroup{ID: document.Group, Evaluator: evaluator}
			groups[document.Group], order = group, append(order, document.Group)
		}
		if group.Evaluator != evaluator {
			return nil, nil, errors.New("training workflow: rollout group evaluator differs")
		}
		group.Rollouts = append(group.Rollouts, encodedRollout{ID: document.ID, Tokens: tokens, Reward: document.Reward})
	}
	result := make([]encodedRolloutGroup, len(order))
	evaluatorSet := make(map[artifact.ID]struct{})
	for index, id := range order {
		group := groups[id]
		if len(group.Rollouts) < 2 {
			return nil, nil, errors.New("training workflow: rollout group needs multiple candidates")
		}
		prefix := len(group.Rollouts[0].Tokens)
		for _, rollout := range group.Rollouts[1:] {
			prefix = min(prefix, trainingdata.CommonTokenPrefix(group.Rollouts[0].Tokens, rollout.Tokens))
		}
		for _, rollout := range group.Rollouts {
			if prefix == 0 || prefix >= len(rollout.Tokens) {
				return nil, nil, fmt.Errorf("training workflow: rollout group lacks exact prompt prefix (%d of %d)", prefix, len(rollout.Tokens))
			}
		}
		group.Prefix = prefix
		result[index] = *group
		evaluatorSet[group.Evaluator] = struct{}{}
	}
	evaluators := make([]artifact.ID, 0, len(evaluatorSet))
	for evaluator := range evaluatorSet {
		evaluators = append(evaluators, evaluator)
	}
	slices.SortFunc(evaluators, func(left, right artifact.ID) int { return cmp.Compare(left.String(), right.String()) })
	return result, evaluators, nil
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
