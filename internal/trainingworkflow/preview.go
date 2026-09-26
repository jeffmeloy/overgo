package trainingworkflow

import (
	"errors"
	"fmt"
	"os"

	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// PreviewToken is one token of a record as training encodes it; a token the
// objective does not train is the shared prompt prefix it masks.
type PreviewToken struct {
	ID      int    `json:"id"`
	Piece   string `json:"piece"`
	Trained bool   `json:"trained"`
}

// PreviewSequence is one encoded sequence of a record: a preference pair's
// chosen or rejected side, or one rollout of a group.
type PreviewSequence struct {
	Label  string         `json:"label"`
	Tokens []PreviewToken `json:"tokens"`
}

// PreviewRecord is one training record: a preference pair or a rollout group.
type PreviewRecord struct {
	ID        string            `json:"id"`
	Sequences []PreviewSequence `json:"sequences"`
}

// Preview is one page of a dataset's records as the objective encodes them,
// with every sequence's token length across the dataset.
type Preview struct {
	Objective trainingprogram.ObjectiveKind `json:"objective"`
	Records   int                           `json:"records"`
	Position  int                           `json:"position"`
	Page      []PreviewRecord               `json:"page"`
	Lengths   []int                         `json:"lengths"`
}

// PreviewDataset reads the dataset at datasetPath through the objective's own
// parse with the tokenizer beside modelInput -- prompt and response
// concatenated as training reads them -- and pages records [position,
// position+limit), marking the tokens past each record's shared prompt prefix
// as trained.
func PreviewDataset(objective trainingprogram.ObjectiveKind, modelInput, datasetPath string, position, limit int) (Preview, error) {
	if position < 0 || limit <= 0 {
		return Preview{}, errors.New("training workflow: preview page is invalid")
	}
	tokens, err := loadTrainableTokenizer(modelInput)
	if err != nil {
		return Preview{}, err
	}
	raw, err := os.ReadFile(datasetPath)
	if err != nil {
		return Preview{}, err
	}
	preview := Preview{Objective: objective, Position: position}
	paged := func(index int) bool { return index >= position && index < position+limit }
	sequence := func(label string, ids []int, prefix int) PreviewSequence {
		preview.Lengths = append(preview.Lengths, len(ids))
		result := PreviewSequence{Label: label, Tokens: make([]PreviewToken, len(ids))}
		for index, id := range ids {
			result.Tokens[index] = PreviewToken{ID: id, Piece: tokens.piece(id), Trained: index >= prefix}
		}
		return result
	}
	switch objective {
	case trainingprogram.ObjectiveDPO:
		documents, err := parsePreferences(raw, tokens.encode)
		if err != nil {
			return Preview{}, err
		}
		preview.Records = len(documents)
		for index, document := range documents {
			prefix := trainingdata.CommonTokenPrefix(document.encoded.Chosen, document.encoded.Rejected)
			record := PreviewRecord{ID: document.id, Sequences: []PreviewSequence{
				sequence("chosen", document.encoded.Chosen, prefix), sequence("rejected", document.encoded.Rejected, prefix),
			}}
			if paged(index) {
				preview.Page = append(preview.Page, record)
			}
		}
	case trainingprogram.ObjectiveGRPO:
		groups, _, err := parseRollouts(raw, tokens.encode)
		if err != nil {
			return Preview{}, err
		}
		preview.Records = len(groups)
		for index, group := range groups {
			record := PreviewRecord{ID: group.ID}
			for _, rollout := range group.Rollouts {
				label := rollout.ID + " · reward " + fmt.Sprint(rollout.Reward)
				record.Sequences = append(record.Sequences, sequence(label, rollout.Tokens, group.Prefix))
			}
			if paged(index) {
				preview.Page = append(preview.Page, record)
			}
		}
	default:
		return Preview{}, fmt.Errorf("training workflow: %s datasets have no preview", objective)
	}
	return preview, nil
}
