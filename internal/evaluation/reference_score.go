package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	referenceScoresMediaType   = "application/vnd.overgo.evaluation-references+json"
	referenceScoresSchema      = "overgo/evaluation-references/v1"
	referenceScoresAliasPrefix = "evaluation/references/"
)

// ReferenceScore is one published result for a model on a suite: the
// value a model card or paper reports, the protocol it was measured
// under (shots, scoring), and the source it came from. A reference is a
// declared fact beside the measured value, never a target the measured
// value is adjusted toward; the report prints both so a reader judges
// the gap with the protocol difference in view.
type ReferenceScore struct {
	Suite    string  `json:"suite"`
	Metric   string  `json:"metric"`
	Value    float64 `json:"value"`
	Protocol string  `json:"protocol"`
	Source   string  `json:"source"`
}

// ReferenceScoreDeclaration binds one model to its published results.
type ReferenceScoreDeclaration struct {
	ID         artifact.ID      `json:"-"`
	Version    uint16           `json:"version"`
	Model      artifact.ID      `json:"model"`
	References []ReferenceScore `json:"references"`
}

var referenceScoreCodec = artifact.JSONDocumentCodec(
	"evaluation references", artifact.KindEvidence, referenceScoresMediaType, referenceScoresSchema,
	canonicalizeReferenceScores, func(value ReferenceScoreDeclaration) artifact.ID { return value.ID },
	func(value *ReferenceScoreDeclaration, id artifact.ID) { value.ID = id },
	func(value ReferenceScoreDeclaration) ReferenceScoreDeclaration {
		value.References = slices.Clone(value.References)
		return value
	},
)

func compareReferenceScores(a, b ReferenceScore) int {
	if order := strings.Compare(a.Suite, b.Suite); order != 0 {
		return order
	}
	if order := strings.Compare(a.Metric, b.Metric); order != 0 {
		return order
	}
	return strings.Compare(a.Source, b.Source)
}

func canonicalizeReferenceScores(value *ReferenceScoreDeclaration) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Model.Kind() != artifact.KindModel || len(value.References) == 0 {
		return errors.New("evaluation: invalid reference declaration")
	}
	value.References = slices.Clone(value.References)
	for _, reference := range value.References {
		for _, field := range []string{reference.Suite, reference.Metric, reference.Protocol, reference.Source} {
			if field == "" || strings.TrimSpace(field) != field {
				return errors.New("evaluation: invalid reference field")
			}
		}
		if math.IsNaN(reference.Value) || math.IsInf(reference.Value, 0) {
			return errors.New("evaluation: reference value is not finite")
		}
	}
	slices.SortFunc(value.References, compareReferenceScores)
	for index := 1; index < len(value.References); index++ {
		if compareReferenceScores(value.References[index-1], value.References[index]) == 0 {
			return errors.New("evaluation: duplicate reference")
		}
	}
	return nil
}

// DeclareReferenceScores merges the given references into the model's
// declaration -- a reference with the same suite, metric, and source
// replaces its predecessor -- and rebinds the alias.
func DeclareReferenceScores(
	ctx context.Context, repository artifact.Repository, model artifact.ID, references []ReferenceScore,
) (artifact.ID, error) {
	current, err := LoadReferenceScores(ctx, repository, model)
	if err != nil {
		return artifact.ID{}, err
	}
	merged := slices.Clone(current)
	for _, reference := range references {
		replaced := false
		for index := range merged {
			if compareReferenceScores(merged[index], reference) == 0 {
				merged[index], replaced = reference, true
			}
		}
		if !replaced {
			merged = append(merged, reference)
		}
	}
	declaration, err := referenceScoreCodec.New(ReferenceScoreDeclaration{
		Version: artifact.InitialDocumentVersion, Model: model, References: merged,
	})
	if err != nil {
		return artifact.ID{}, err
	}
	alias := artifact.AliasBinding{Name: referenceScoresAliasPrefix + model.String(), Target: declaration.ID}
	previous, bound, err := artifact.ResolveAlias(ctx, repository, alias.Name)
	if err != nil {
		return artifact.ID{}, err
	}
	if bound {
		if previous == declaration.ID {
			return declaration.ID, nil
		}
		alias.Previous = &previous
	}
	batch, err := referenceScoreCodec.Batch(
		referenceScoresAliasPrefix+declaration.ID.String(), declaration, nil, []artifact.AliasBinding{alias},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return artifact.ID{}, err
	}
	return declaration.ID, nil
}

// LoadReferenceScores reads one model's declared published results;
// an undeclared model has none.
func LoadReferenceScores(ctx context.Context, reader artifact.Reader, model artifact.ID) ([]ReferenceScore, error) {
	id, bound, err := artifact.ResolveAlias(ctx, reader, referenceScoresAliasPrefix+model.String())
	if err != nil || !bound {
		return nil, err
	}
	declaration, found, err := referenceScoreCodec.Read(ctx, reader, id)
	if err != nil || !found {
		return nil, err
	}
	return declaration.References, nil
}
