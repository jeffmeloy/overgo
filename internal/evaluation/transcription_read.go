package evaluation

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

// RequireTranscriptionReport verifies stored quality by re-scoring the exact
// persisted outputs under the original plan. It performs no inference or writes.
func RequireTranscriptionReport(ctx context.Context, reader artifact.Reader, id artifact.ID) (TranscriptionReport, error) {
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return TranscriptionReport{}, err
	}
	if !found {
		return TranscriptionReport{}, errors.New("evaluation: transcription report is absent")
	}
	if err := transcriptionReportContract.ValidateContent(content, id); err != nil {
		return TranscriptionReport{}, err
	}
	var report TranscriptionReport
	if err := strictjson.DecodeBytes(content.Data, &report); err != nil {
		return TranscriptionReport{}, err
	}
	report.ID = id
	plan, err := loadEvidencePlan(ctx, reader, report.Plan)
	if err != nil {
		return TranscriptionReport{}, err
	}
	profile, found, err := artifact.ReadContent(ctx, reader, plan.body.CaseProfile)
	if err != nil {
		return TranscriptionReport{}, err
	}
	if !found {
		return TranscriptionReport{}, errors.New("evaluation: transcription case profile is absent")
	}
	var suite TranscriptionSuite
	if err := strictjson.DecodeBytes(profile.Data, &suite); err != nil {
		return TranscriptionReport{}, err
	}
	compiled, err := CompileTranscription(suite)
	if err != nil {
		return TranscriptionReport{}, err
	}
	predictions := make([]TranscriptionPrediction, len(report.Observations))
	for i, observation := range report.Observations {
		predictions[i] = TranscriptionPrediction{Name: observation.Name, Run: observation.Run}
	}
	expected, err := scoreTranscriptionReport(ctx, reader, compiled, plan, predictions)
	if err != nil {
		return TranscriptionReport{}, err
	}
	if !reflect.DeepEqual(expected, report) {
		return TranscriptionReport{}, errors.New("evaluation: transcription report differs from persisted outputs or scoring protocol")
	}
	return report, nil
}

// pooledRateError is the standard error of a group's word error rate. The
// rate is one sum over the group's utterances divided by another, word edits
// over reference words, and what was sampled is utterances, not words: a
// speaker's errors come together, so the words of one utterance are not so
// many independent trials. The error is that of a ratio of sums over the
// utterances, each utterance's edits taken against what the rate expects of
// its words. Fewer than two utterances, or no words, have none.
func pooledRateError(observations []transcriptionObservation, group string) float64 {
	var edits, words, count float64
	for _, observation := range observations {
		if observation.Group == group && !observation.SilentControl {
			edits, words, count = edits+float64(observation.WordEdits), words+float64(observation.ReferenceWords), count+1
		}
	}
	if count <= 1 || words == 0 {
		return 0
	}
	rate := edits / words
	var squares float64
	for _, observation := range observations {
		if observation.Group == group && !observation.SilentControl {
			residual := float64(observation.WordEdits) - rate*float64(observation.ReferenceWords)
			squares += residual * residual
		}
	}
	return math.Sqrt(squares*count/(count-1)) / words
}

// TranscriptionQualities reads the newest transcription report of each
// runtime recipe, re-scored from its persisted outputs, as the recipe's word
// error rate on the report's largest group. The summary's suite is the
// report's dataset and that group, and its protocol the prompting the plan
// executed under followed by the normalization the report scored with. A
// report whose groups hold no utterance measures nothing and is passed over.
// A report that cannot be reconstructed stops the read with its error, or, when
// unreadable is given, is reported to it with the recipe it judges and the read
// goes on: it claims its recipe like a readable one, newest first, so an older
// report of that recipe does not stand in for it, and costs the recipe its quality
// and no other.
func TranscriptionQualities(ctx context.Context, store *overgodb.Store, unreadable func(recipe artifact.ID, reason error)) (map[artifact.ID]EvalSummary, error) {
	qualities := map[artifact.ID]EvalSummary{}
	claimed := map[artifact.ID]bool{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{transcriptionReportContract}, Order: overgodb.DocumentNewestFirst,
	}, func(data []byte) (TranscriptionReport, error) {
		var report TranscriptionReport
		err := strictjson.DecodeBytes(data, &report)
		return report, err
	}, func(view overgodb.DocumentView, stored TranscriptionReport) error {
		plan, err := loadEvidencePlan(ctx, store, stored.Plan)
		if err != nil {
			return err
		}
		if _, read := qualities[plan.body.RuntimeRecipe]; read || claimed[plan.body.RuntimeRecipe] {
			return nil
		}
		report, err := RequireTranscriptionReport(ctx, store, view.Content.Descriptor.ID)
		if err != nil {
			if unreadable == nil || ctx.Err() != nil {
				return err
			}
			claimed[plan.body.RuntimeRecipe] = true
			unreadable(plan.body.RuntimeRecipe, err)
			return nil
		}
		policy, err := ReadExecutionPolicy(ctx, store, plan.body.Execution)
		if err != nil {
			return err
		}
		// Groups arrive in name order; only a larger group displaces.
		var largest transcriptionSlice
		for _, group := range report.Groups {
			if group.Utterances > largest.Utterances {
				largest = group
			}
		}
		if largest.Utterances == 0 {
			return nil
		}
		protocol := []string{policy.Prompting.Label()}
		for _, operation := range report.Normalization {
			protocol = append(protocol, string(operation))
		}
		summary := EvalSummary{
			Suite: report.Dataset.String() + "/" + largest.Name, Prompting: strings.Join(protocol, ", "), Record: report.ID,
			// The rate pools the group's words over its utterances, which are its
			// cases; its noise follows from how the edits fall among them.
			Cases: largest.Utterances, Noise: pooledRateError(report.Observations, largest.Name),
			Metrics:    map[string]float64{wordErrorRateMetric: largest.WordErrorRate},
			Directions: map[string]runrecord.Direction{wordErrorRateMetric: runrecord.DirectionMinimize},
		}
		// Re-scoring already checked every prediction against the plan. Missing
		// environment content retains quality but cannot earn comparison credit.
		if environment, err := runrecord.RequireEnvironment(ctx, store, plan.body.Environment); err == nil {
			comparison, err := plan.comparisonIdentity()
			if err != nil {
				return err
			}
			summary.Comparison, summary.Environment, summary.CodeCommit = comparison, environment.ID, plan.body.CodeCommit
			summary.ExecutionKnown, summary.Reproducible = true, environment.Reproducible()
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		qualities[plan.body.RuntimeRecipe] = summary
		return nil
	})
	return qualities, err
}
