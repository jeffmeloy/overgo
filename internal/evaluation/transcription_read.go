package evaluation

import (
	"context"
	"errors"
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

// TranscriptionQualities reads the newest transcription report of each
// runtime recipe, re-scored from its persisted outputs, as the recipe's word
// error rate on the report's largest group. The summary's suite is the
// report's dataset and that group, and its protocol the prompting the plan
// executed under followed by the normalization the report scored with. A
// report whose groups hold no utterance measures nothing and is passed over.
func TranscriptionQualities(ctx context.Context, store *overgodb.Store) (map[artifact.ID]EvalSummary, error) {
	qualities := map[artifact.ID]EvalSummary{}
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
		if _, read := qualities[plan.body.RuntimeRecipe]; read {
			return nil
		}
		report, err := RequireTranscriptionReport(ctx, store, view.Content.Descriptor.ID)
		if err != nil {
			return err
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
		qualities[plan.body.RuntimeRecipe] = EvalSummary{
			Suite: report.Dataset.String() + "/" + largest.Name, Prompting: strings.Join(protocol, ", "), Record: report.ID,
			Metrics:    map[string]float64{wordErrorRateMetric: largest.WordErrorRate},
			Directions: map[string]runrecord.Direction{wordErrorRateMetric: runrecord.DirectionMinimize},
		}
		return nil
	})
	return qualities, err
}
