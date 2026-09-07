package evaluation

import (
	"context"
	"errors"
	"reflect"

	"overgo/internal/artifact"
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
