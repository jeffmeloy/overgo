package main

import (
	"context"
	"crypto/sha1"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func checkModelValidation(ctx context.Context, store artifact.Reader, spec modelValidationSpecification, required []string) error {
	if ctx == nil || store == nil || spec.Model.Kind() != artifact.KindModel || spec.Projector.Kind() != artifact.KindProjector ||
		len(spec.CodeCommit) != sha1.Size*2 || strings.Trim(spec.CodeCommit, "0123456789abcdef") != "" || len(required) == 0 || len(spec.Cells) != len(required) {
		return errors.New("model validation: model, source or complete cell denominator is absent")
	}
	wanted := make(map[string]bool, len(required))
	for _, name := range required {
		if name == "" || wanted[name] {
			return errors.New("model validation: required cell is empty or duplicated")
		}
		wanted[name] = true
	}
	seenEvidence, seenPlans := map[artifact.ID]bool{}, map[artifact.ID]bool{}
	for _, cell := range spec.Cells {
		if !wanted[cell.Name] {
			return fmt.Errorf("model validation: duplicate, extra or reused cell %q", cell.Name)
		}
		delete(wanted, cell.Name)
		protocol := strings.HasPrefix(cell.Name, "protocol/")
		resources := strings.HasPrefix(cell.Name, "resources/")
		if !protocol && !resources {
			if seenEvidence[cell.Evidence] || seenPlans[cell.Plan] {
				return fmt.Errorf("model validation: reused non-protocol cell %q", cell.Name)
			}
			seenEvidence[cell.Evidence], seenPlans[cell.Plan] = true, true
			if cell.Run.Valid() || cell.OracleSHA256 != "" {
				return errors.New("model validation: protocol proof cannot replace quality or resource observations")
			}
		}
		if cell.Name == "quality/audio/native" && cell.Evidence.Kind() != artifact.KindEvaluation {
			return errors.New("model validation: audio quality requires a scored transcription report")
		}
		if resources {
			if err := checkResourceValidationCell(ctx, store, spec, cell); err != nil {
				return err
			}
		} else if protocol {
			if err := checkProtocolValidationCell(ctx, store, spec, cell); err != nil {
				return err
			}
		} else if cell.Evidence.Kind() == artifact.KindEvaluation {
			if err := checkTranscriptionValidationCell(ctx, store, spec, cell); err != nil {
				return err
			}
		} else {
			value, err := evaluation.RequireEvaluationEvidence(ctx, store, cell.Evidence)
			if err != nil {
				return fmt.Errorf("model validation %s: %w", cell.Name, err)
			}
			if len(cell.Cases) != 0 {
				return fmt.Errorf("model validation %s: named transcription cases on sharded evidence", cell.Name)
			}
			if err := checkModelValidationCell(spec, cell, value); err != nil {
				return err
			}
		}
		definition, err := recipe.RequireDefinition(ctx, store, cell.Recipe)
		if err != nil {
			return err
		}
		model, found := definition.PrimaryDependency(recipe.DependencyModel)
		if !found || model != spec.Model {
			return fmt.Errorf("model validation %s: recipe model differs", cell.Name)
		}
		if !strings.Contains(cell.Name, "/text/") {
			projector, found := definition.PrimaryDependency(recipe.DependencyProjector)
			if !found || projector != spec.Projector {
				return fmt.Errorf("model validation %s: recipe projector differs", cell.Name)
			}
		}
	}
	return nil
}

func checkModelValidationCell(spec modelValidationSpecification, cell modelValidationCell, value evaluation.EvaluationEvidence) error {
	if strings.HasPrefix(cell.Name, "resources/") {
		return errors.New("model validation: exact-output evidence cannot replace observed media recovery")
	}
	if value.Plan != cell.Plan || value.ModelDefinition != cell.ModelDefinition || value.Recipe != cell.Recipe ||
		value.Dataset != cell.Dataset || value.Split != cell.Split || value.Environment != cell.Environment || value.CodeCommit != spec.CodeCommit ||
		len(cell.Shards) == 0 || len(cell.Bounds) == 0 || len(cell.Shards) != len(value.Shards) {
		return fmt.Errorf("model validation %s: source, authority or case denominator differs", cell.Name)
	}
	expected, actual := slices.Clone(cell.Shards), slices.Clone(value.Shards)
	slices.SortFunc(expected, artifact.CompareID)
	slices.SortFunc(actual, artifact.CompareID)
	if len(slices.Compact(expected)) != len(cell.Shards) || !slices.Equal(expected, actual) {
		return fmt.Errorf("model validation %s: selected cases differ or repeat", cell.Name)
	}
	if err := checkValidationBounds(cell, value.Metrics); err != nil {
		return err
	}

	return nil
}

func checkValidationBounds(cell modelValidationCell, metrics []runrecord.Metric) error {
	if len(cell.Bounds) == 0 {
		return fmt.Errorf("model validation %s: quantitative acceptance is absent", cell.Name)
	}
	seen := map[string]bool{}
	for _, bound := range cell.Bounds {
		if bound.Metric == "" || seen[bound.Metric] || !checked.Finite64(bound.Minimum) || !checked.Finite64(bound.Maximum) || bound.Minimum > bound.Maximum {
			return fmt.Errorf("model validation %s: invalid quantitative bound", cell.Name)
		}
		seen[bound.Metric] = true
		index := slices.IndexFunc(metrics, func(metric runrecord.Metric) bool { return metric.Name == bound.Metric })
		if index < 0 {
			return fmt.Errorf("model validation %s: missing metric %s", cell.Name, bound.Metric)
		}
		metric := metrics[index]
		if metric.Unit != bound.Unit || metric.Direction != bound.Direction || !checked.Finite64(metric.Value) ||
			metric.Value < bound.Minimum || metric.Value > bound.Maximum {
			return fmt.Errorf("model validation %s: %s=%g outside [%g,%g] or protocol differs", cell.Name, bound.Metric, metric.Value, bound.Minimum, bound.Maximum)
		}
	}
	return nil
}

func checkTranscriptionValidationCell(ctx context.Context, store artifact.Reader, spec modelValidationSpecification, cell modelValidationCell) error {
	if cell.Name != "quality/audio/native" || len(cell.Cases) == 0 || len(cell.Shards) != 0 {
		return errors.New("model validation: transcription quality cannot substitute for protocol or resource evidence")
	}
	report, err := evaluation.RequireTranscriptionReport(ctx, store, cell.Evidence)
	if err != nil {
		return err
	}
	if report.Plan != cell.Plan || report.Dataset != cell.Dataset || report.Split != cell.Split ||
		report.Overall.AdmissionFailures != 0 || report.Overall.InferenceFailures != 0 || report.Overall.SilentControlFailures != 0 {
		return errors.New("model validation: transcription authority or completion differs")
	}
	// RequireTranscriptionReport has already validated this plan's typed
	// identity and referenced authorities; select the comparison fields only.
	content, found, err := artifact.ReadContent(ctx, store, report.Plan)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("model validation: transcription plan is absent")
	}
	var authorities struct {
		ModelDefinition artifact.ID `json:"model_definition"`
		Recipe          artifact.ID `json:"runtime_recipe"`
		Environment     artifact.ID `json:"environment"`
		CodeCommit      string      `json:"code_commit"`
	}
	if err := json.Unmarshal(content.Data, &authorities); err != nil {
		return err
	}
	if authorities.ModelDefinition != cell.ModelDefinition || authorities.Recipe != cell.Recipe || authorities.Environment != cell.Environment || authorities.CodeCommit != spec.CodeCommit {
		return errors.New("model validation: transcription execution authority differs")
	}
	expected := slices.Clone(cell.Cases)
	actual := make([]string, len(report.Observations))
	for i, observation := range report.Observations {
		actual[i] = observation.Name
	}
	slices.Sort(expected)
	slices.Sort(actual)
	if len(slices.Compact(expected)) != len(cell.Cases) || !slices.Equal(expected, actual) ||
		report.Overall.Utterances+report.Overall.SilentControls != uint64(len(actual)) {
		return errors.New("model validation: transcription selected-case denominator differs")
	}
	for _, name := range []string{"wer", "cer"} {
		if !slices.ContainsFunc(cell.Bounds, func(bound modelValidationBound) bool {
			return bound.Metric == name && bound.Unit == "ratio" && bound.Direction == runrecord.DirectionMinimize
		}) {
			return fmt.Errorf("model validation: transcription %s bound is absent", name)
		}
	}
	return checkValidationBounds(cell, report.Metrics)
}

func e4bValidationCells() []string {
	var cells []string
	for _, mode := range []string{"text", "image", "multi-image", "audio", "video", "mixed-history"} {
		for _, protocol := range []string{"native", "chat", "responses"} {
			cells = append(cells, "protocol/"+mode+"/"+protocol)
		}
		cells = append(cells, "quality/"+mode+"/native", "resources/"+mode+"/native")
	}
	return cells
}
