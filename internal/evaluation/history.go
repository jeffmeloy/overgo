package evaluation

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

type HistoryEntry struct {
	Evaluation artifact.ID        `json:"evaluation,omitempty"`
	Run        artifact.ID        `json:"run"`
	Report     artifact.ID        `json:"report,omitempty"`
	Dataset    artifact.ID        `json:"dataset"`
	Recipe     artifact.ID        `json:"recipe"`
	Outcome    runrecord.Outcome  `json:"outcome"`
	Failure    string             `json:"failure,omitempty"`
	CodeCommit string             `json:"code_commit,omitempty"`
	MeasuredNS uint64             `json:"measured_ns,omitempty"`
	Metrics    []runrecord.Metric `json:"metrics"`
	Inputs     []artifact.ID      `json:"inputs,omitempty"`
	Outputs    []artifact.ID      `json:"outputs,omitempty"`
}

// History follows this campaign's model index.
func (campaign *Campaign) History(ctx context.Context, suites []CompiledSuite) ([]HistoryEntry, error) {
	if campaign == nil || ctx == nil || campaign.repository == nil || len(suites) == 0 {
		return nil, errors.New("evaluation: history authorities are incomplete")
	}
	datasetByPlan := make(map[artifact.ID]artifact.ID, len(suites))
	for _, suite := range suites {
		descriptor := suite.Descriptor()
		if descriptor.Plan.Kind() != artifact.KindProfile || descriptor.Dataset.Kind() != artifact.KindDataset {
			return nil, errors.New("evaluation: history suite is invalid")
		}
		if _, duplicate := datasetByPlan[descriptor.Plan]; duplicate {
			return nil, errors.New("evaluation: duplicate history suite")
		}
		datasetByPlan[descriptor.Plan] = descriptor.Dataset
	}
	repository := campaign.repository
	model, recipe := campaign.identity.Model, campaign.identity.Recipe
	edges, err := repository.Children(ctx, model)
	if err != nil {
		return nil, err
	}
	runs := make(map[artifact.ID]runrecord.Run)
	evaluations := make(map[artifact.ID]runrecord.Evaluation)
	for _, edge := range edges {
		if edge.Parent != model || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		descriptor, found, err := repository.Artifact(ctx, edge.Child)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, errors.New("evaluation: indexed history artifact is absent")
		}
		if descriptor.MediaType != runrecord.RunMediaType && descriptor.MediaType != runrecord.EvaluationMediaType {
			continue
		}
		switch {
		case descriptor.MediaType == runrecord.RunMediaType:
			run, err := runrecord.RequireRun(ctx, repository, edge.Child)
			if err != nil {
				return nil, errors.Join(err, errors.New("evaluation: indexed run is invalid"))
			}
			if run.Recipe == recipe && len(run.Inputs) != 0 {
				if _, admitted := datasetByPlan[run.Inputs[0]]; admitted {
					runs[run.ID] = run
				}
			}
		case descriptor.MediaType == runrecord.EvaluationMediaType && descriptor.Schema == runrecord.EvaluationSchema:
			record, err := runrecord.RequireEvaluation(ctx, repository, edge.Child)
			if err != nil {
				return nil, errors.Join(err, errors.New("evaluation: indexed metric record is invalid"))
			}
			if record.Recipe == recipe {
				if previous, duplicate := evaluations[record.Run]; duplicate && previous.ID != record.ID {
					return nil, errors.New("evaluation: indexed run has multiple metric records")
				}
				evaluations[record.Run] = record
			}
		}
	}
	entries := make([]HistoryEntry, 0, len(runs))
	for _, run := range runs {
		dataset := datasetByPlan[run.Inputs[0]]
		record, evaluated := evaluations[run.ID]
		if run.Outcome == runrecord.OutcomeSucceeded && !evaluated {
			return nil, errors.New("evaluation: successful indexed run has no metric record")
		}
		if evaluated && record.Dataset != dataset {
			return nil, errors.New("evaluation: indexed run dataset differs from plan")
		}
		entry := HistoryEntry{
			Run: run.ID, Dataset: dataset, Recipe: run.Recipe, Outcome: run.Outcome,
			Failure: run.Failure, CodeCommit: run.CodeCommit, MeasuredNS: run.MeasuredNS,
			Inputs: slices.Clone(run.Inputs), Outputs: slices.Clone(run.Outputs),
		}
		if evaluated {
			entry.Evaluation, entry.Metrics = record.ID, slices.Clone(record.Metrics)
		}
		if len(run.Outputs) != 0 {
			entry.Report = run.Outputs[0]
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Run.String() < entries[j].Run.String() })
	return entries, nil
}
