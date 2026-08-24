package evaluation

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
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
func (campaign *Campaign) History(ctx context.Context, suites []CompiledSuite, maxResults int) ([]HistoryEntry, error) {
	if campaign == nil || ctx == nil || campaign.repository == nil || campaign.documents == nil ||
		len(suites) == 0 || maxResults <= 0 {
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
	query := overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{
			{Kind: artifact.KindRun, MediaType: runrecord.RunMediaType, Schema: runrecord.LegacyRunSchema},
			{Kind: artifact.KindRun, MediaType: runrecord.RunMediaType, Schema: runrecord.RunSchema},
		},
		Order: overgodb.DocumentNewestFirst, MaxResults: maxResults,
	}
	entries := make([]HistoryEntry, 0, maxResults)
	for len(entries) < maxResults {
		page, err := overgodb.VisitDecodedDocuments(ctx, campaign.documents, query, runrecord.ParseRun,
			func(_ overgodb.DocumentView, run runrecord.Run) error {
				if len(entries) == maxResults || run.Recipe != campaign.identity.Recipe || len(run.Inputs) == 0 {
					return nil
				}
				dataset, admitted := datasetByPlan[run.Inputs[0]]
				if !admitted {
					return nil
				}
				owned, err := campaign.runBelongsToModel(ctx, run.ID)
				if err != nil || !owned {
					return err
				}
				record, evaluated, err := campaign.evaluationForRun(ctx, run.ID)
				if err != nil {
					return err
				}
				if run.Outcome == runrecord.OutcomeSucceeded && !evaluated {
					return errors.New("evaluation: successful indexed run has no metric record")
				}
				if evaluated && record.Dataset != dataset {
					return errors.New("evaluation: indexed run dataset differs from plan")
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
				return nil
			})
		if err != nil {
			return nil, err
		}
		if page.Next == nil {
			break
		}
		query.Cursor = page.Next
	}
	return entries, nil
}

func (campaign *Campaign) runBelongsToModel(ctx context.Context, run artifact.ID) (bool, error) {
	parents, err := campaign.repository.Parents(ctx, run)
	if err != nil {
		return false, err
	}
	return slices.Contains(parents, artifact.Lineage{
		Child: run, Parent: campaign.identity.Model, Relation: artifact.RelationDependsOn,
	}), nil
}

func (campaign *Campaign) evaluationForRun(ctx context.Context, run artifact.ID) (runrecord.Evaluation, bool, error) {
	id, found, err := artifact.ResolveAlias(ctx, campaign.repository, runrecord.EvaluationRunAlias(run))
	if err != nil {
		return runrecord.Evaluation{}, false, err
	}
	if !found {
		return runrecord.Evaluation{}, false, nil
	}
	record, err := runrecord.RequireEvaluation(ctx, campaign.repository, id)
	if err != nil || record.Run != run || record.Recipe != campaign.identity.Recipe {
		return runrecord.Evaluation{}, false, errors.Join(err, errors.New("evaluation: indexed metric record is invalid"))
	}
	return record, true, nil
}
