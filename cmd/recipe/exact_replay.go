package main

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Case outputs precede runtime cleanup. Only a bound terminal gate/run permits
// lifecycle reuse; equivalent historical publications share one outcome.
func retainedExactVerification(ctx context.Context, store *overgodb.Store, recipeID artifact.ID, revision string,
	environment, planID artifact.ID,
) (runrecord.Verification, artifact.ID, error) {
	var retained runrecord.Verification
	var report artifact.ID
	edges, err := store.Children(ctx, recipeID)
	if err != nil {
		return retained, report, err
	}
	for _, edge := range edges {
		if edge.Relation != artifact.RelationDependsOn || edge.Child.Kind() != artifact.KindEvidence {
			continue
		}
		content, found, err := artifact.ReadContent(ctx, store, edge.Child)
		if err != nil {
			return retained, report, err
		}
		if !found || content.Descriptor.MediaType != runrecord.GateMediaType || content.Descriptor.Schema != runrecord.GateSchema {
			continue
		}
		gate, err := runrecord.ParseGateResult(content.Data)
		if err != nil {
			return retained, report, err
		}
		if gate.ID != edge.Child || gate.Recipe != recipeID {
			return retained, report, errors.New("recipe: retained gate lineage differs")
		}
		if gate.CodeCommit != revision || gate.Environment != environment || len(gate.Steps) != 1 || gate.Steps[0].Name != "candidate-execution" {
			continue
		}
		fields := strings.Split(gate.Steps[0].Evidence, ";")
		// Both historical formats put contract, plan and report first; never
		// interpret failure text as authority, even if it contains semicolons.
		if len(fields) < 3 || strings.TrimSpace(fields[0]) != "contract=exact" || strings.TrimSpace(fields[1]) != "plan="+planID.String() {
			continue
		}
		reportText, declared := strings.CutPrefix(strings.TrimSpace(fields[2]), "report=")
		reportID, err := artifact.ParseID(reportText)
		if !declared || err != nil || reportID.Kind() != artifact.KindEvaluation {
			continue // Interrupted evaluations have no terminal campaign report.
		}
		parents, err := store.Parents(ctx, gate.ID)
		if err != nil {
			return retained, report, err
		}
		var candidate runrecord.Verification
		for _, parent := range parents {
			if parent.Relation != artifact.RelationProducedBy || parent.Parent.Kind() != artifact.KindRun {
				continue
			}
			if candidate.Run.ID != (artifact.ID{}) {
				return retained, report, errors.New("recipe: retained gate has multiple bound runs")
			}
			verify := runrecord.VerifyGateRun
			if gate.Outcome == runrecord.OutcomeFailed {
				verify = runrecord.VerifyFailedGateRun
			}
			candidate, err = verify(ctx, store, recipeID, gate.ID, parent.Parent)
			if err != nil {
				return retained, report, err
			}
		}
		if candidate.Run.ID == (artifact.ID{}) {
			return retained, report, errors.New("recipe: retained terminal gate has no bound run")
		}
		if retained.Gate.ID != (artifact.ID{}) {
			if report != reportID || retained.Gate.Outcome != gate.Outcome || retained.Gate.Steps[0].Evidence != gate.Steps[0].Evidence {
				return retained, report, errors.New("recipe: conflicting terminal exact verifications")
			}
			if retained.Gate.ID.String() < gate.ID.String() {
				continue
			}
		}
		retained, report = candidate, reportID
	}
	return retained, report, nil
}
