package server

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
)

// workspaceSuiteNames maps suite dataset identities to their derived
// source names through the evaluation workspace's compiled suites, so
// evidence rows name the suite a human recognizes.
func (h *Handler) workspaceSuiteNames(ctx context.Context) map[artifact.ID]string {
	names := map[artifact.ID]string{}
	workspace := h.config.Evaluation
	if workspace == nil {
		return names
	}
	capabilities, err := workspace.EvaluationCapabilities(ctx)
	if err != nil {
		return names
	}
	for _, capability := range capabilities {
		if capability.Suite.Dataset.Valid() {
			names[capability.Suite.Dataset] = capability.Suite.Source
		}
	}
	return names
}

// catalogEvidence joins the store's committed evidence onto catalog
// entries through the shared index: every number shown beside a model
// in the workbench is a committed record, never a cached impression.
func (h *Handler) catalogEvidence(ctx context.Context, suiteNames map[artifact.ID]string) evaluation.EvidenceIndex {
	if h.config.Repository == nil {
		return evaluation.EvidenceIndex{
			BenchmarksByLocation: map[string]evaluation.BenchmarkSummary{},
			EvaluationsByRecipe:  map[artifact.ID][]evaluation.EvalSummary{},
		}
	}
	return evaluation.LatestEvidence(ctx, h.config.Repository, suiteNames, h.config.MaxStoredResponses)
}
