package server

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// catalogBenchmarkSummary is the perf evidence beside a catalog entry:
// the strongest committed benchmark claim's headline numbers.
type catalogBenchmarkSummary struct {
	Tier                  string  `json:"tier"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second_p50,omitempty"`
	WallNS                uint64  `json:"wall_ns,omitempty"`
	PeakDeviceBytes       uint64  `json:"peak_device_bytes,omitempty"`
}

// catalogEvalSummary is one suite's latest committed result for a
// recipe: the suite's source name and its metric values.
type catalogEvalSummary struct {
	Suite   string             `json:"suite"`
	Metrics map[string]float64 `json:"metrics"`
}

// catalogEvidenceIndex joins the store's committed evidence back onto
// catalog entries: benchmark claims keyed by model identity, latest
// evaluation metrics keyed by recipe. Every number shown beside a
// model in the workbench is a committed record, never a cached
// impression.
type catalogEvidenceIndex struct {
	benchmarks  map[artifact.ID]catalogBenchmarkSummary
	evaluations map[artifact.ID][]catalogEvalSummary
}

// benchmarkEvidenceSummary tolerantly lifts the headline metric out of
// a committed benchmark evidence document (the benchmark CLI's result
// JSON); an absent or differently shaped document leaves the claim's
// own fields standing.
type benchmarkEvidenceSummary struct {
	Summary struct {
		DecodeTokensPerSecondP50 float64 `json:"decode_tokens_per_second_p50"`
	} `json:"summary"`
}

func (h *Handler) catalogEvidence(ctx context.Context, suiteNames map[artifact.ID]string) catalogEvidenceIndex {
	index := catalogEvidenceIndex{
		benchmarks:  map[artifact.ID]catalogBenchmarkSummary{},
		evaluations: map[artifact.ID][]catalogEvalSummary{},
	}
	documents, ok := any(h.config.Repository).(overgodb.DocumentReader)
	if !ok || h.config.Repository == nil {
		return index
	}
	// Newest-first scans keep the first summary seen per key, so each
	// entry reports its latest committed evidence.
	_, _ = overgodb.VisitDecodedDocuments(ctx, documents, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema,
		}},
		Order: overgodb.DocumentNewestFirst, MaxResults: h.config.MaxStoredResponses,
	}, runrecord.ParseModelVerification, func(_ overgodb.DocumentView, record runrecord.ModelVerification) error {
		if _, seen := index.benchmarks[record.Model]; seen {
			return nil
		}
		for _, claim := range record.Claims {
			if claim.Capability != "benchmark" {
				continue
			}
			summary := catalogBenchmarkSummary{
				Tier: string(claim.Tier), WallNS: claim.WallNS, PeakDeviceBytes: claim.PeakDeviceBytes,
			}
			if len(claim.Evidence) > 0 {
				if content, found, err := artifact.ReadContent(ctx, h.config.Repository, claim.Evidence[0]); err == nil && found {
					var evidence benchmarkEvidenceSummary
					if json.Unmarshal(content.Data, &evidence) == nil {
						summary.DecodeTokensPerSecond = evidence.Summary.DecodeTokensPerSecondP50
					}
				}
			}
			index.benchmarks[record.Model] = summary
			break
		}
		return nil
	})
	seenSuite := map[artifact.ID]map[string]bool{}
	_, _ = overgodb.VisitDecodedDocuments(ctx, documents, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvaluation, MediaType: runrecord.EvaluationMediaType, Schema: runrecord.EvaluationSchema,
		}},
		Order: overgodb.DocumentNewestFirst, MaxResults: h.config.MaxStoredResponses,
	}, runrecord.ParseEvaluation, func(_ overgodb.DocumentView, record runrecord.Evaluation) error {
		suite, named := suiteNames[record.Dataset]
		if !named {
			return nil
		}
		if seenSuite[record.Recipe] == nil {
			seenSuite[record.Recipe] = map[string]bool{}
		}
		if seenSuite[record.Recipe][suite] {
			return nil
		}
		seenSuite[record.Recipe][suite] = true
		metrics := make(map[string]float64, len(record.Metrics))
		for _, metric := range record.Metrics {
			metrics[metric.Name] = metric.Value
		}
		index.evaluations[record.Recipe] = append(index.evaluations[record.Recipe], catalogEvalSummary{
			Suite: suite, Metrics: metrics,
		})
		return nil
	})
	return index
}

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
