package evaluation

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// BenchmarkSummary is the latest committed benchmark claim for one
// weights file: tier, wall, peak, and the headline decode rate lifted
// from the claim's evidence document.
type BenchmarkSummary struct {
	Tier                  string      `json:"tier"`
	DecodeTokensPerSecond float64     `json:"decode_tokens_per_second_p50,omitempty"`
	WallNS                uint64      `json:"wall_ns,omitempty"`
	PeakDeviceBytes       uint64      `json:"peak_device_bytes,omitempty"`
	Record                artifact.ID `json:"record,omitzero"`
}

// EvalSummary is one suite's latest committed evaluation for a recipe.
type EvalSummary struct {
	Suite   string             `json:"suite"`
	Record  artifact.ID        `json:"record,omitzero"`
	Metrics map[string]float64 `json:"metrics"`
}

// EvidenceIndex joins the store's committed evidence back onto models:
// benchmark claims keyed by the registered weights location (the claim
// identifies the weights digest, catalogs identify manifests, and the
// on-disk file is the identity both register), evaluations keyed by
// recipe. Every consumer -- the workbench catalog, the training
// bracket -- reads the same committed records.
type EvidenceIndex struct {
	BenchmarksByLocation map[string]BenchmarkSummary
	EvaluationsByRecipe  map[artifact.ID][]EvalSummary
}

// benchmarkEvidenceHeadline tolerantly lifts the headline metric out of
// a committed benchmark evidence document (the benchmark CLI's result
// JSON); an absent or differently shaped document leaves the claim's
// own fields standing.
type benchmarkEvidenceHeadline struct {
	Summary struct {
		DecodeTokensPerSecondP50 float64 `json:"decode_tokens_per_second_p50"`
	} `json:"summary"`
}

// LatestEvidence scans newest-first and keeps the first summary per
// key, so each entry reports its latest committed evidence. suiteNames
// maps evaluation dataset identities to display names; evaluations
// against unnamed datasets are skipped.
func LatestEvidence(
	ctx context.Context,
	store *overgodb.Store,
	suiteNames map[artifact.ID]string,
	limit int,
) EvidenceIndex {
	index := EvidenceIndex{
		BenchmarksByLocation: map[string]BenchmarkSummary{},
		EvaluationsByRecipe:  map[artifact.ID][]EvalSummary{},
	}
	if store == nil {
		return index
	}
	_, _ = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema,
		}},
		Order: overgodb.DocumentNewestFirst, MaxResults: limit,
	}, runrecord.ParseModelVerification, func(_ overgodb.DocumentView, record runrecord.ModelVerification) error {
		locations, err := store.Locations(ctx, record.Model)
		if err != nil || len(locations) == 0 {
			return nil
		}
		for _, claim := range record.Claims {
			if claim.Capability != "benchmark" {
				continue
			}
			summary := BenchmarkSummary{
				Tier: string(claim.Tier), WallNS: claim.WallNS, PeakDeviceBytes: claim.PeakDeviceBytes,
				Record: record.ID,
			}
			if len(claim.Evidence) > 0 {
				if content, found, err := artifact.ReadContent(ctx, store, claim.Evidence[0]); err == nil && found {
					var headline benchmarkEvidenceHeadline
					if json.Unmarshal(content.Data, &headline) == nil {
						summary.DecodeTokensPerSecond = headline.Summary.DecodeTokensPerSecondP50
					}
				}
			}
			for _, location := range locations {
				if location.Kind != artifact.LocationFile {
					continue
				}
				if _, seen := index.BenchmarksByLocation[location.Value]; !seen {
					index.BenchmarksByLocation[location.Value] = summary
				}
			}
			break
		}
		return nil
	})
	seenSuite := map[artifact.ID]map[string]bool{}
	_, _ = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvaluation, MediaType: runrecord.EvaluationMediaType, Schema: runrecord.EvaluationSchema,
		}},
		Order: overgodb.DocumentNewestFirst, MaxResults: limit,
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
		index.EvaluationsByRecipe[record.Recipe] = append(index.EvaluationsByRecipe[record.Recipe], EvalSummary{
			Suite: suite, Record: record.ID, Metrics: metrics,
		})
		return nil
	})
	return index
}

// ListingAuthorities admit suite compilation for descriptor reads
// alone -- listing what would run, or naming a suite's dataset.
// Running a suite binds the real model, recipe, and environment.
func ListingAuthorities() ExactAuthorities {
	id := func(kind artifact.Kind, name string) artifact.ID {
		value, err := artifact.JSONID(kind, name)
		if err != nil {
			panic(err)
		}
		return value
	}
	return ExactAuthorities{
		ModelDefinition: id(artifact.KindModelDefinition, "evaluation/listing"),
		RuntimeRecipe:   id(artifact.KindRecipe, "evaluation/listing"),
		CodeCommit:      "0123456789abcdef0123456789abcdef01234567",
		Environment:     id(artifact.KindEvidence, "evaluation/listing"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
}

// DerivedSuiteNames maps the store's derived suite dataset identities
// to their source names without binding a model: the same catalog
// derivation the workspace performs, compiled under listing
// authorities purely to read descriptors.
func DerivedSuiteNames(ctx context.Context, store *overgodb.Store, authorities ExactAuthorities) map[artifact.ID]string {
	names := map[artifact.ID]string{}
	suites, _, err := DeriveStoreSuites(ctx, store, authorities)
	if err != nil {
		return names
	}
	for _, suite := range suites {
		descriptor := suite.Descriptor()
		if descriptor.Dataset.Valid() {
			names[descriptor.Dataset] = descriptor.Source
		}
	}
	return names
}
