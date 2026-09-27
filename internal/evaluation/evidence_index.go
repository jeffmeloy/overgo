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
	PromptTokensPerSecond float64     `json:"prompt_tokens_per_second_p50,omitzero"`
	DecodeTokensPerSecond float64     `json:"decode_tokens_per_second_p50,omitzero"`
	WallNS                uint64      `json:"wall_ns,omitzero"`
	PeakDeviceBytes       uint64      `json:"peak_device_bytes,omitzero"`
	Record                artifact.ID `json:"record,omitzero"`
}

// EvalSummary is one suite's latest committed evaluation for a recipe
// under one prompting protocol; the raw-completion anchor and the
// chat-template pass of the same suite are two summaries.
type EvalSummary struct {
	Suite     string             `json:"suite"`
	Prompting string             `json:"prompting"`
	Record    artifact.ID        `json:"record,omitzero"`
	Metrics   map[string]float64 `json:"metrics"`
	// Reproducible: the run's environment is one the store can reproduce;
	// false for a hosted model scored through the relay.
	Reproducible bool `json:"reproducible"`
}

// recordExecution reads the prompting protocol an evaluation record ran
// under (its run's plan input binds the execution policy) and whether its
// environment is reproducible. A record whose plan cannot be read reports
// the raw anchor, the only protocol earlier records could have run; a run
// whose environment cannot be read reports it reproducible, as every
// earlier environment was.
func recordExecution(ctx context.Context, store *overgodb.Store, record runrecord.Evaluation) (Prompting, bool) {
	bound, err := runrecord.RequireRun(ctx, store, record.Run)
	if err != nil || len(bound.Inputs) == 0 {
		return PromptingRawCompletion, true
	}
	reproducible := true
	if environment, err := runrecord.RequireEnvironment(ctx, store, bound.Environment); err == nil {
		reproducible = environment.Reproducible()
	}
	content, found, err := artifact.ReadContent(ctx, store, bound.Inputs[0])
	if err != nil || !found {
		return PromptingRawCompletion, reproducible
	}
	plan, err := ParsePlan(content.Data)
	if err != nil {
		return PromptingRawCompletion, reproducible
	}
	policy, err := ReadExecutionPolicy(ctx, store, plan.Execution())
	if err != nil {
		return PromptingRawCompletion, reproducible
	}
	return policy.Prompting, reproducible
}

// EvidenceIndex joins the store's committed evidence back onto models:
// benchmark and long-form claims keyed by the registered weights location
// (the claim identifies the weights digest, catalogs identify manifests,
// and the on-disk file is the identity both register), evaluations keyed
// by recipe. Every consumer -- the workbench catalog, the training
// bracket -- reads the same committed records.
type EvidenceIndex struct {
	BenchmarksByLocation map[string]BenchmarkSummary
	GuardsByLocation     map[string]GuardSummary
	EvaluationsByRecipe  map[artifact.ID][]EvalSummary
}

// GuardSummary is a model's latest long-form record, lifted tolerantly from
// its result document: whether it passed its floors, and the commit and
// inference surface it measured, which a later code change can expire.
type GuardSummary struct {
	Record  artifact.ID `json:"-"`
	Commit  string      `json:"commit"`
	Surface string      `json:"surface"`
	Verdict struct {
		Passed bool `json:"passed"`
	} `json:"verdict"`
}

// readClaimEvidence decodes a claim's first evidence document into a
// tolerant headline; an absent or differently shaped document reports false
// and leaves the headline as it was.
func readClaimEvidence(ctx context.Context, store *overgodb.Store, claim runrecord.CapabilityClaim, headline any) bool {
	if len(claim.Evidence) == 0 {
		return false
	}
	content, found, err := artifact.ReadContent(ctx, store, claim.Evidence[0])
	return err == nil && found && json.Unmarshal(content.Data, headline) == nil
}

// benchmarkEvidenceHeadline tolerantly lifts the headline metric out of
// a committed benchmark evidence document (the benchmark CLI's result
// JSON); an absent or differently shaped document leaves the claim's
// own fields standing.
type benchmarkEvidenceHeadline struct {
	Summary struct {
		PromptTokensPerSecondP50 float64 `json:"prompt_tokens_per_second_p50"`
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
		GuardsByLocation:     map[string]GuardSummary{},
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
		// keep indexes a record's first claim of a kind under each weights
		// file the model registers, unless a newer record holds the file.
		keep := func(add func(string)) {
			for _, location := range locations {
				if location.Kind == artifact.LocationFile {
					add(location.Value)
				}
			}
		}
		benchmarked, guarded := false, false
		for _, claim := range record.Claims {
			switch {
			case claim.Capability == "benchmark" && !benchmarked:
				benchmarked = true
				summary := BenchmarkSummary{
					Tier: string(claim.Tier), WallNS: claim.WallNS, PeakDeviceBytes: claim.PeakDeviceBytes,
					Record: record.ID,
				}
				var headline benchmarkEvidenceHeadline
				if readClaimEvidence(ctx, store, claim, &headline) {
					summary.PromptTokensPerSecond = headline.Summary.PromptTokensPerSecondP50
					summary.DecodeTokensPerSecond = headline.Summary.DecodeTokensPerSecondP50
				}
				keep(func(location string) {
					if _, seen := index.BenchmarksByLocation[location]; !seen {
						index.BenchmarksByLocation[location] = summary
					}
				})
			case claim.Capability == "long-form" && !guarded:
				guarded = true
				summary := GuardSummary{Record: record.ID}
				readClaimEvidence(ctx, store, claim, &summary)
				keep(func(location string) {
					if _, seen := index.GuardsByLocation[location]; !seen {
						index.GuardsByLocation[location] = summary
					}
				})
			}
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
		protocol, reproducible := recordExecution(ctx, store, record)
		prompting := protocol.Label()
		if seenSuite[record.Recipe] == nil {
			seenSuite[record.Recipe] = map[string]bool{}
		}
		key := suite + "\x00" + prompting
		if seenSuite[record.Recipe][key] {
			return nil
		}
		seenSuite[record.Recipe][key] = true
		metrics := make(map[string]float64, len(record.Metrics))
		for _, metric := range record.Metrics {
			metrics[metric.Name] = metric.Value
		}
		index.EvaluationsByRecipe[record.Recipe] = append(index.EvaluationsByRecipe[record.Recipe], EvalSummary{
			Suite: suite, Prompting: prompting, Record: record.ID, Metrics: metrics, Reproducible: reproducible,
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

// DerivedSuiteDescriptors compiles the store's benchmark catalog and keys
// every suite's descriptor by its source name, so a report can state each
// suite's kind and case count beside the records that ran it.
func DerivedSuiteDescriptors(ctx context.Context, store *overgodb.Store, authorities ExactAuthorities) map[string]SuiteDescriptor {
	descriptors := map[string]SuiteDescriptor{}
	suites, _, err := DeriveStoreSuites(ctx, store, authorities)
	if err != nil {
		return descriptors
	}
	for _, suite := range suites {
		descriptor := suite.Descriptor()
		descriptors[descriptor.Source] = descriptor
	}
	return descriptors
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
