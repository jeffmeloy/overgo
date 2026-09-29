package evaluation

import (
	"context"
	"encoding/json"
	"maps"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// BenchmarkSummary is the latest committed benchmark claim for one
// weights file: tier, wall, peak, and the headline rates, time to first
// token and load time lifted from the claim's evidence document.
type BenchmarkSummary struct {
	Tier                  string  `json:"tier"`
	PromptTokensPerSecond float64 `json:"prompt_tokens_per_second_p50,omitzero"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second_p50,omitzero"`
	FirstTokenMS          float64 `json:"ttft_ms_p50,omitzero"`
	LoadMS                float64 `json:"load_ms,omitzero"`
	WallNS                uint64  `json:"wall_ns,omitzero"`
	PeakDeviceBytes       uint64  `json:"peak_device_bytes,omitzero"`
	// DeviceAfterLoadBytes, DeviceTotalBytes and RealizedResidency are the
	// device the benchmark loaded onto, as its evidence document records.
	DeviceAfterLoadBytes uint64      `json:"device_after_load_bytes,omitzero"`
	DeviceTotalBytes     uint64      `json:"device_total_bytes,omitzero"`
	RealizedResidency    string      `json:"realized_residency,omitzero"`
	Record               artifact.ID `json:"record,omitzero"`
}

// headlineMetrics orders the metric that stands for an evaluation: an
// accuracy when the suite scores one, else the suite's own aggregate.
var headlineMetrics = []string{"accuracy", "prompt-strict", "perplexity", "mean"}

// HeadlineMetric names the metric that stands for an evaluation and its
// value: the first of the ordered headline names the metrics carry, else
// the alphabetically first. ok is false when there are no metrics.
func HeadlineMetric(metrics map[string]float64) (name string, value float64, ok bool) {
	for _, name := range headlineMetrics {
		if value, found := metrics[name]; found {
			return name, value, true
		}
	}
	names := slices.Sorted(maps.Keys(metrics))
	if len(names) == 0 {
		return "", 0, false
	}
	return names[0], metrics[names[0]], true
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
func recordExecution(ctx context.Context, store *overgodb.Store, record runrecord.Evaluation) (prompting, bool) {
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
	// Rungs are the prompt lengths the run climbed, each with the device
	// peak since the model opened, on the device total it ran on.
	Rungs []struct {
		Measure struct {
			PromptTokens int `json:"prompt_tokens"`
			Memory       struct {
				PeakBytes uint64 `json:"peakBytes"`
			} `json:"memory"`
		} `json:"measure"`
	} `json:"rungs"`
	Device deviceMemory `json:"device"`
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
	LoadMS  float64 `json:"load_ms"`
	Summary struct {
		PromptTokensPerSecondP50 float64 `json:"prompt_tokens_per_second_p50"`
		DecodeTokensPerSecondP50 float64 `json:"decode_tokens_per_second_p50"`
		FirstTokenMSP50          float64 `json:"ttft_ms_p50"`
	} `json:"summary"`
	DeviceAfterLoadBytes uint64       `json:"device_after_load_bytes"`
	RealizedResidency    string       `json:"realized_residency"`
	Device               deviceMemory `json:"device"`
}

// deviceMemory is the device total a run measured on.
type deviceMemory struct {
	TotalMemoryBytes uint64 `json:"totalMemoryBytes"`
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
					summary.DeviceAfterLoadBytes, summary.DeviceTotalBytes = headline.DeviceAfterLoadBytes, headline.Device.TotalMemoryBytes
					summary.RealizedResidency = headline.RealizedResidency
					summary.FirstTokenMS, summary.LoadMS = headline.Summary.FirstTokenMSP50, headline.LoadMS
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

// DerivedSuiteCatalog compiles the store's benchmark catalog once, without
// binding a model, and returns its suites' source names keyed by dataset
// identity and their descriptors keyed by source name: a report reads which
// records ran which suite, and each suite's kind and case count.
func DerivedSuiteCatalog(ctx context.Context, store *overgodb.Store, authorities ExactAuthorities) (map[artifact.ID]string, map[string]SuiteDescriptor) {
	names, descriptors := map[artifact.ID]string{}, map[string]SuiteDescriptor{}
	suites, _, err := DeriveStoreSuites(ctx, store, authorities)
	if err != nil {
		return names, descriptors
	}
	for _, suite := range suites {
		descriptor := suite.Descriptor()
		descriptors[descriptor.Source] = descriptor
		if descriptor.Dataset.Valid() {
			names[descriptor.Dataset] = descriptor.Source
		}
	}
	return names, descriptors
}
