package evaluation

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/benchmarkrecord"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
)

// BenchmarkSummary is the latest committed benchmark claim for one
// weights file: tier, wall, peak, and the headline rates, time to first
// token and load time lifted from the claim's evidence document. Producer
// facts are independent: absent or inconsistent facts stay unknown, and
// neither the claim's publication nor these facts prove source currency.
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
	CodeCommit           string      `json:"code_commit,omitzero"`
	Environment          artifact.ID `json:"environment,omitzero"`
	// Comparison binds a known full same-producer benchmark contract; it does
	// not establish source currency or cross-revision performance equivalence.
	Comparison artifact.ID `json:"comparison,omitzero"`
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
// under one acceptance/environment contract. Prompting is only a display label;
// Comparison binds full acceptance: executable plans include environment;
// reconstructed target reports bind exact targets/inputs without asserting
// known execution. Neither identity certifies current source.
type EvalSummary struct {
	Suite     string             `json:"suite"`
	Prompting string             `json:"prompting"`
	Record    artifact.ID        `json:"record,omitzero"`
	Metrics   map[string]float64 `json:"metrics"`
	// Cases is how many cases the metrics rest on, where the record's reader
	// counts them: a held-out verdict's records, a transcription report's
	// utterances. Zero where a suite's descriptor is what counts them.
	Cases uint64 `json:"cases,omitzero"`
	// Noise is the standard error of the headline metric where the record
	// keeps what it follows from: for a held-out verdict, that of the mean
	// of its records' scores. Zero where it does not.
	Noise float64 `json:"noise,omitzero"`
	// Constant is what an answer that reads no input scores on the headline
	// metric, where the record's reader can tell: for a held-out verdict judged
	// on label accuracy, always answering the commonest label or class. Nil
	// where the metric has no such answer or a suite's descriptor gives it.
	Constant *float64 `json:"constant,omitempty"`
	// Unread is how many of the answers generated under the chat-template
	// protocol named no option the evaluation could read. Each is counted
	// wrong: a model that begins to reason, or says the answer is not among
	// the options, is scored as one that chose badly. Zero where every answer
	// named one, and where answers were scored and not generated.
	Unread      uint64      `json:"unread,omitzero"`
	Comparison  artifact.ID `json:"comparison,omitzero"`
	Environment artifact.ID `json:"environment,omitzero"`
	CodeCommit  string      `json:"code_commit,omitzero"`
	// Directions says which way each metric improves: the scorecard judges
	// a held-out regression by it.
	Directions map[string]runrecord.Direction `json:"-"`
	// ExecutionKnown says the succeeded run, plan and typed environment agree.
	// It does not establish source currency or full measurement comparability.
	ExecutionKnown bool `json:"execution_known"`
	// Reproducible: the run's environment is one the store can reproduce;
	// false for a hosted model scored through the relay.
	Reproducible bool `json:"reproducible"`
	// Carried names the ReplayReceipt of a record that carried an earlier
	// result forward; zero when its run scored every case.
	Carried artifact.ID `json:"carried,omitzero"`
	// ScoredCommit is the code commit of the run that scored the cases the
	// metrics stand on: CodeCommit for a run that scored them all, and for a
	// carried result the commit of the run its chain of carries began at,
	// which CodeCommit, the carrying run's, is not. Empty when a link of the
	// chain cannot be read.
	ScoredCommit string `json:"scored_commit,omitzero"`
}

// unknownPrompting renders an absent or inconsistent execution authority. It
// is never an execution policy accepted by a plan.
const unknownPrompting = "unknown"

// recordExecution reads execution facts through their existing typed owners.
// Unknown authority retains the measurement and any identified replay receipt,
// without inventing a protocol or a reproducible environment.
func recordExecution(ctx context.Context, store *overgodb.Store, record runrecord.Evaluation) EvalSummary {
	summary := EvalSummary{Record: record.ID, Prompting: unknownPrompting}
	bound, err := runrecord.RequireRun(ctx, store, record.Run)
	if err != nil {
		return summary
	}
	var carried artifact.ID
	summary.Carried, carried, _ = carriedReceipt(ctx, store, bound)
	if carried.Valid() {
		summary.ScoredCommit = scoredCommit(ctx, store, carried)
	}
	if bound.Outcome != runrecord.OutcomeSucceeded || bound.Recipe != record.Recipe || len(bound.Inputs) != 1 {
		return summary
	}
	plan, err := loadEvidencePlan(ctx, store, bound.Inputs[0])
	if err != nil || plan.body.RuntimeRecipe != record.Recipe || plan.body.Dataset != record.Dataset ||
		plan.body.CodeCommit != bound.CodeCommit || plan.body.Environment != bound.Environment {
		return summary
	}
	policy, err := ReadExecutionPolicy(ctx, store, plan.Execution())
	if err != nil {
		return summary
	}
	environment, err := runrecord.RequireEnvironment(ctx, store, bound.Environment)
	if err != nil {
		return summary
	}
	comparison, err := plan.comparisonIdentity()
	if err != nil {
		return summary
	}
	summary.Comparison, summary.Environment, summary.CodeCommit = comparison, bound.Environment, bound.CodeCommit
	summary.Prompting, summary.Reproducible, summary.ExecutionKnown = policy.Prompting.Label(), environment.Reproducible(), true
	if counting := namedCounting(ctx, store, plan); len(counting) != 0 {
		summary.Prompting = strings.Join(append([]string{summary.Prompting}, counting...), protocolSeparator)
	}
	if policy.Prompting == PromptingRawCompletion && scoredGlued(ctx, store, plan) {
		summary.Prompting += protocolSeparator + gluedLetters
	}
	if policy.Prompting == PromptingChatTemplate {
		summary.Unread = unreadAnswers(ctx, store, bound)
	}
	return summary
}

// unreadAnswers counts, in the report of the run that scored a record's
// cases, the generated answers in which no option was read. A run that
// carried an earlier result forward is followed to the run that scored it. A
// report that cannot be read, or that holds no generated answer, counts none:
// the count says what is known and does not stand in the way of a reading.
func unreadAnswers(ctx context.Context, store *overgodb.Store, bound runrecord.Run) uint64 {
	for {
		_, earlier, readable := carriedReceipt(ctx, store, bound)
		if !readable {
			return 0
		}
		if !earlier.Valid() {
			break
		}
		record, err := runrecord.RequireEvaluation(ctx, store, earlier)
		if err != nil {
			return 0
		}
		if bound, err = runrecord.RequireRun(ctx, store, record.Run); err != nil {
			return 0
		}
	}
	if len(bound.Outputs) != 1 {
		return 0
	}
	content, found, err := artifact.ReadContent(ctx, store, bound.Outputs[0])
	if err != nil || !found {
		return 0
	}
	return countUnread(content.Data)
}

// countUnread counts the answers of a report's observations that were
// generated and named no option.
func countUnread(report []byte) uint64 {
	var answers struct {
		Observations []struct {
			Selected int    `json:"selected"`
			Raw      string `json:"raw"`
		} `json:"observations"`
	}
	if json.Unmarshal(report, &answers) != nil {
		return 0
	}
	var unread uint64
	for _, answer := range answers.Observations {
		if answer.Raw != "" && answer.Selected == unmatchedChoice {
			unread++
		}
	}
	return unread
}

// carriedReceipt names the replay receipt a run reports and the record that
// receipt carried forward; both are zero for a run that scored its own cases,
// and readable is false for a run that reports a receipt it cannot read, which
// is no run that scored its own.
func carriedReceipt(ctx context.Context, store *overgodb.Store, bound runrecord.Run) (receipt, carried artifact.ID, readable bool) {
	if len(bound.Outputs) != 1 {
		return artifact.ID{}, artifact.ID{}, true
	}
	descriptor, found, _ := store.Artifact(ctx, bound.Outputs[0])
	if !found || descriptor.Schema != replayReceiptContract.Schema {
		return artifact.ID{}, artifact.ID{}, true
	}
	named, err := RequireReplayReceipt(ctx, store, descriptor.ID)
	if err != nil {
		return artifact.ID{}, artifact.ID{}, false
	}
	if !named.Carried.Valid() {
		return artifact.ID{}, artifact.ID{}, true
	}
	return descriptor.ID, named.Carried, true
}

// scoredCommit is the code commit of the run that scored the cases a carried
// record's result stands on: its chain of carries followed to the first record
// whose run carried nothing, or empty when a link cannot be read, a run that
// reports a receipt it cannot read being one. Each record's identity holds the
// one it carried, so a chain cannot loop.
func scoredCommit(ctx context.Context, store *overgodb.Store, carried artifact.ID) string {
	for carried.Valid() {
		record, err := runrecord.RequireEvaluation(ctx, store, carried)
		if err != nil {
			return ""
		}
		bound, err := runrecord.RequireRun(ctx, store, record.Run)
		if err != nil {
			return ""
		}
		_, earlier, readable := carriedReceipt(ctx, store, bound)
		if !readable {
			return ""
		}
		if !earlier.Valid() {
			return bound.CodeCommit
		}
		carried = earlier
	}
	return ""
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

// claimEvidence reads a claim's first evidence document; an absent one is
// empty, so decoding it into a tolerant headline fails and leaves the
// headline as it was.
func claimEvidence(ctx context.Context, store *overgodb.Store, claim runrecord.CapabilityClaim) []byte {
	if len(claim.Evidence) == 0 {
		return nil
	}
	content, found, err := artifact.ReadContent(ctx, store, claim.Evidence[0])
	if err != nil || !found {
		return nil
	}
	return content.Data
}

// benchmarkEvidenceHeadline tolerantly lifts the headline metric out of
// a committed benchmark evidence document (the benchmark CLI's result
// JSON); an absent or differently shaped document leaves the claim's
// own fields standing.
type benchmarkEvidenceHeadline struct {
	CodeCommit    string                `json:"code_commit"`
	EnvironmentID artifact.ID           `json:"environment_id"`
	Environment   runrecord.Environment `json:"environment"`
	LoadMS        float64               `json:"load_ms"`
	Summary       struct {
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

// benchmarkEvidenceContract decodes execution facts independently of the
// headline. Unknown or malformed optional facts cannot erase measured values.
type benchmarkEvidenceContract struct {
	CachePrompt          bool                     `json:"cache_prompt"`
	BatchSequences       int                      `json:"batch_sequences"`
	Speculative          bool                     `json:"speculative"`
	ContextShift         bool                     `json:"context_shift"`
	Temperature          float64                  `json:"temperature"`
	TopK                 int                      `json:"top_k"`
	DeviceTopK           bool                     `json:"device_top_k"`
	ModelID              artifact.ID              `json:"model_id"`
	RecipeID             artifact.ID              `json:"recipe_id"`
	TokenizerContainer   artifact.ID              `json:"tokenizer_container"`
	ModuleDigest         string                   `json:"module_digest"`
	OptionsDigest        string                   `json:"options_digest"`
	AdapterIDs           []artifact.ID            `json:"adapter_ids"`
	TokensPerSequence    int                      `json:"tokens_per_sequence"`
	RequestedRuns        int                      `json:"requested_runs"`
	RequestedWarmup      int                      `json:"requested_warmup"`
	PromptSuite          []string                 `json:"prompt_suite"`
	WorkloadDigest       string                   `json:"workload_digest"`
	Residency            recipe.ResidencyPolicy   `json:"residency"`
	RealizedResidency    recipe.RealizedResidency `json:"realized_residency"`
	SamplingProtocol     string                   `json:"sampling_protocol"`
	SamplerOrder         json.RawMessage          `json:"sampler_order"`
	EndOfSequenceIgnored *bool                    `json:"end_of_sequence_ignored"`
}

func benchmarkComparison(data []byte, model artifact.ID, summary BenchmarkSummary) artifact.ID {
	var headline benchmarkEvidenceContract
	if json.Unmarshal(data, &headline) != nil {
		return artifact.ID{}
	}
	if headline.ModelID != model || summary.CodeCommit == "" || !summary.Environment.Valid() ||
		!headline.Residency.Valid() || headline.SamplingProtocol == "" || headline.EndOfSequenceIgnored == nil || len(headline.SamplerOrder) == 0 {
		return artifact.ID{}
	}
	var order []sampling.SamplerStage
	if json.Unmarshal(headline.SamplerOrder, &order) != nil {
		return artifact.ID{}
	}
	if len(order) == 0 {
		order = nil
	}
	configuration := benchmarkrecord.BenchmarkConfiguration{Tokens: headline.TokensPerSequence, Runs: headline.RequestedRuns,
		Warmup: headline.RequestedWarmup, CachePrompt: headline.CachePrompt, BatchSequences: headline.BatchSequences,
		Speculative: headline.Speculative, ContextShift: headline.ContextShift, Temperature: headline.Temperature,
		TopK: headline.TopK, DeviceTopK: headline.DeviceTopK, Adapters: headline.AdapterIDs}
	options, err := configuration.Digest()
	if err != nil || options != headline.OptionsDigest {
		return artifact.ID{}
	}
	workload, err := artifact.JSONID(artifact.KindEvidence, headline.PromptSuite)
	if err != nil || workload.DigestHex() != headline.WorkloadDigest {
		return artifact.ID{}
	}
	prompts := make([]string, len(headline.PromptSuite))
	for index, prompt := range headline.PromptSuite {
		if prompt == "" {
			return artifact.ID{}
		}
		id, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte(prompt))
		prompts[index] = id.DigestHex()
	}
	protocol, err := artifact.JSONID(artifact.KindProfile, struct {
		Residency        recipe.ResidencyPolicy  `json:"residency"`
		Sampling         string                  `json:"sampling"`
		Order            []sampling.SamplerStage `json:"order"`
		ContinueAfterEOG bool                    `json:"continue_after_eog"`
	}{headline.Residency, headline.SamplingProtocol, order, *headline.EndOfSequenceIgnored})
	if err != nil {
		return artifact.ID{}
	}
	identity := benchmarkrecord.BenchmarkIdentity{Model: model, Recipe: headline.RecipeID,
		TokenizerContainer: headline.TokenizerContainer, Environment: summary.Environment,
		CodeCommit: summary.CodeCommit, ModuleDigest: headline.ModuleDigest, WorkloadDigest: headline.WorkloadDigest,
		PromptDigests: prompts, RealizedResidency: headline.RealizedResidency}
	comparison, _ := identity.Comparison(configuration, protocol)
	return comparison
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
				document := claimEvidence(ctx, store, claim)
				var headline benchmarkEvidenceHeadline
				if json.Unmarshal(document, &headline) == nil {
					summary.PromptTokensPerSecond = headline.Summary.PromptTokensPerSecondP50
					summary.DecodeTokensPerSecond = headline.Summary.DecodeTokensPerSecondP50
					summary.DeviceAfterLoadBytes, summary.DeviceTotalBytes = headline.DeviceAfterLoadBytes, headline.Device.TotalMemoryBytes
					summary.RealizedResidency = headline.RealizedResidency
					summary.FirstTokenMS, summary.LoadMS = headline.Summary.FirstTokenMSP50, headline.LoadMS
					if headline.CodeCommit == claim.Commit {
						summary.CodeCommit = headline.CodeCommit
					}
					headline.Environment.ID = headline.EnvironmentID
					if headline.Environment.ValidateIdentity() == nil {
						summary.Environment = headline.EnvironmentID
					}
					summary.Comparison = benchmarkComparison(document, record.Model, summary)
				}
				keep(func(location string) {
					if _, seen := index.BenchmarksByLocation[location]; !seen {
						index.BenchmarksByLocation[location] = summary
					}
				})
			case claim.Capability == "long-form" && !guarded:
				guarded = true
				summary := GuardSummary{Record: record.ID}
				if json.Unmarshal(claimEvidence(ctx, store, claim), &summary) != nil {
					// A record whose evidence cannot be read states no guard, passed or failed.
					break
				}
				keep(func(location string) {
					if _, seen := index.GuardsByLocation[location]; !seen {
						index.GuardsByLocation[location] = summary
					}
				})
			}
		}
		return nil
	})
	seenSuite := map[artifact.ID]map[artifact.ID]bool{}
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
		summary := recordExecution(ctx, store, record)
		if seenSuite[record.Recipe] == nil {
			seenSuite[record.Recipe] = map[artifact.ID]bool{}
		}
		key := summary.Comparison
		if !summary.ExecutionKnown || !key.Valid() {
			// Unknowns cannot prove equivalence to another record.
			key = record.ID
		}
		if seenSuite[record.Recipe][key] {
			return nil
		}
		seenSuite[record.Recipe][key] = true
		metrics := make(map[string]float64, len(record.Metrics))
		directions := make(map[string]runrecord.Direction, len(record.Metrics))
		for _, metric := range record.Metrics {
			metrics[metric.Name], directions[metric.Name] = metric.Value, metric.Direction
		}
		summary.Suite, summary.Metrics, summary.Directions = suite, metrics, directions
		index.EvaluationsByRecipe[record.Recipe] = append(index.EvaluationsByRecipe[record.Recipe], summary)
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
