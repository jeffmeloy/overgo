// Package benchmarkrecord holds exact, lossless paired inference measurements.
package benchmarkrecord

import (
	"cmp"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	// BenchmarkComparisonMediaType identifies immutable paired benchmark evidence.
	BenchmarkComparisonMediaType = "application/vnd.overgo.benchmark-comparison+json"
	// BenchmarkComparisonSchema identifies its wire contract.
	BenchmarkComparisonSchema = "overgo/benchmark-comparison/v1"
)

// BenchmarkConfiguration includes every execution and sampling switch that
// can change the measured output or cost. Prompt content is bound separately.
type BenchmarkConfiguration struct {
	Tokens         int           `json:"tokens"`
	Runs           int           `json:"runs"`
	Warmup         int           `json:"warmup"`
	CachePrompt    bool          `json:"cache_prompt"`
	BatchSequences int           `json:"batch_sequences"`
	ContextShift   bool          `json:"context_shift"`
	Speculative    bool          `json:"speculative"`
	Temperature    float64       `json:"temperature"`
	TopK           int           `json:"top_k"`
	DeviceTopK     bool          `json:"device_top_k"`
	Adapters       []artifact.ID `json:"adapters"`
}

// Digest identifies every execution switch and adapter in this configuration.
func (configuration BenchmarkConfiguration) Digest() (string, error) {
	id, err := artifact.JSONID(artifact.KindEvidence, configuration)
	return id.DigestHex(), err
}

// BenchmarkIdentity binds both arms to exact model, code, platform, and workload facts.
type BenchmarkIdentity struct {
	Model              artifact.ID              `json:"model"`
	Recipe             artifact.ID              `json:"recipe"`
	TokenizerContainer artifact.ID              `json:"tokenizer_container"`
	Environment        artifact.ID              `json:"environment"`
	CodeCommit         string                   `json:"code_commit"`
	ModuleDigest       string                   `json:"module_digest"`
	WorkloadDigest     string                   `json:"workload_digest"`
	PromptDigests      []string                 `json:"prompt_digests"`
	RealizedResidency  recipe.RealizedResidency `json:"realized_residency"`
}

// BenchmarkObservation is a raw measured run, not a summary or an estimate.
type BenchmarkObservation struct {
	PromptDigest            string   `json:"prompt_digest"`
	TokenDigests            []string `json:"token_digests"`
	OutputTokens            int      `json:"output_tokens"`
	PromptTokens            int      `json:"prompt_tokens"`
	CachedPromptTokens      int      `json:"cached_prompt_tokens"`
	HostLogitTokens         int      `json:"host_logit_tokens"`
	DeviceSelectedTokens    int      `json:"device_selected_tokens"`
	DeviceTopKTokens        int      `json:"device_top_k_tokens"`
	PromptMilliseconds      float64  `json:"prompt_ms"`
	PromptTokensPerSecond   float64  `json:"prompt_tokens_per_second"`
	EndToEndTokensPerSecond float64  `json:"end_to_end_tokens_per_second"`
	DecodeTokensPerSecond   float64  `json:"decode_tokens_per_second"`
	HostToDeviceCopies      uint64   `json:"host_to_device_copies"`
	DeviceToHostCopies      uint64   `json:"device_to_host_copies"`
	DeviceToDeviceCopies    uint64   `json:"device_to_device_copies"`
	DeviceToDeviceBytes     uint64   `json:"device_to_device_bytes"`
	DeviceMemsets           uint64   `json:"device_memsets"`
	DeviceMemsetBytes       uint64   `json:"device_memset_bytes"`
	GraphInstantiations     uint64   `json:"graph_instantiations"`
	GraphUpdates            uint64   `json:"graph_updates"`
	GraphLaunches           uint64   `json:"graph_launches"`
	TotalMilliseconds       float64  `json:"total_ms"`
	TTFTMilliseconds        float64  `json:"ttft_ms"`
	DecodeMilliseconds      float64  `json:"decode_ms"`
	KernelLaunches          uint64   `json:"kernel_launches"`
	StreamSynchronizations  uint64   `json:"stream_synchronizations"`
	HostToDeviceBytes       uint64   `json:"host_to_device_bytes"`
	DeviceToHostBytes       uint64   `json:"device_to_host_bytes"`
}

// BenchmarkPair retains one alternating-order pair of raw observations.
type BenchmarkPair struct {
	Index     int                  `json:"index"`
	First     string               `json:"first"`
	Baseline  BenchmarkObservation `json:"baseline"`
	Candidate BenchmarkObservation `json:"candidate"`
}

// BenchmarkComparisonRecord preserves interleaved raw pairs and an explicitly
// descriptive latency delta. An all-faster result is not a significance claim.
type BenchmarkComparisonRecord struct {
	Version                 uint16                 `json:"version"`
	Identity                BenchmarkIdentity      `json:"identity"`
	Factor                  string                 `json:"factor"`
	Baseline                BenchmarkConfiguration `json:"baseline"`
	Candidate               BenchmarkConfiguration `json:"candidate"`
	Exclusive               bool                   `json:"exclusive"`
	SharedLoadMilliseconds  float64                `json:"shared_load_ms"`
	SharedDevicePeakBytes   uint64                 `json:"shared_device_peak_bytes"`
	Warmups                 []BenchmarkObservation `json:"warmups"`
	Pairs                   []BenchmarkPair        `json:"pairs"`
	Outcome                 string                 `json:"outcome"`
	MedianDeltaMilliseconds float64                `json:"median_delta_ms"`
	CacheUpperBound         bool                   `json:"cache_upper_bound"`
	ID                      artifact.ID            `json:"-"`
}

var benchmarkComparisonCodec = artifact.JSONDocumentCodec(
	"run record benchmark comparison", artifact.KindEvidence,
	BenchmarkComparisonMediaType, BenchmarkComparisonSchema,
	canonicalizeBenchmarkComparison,
	func(value BenchmarkComparisonRecord) artifact.ID { return value.ID },
	func(value *BenchmarkComparisonRecord, id artifact.ID) { value.ID = id },
	cloneBenchmarkComparison,
)

// NewBenchmarkComparisonRecord validates and identifies one lossless paired result.
func NewBenchmarkComparisonRecord(value BenchmarkComparisonRecord) (BenchmarkComparisonRecord, error) {
	return benchmarkComparisonCodec.NewInitial(value)
}

// ValidateIdentity checks that the record still matches its content ID.
func (value BenchmarkComparisonRecord) ValidateIdentity() error {
	return benchmarkComparisonCodec.ValidateIdentity(value)
}

// Content returns the typed immutable document bytes.
func (value BenchmarkComparisonRecord) Content() (artifact.Content, error) {
	return benchmarkComparisonCodec.Content(value)
}

// Batch prepares the comparison and its exact source lineage for publication.
func (value BenchmarkComparisonRecord) Batch(key string) (artifact.Batch, error) {
	return benchmarkComparisonCodec.Batch(key, value, value.Lineage(), nil)
}

// Lineage cites the model, recipe, environment, and adapter authorities.
func (value BenchmarkComparisonRecord) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Identity.Model, value.Identity.Recipe, value.Identity.Environment}
	parents = append(parents, value.Baseline.Adapters...)
	parents = append(parents, value.Candidate.Adapters...)
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

func cloneBenchmarkComparison(value BenchmarkComparisonRecord) BenchmarkComparisonRecord {
	value.Identity.PromptDigests = slices.Clone(value.Identity.PromptDigests)
	value.Baseline.Adapters = slices.Clone(value.Baseline.Adapters)
	value.Candidate.Adapters = slices.Clone(value.Candidate.Adapters)
	value.Warmups = slices.Clone(value.Warmups)
	for index := range value.Warmups {
		value.Warmups[index].TokenDigests = slices.Clone(value.Warmups[index].TokenDigests)
	}
	value.Pairs = slices.Clone(value.Pairs)
	for index := range value.Pairs {
		value.Pairs[index].Baseline.TokenDigests = slices.Clone(value.Pairs[index].Baseline.TokenDigests)
		value.Pairs[index].Candidate.TokenDigests = slices.Clone(value.Pairs[index].Candidate.TokenDigests)
	}
	return value
}

func canonicalizeBenchmarkComparison(value *BenchmarkComparisonRecord) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Exclusive ||
		len(value.Pairs) == 0 || len(value.Pairs) > 1000 ||
		!validBenchmarkIdentity(value.Identity) || !validBenchmarkConfiguration(value.Baseline) ||
		!validBenchmarkConfiguration(value.Candidate) || value.Baseline.Runs != len(value.Pairs) ||
		value.Candidate.Runs != len(value.Pairs) ||
		2*value.Baseline.Warmup != len(value.Warmups) || value.Baseline.Warmup != value.Candidate.Warmup ||
		!finiteNonnegative(value.SharedLoadMilliseconds) {
		return errors.New("run record: incomplete benchmark comparison identity or samples")
	}
	if value.CacheUpperBound != (value.Factor == "cache_prompt" && len(value.Identity.PromptDigests) == 1) {
		return errors.New("run record: cache upper-bound label differs from workload")
	}
	if value.Baseline.Temperature != 0 || value.Candidate.Temperature != 0 ||
		!singleBenchmarkFactor(value.Baseline, value.Candidate, value.Factor) {
		return errors.New("run record: benchmark comparison must change exactly one lossless factor")
	}
	expectedSequences := cmp.Or(value.Baseline.BatchSequences, 1)
	for index := 0; index < len(value.Warmups); index += 2 {
		baseline, candidate := value.Warmups[index], value.Warmups[index+1]
		if !validBenchmarkObservation(baseline, expectedSequences) ||
			!validBenchmarkObservation(candidate, expectedSequences) ||
			baseline.PromptDigest != value.Identity.PromptDigests[0] ||
			candidate.PromptDigest != baseline.PromptDigest ||
			baseline.OutputTokens != candidate.OutputTokens ||
			!slices.Equal(baseline.TokenDigests, candidate.TokenDigests) {
			return errors.New("run record: benchmark warmup pair changes lossless output")
		}
	}
	deltas := make([]float64, len(value.Pairs))
	faster, slower := 0, 0
	for index, pair := range value.Pairs {
		first := "baseline"
		if index%2 == 1 {
			first = "candidate"
		}
		if pair.Index != index || pair.First != first ||
			!validBenchmarkObservation(pair.Baseline, expectedSequences) ||
			!validBenchmarkObservation(pair.Candidate, expectedSequences) ||
			pair.Baseline.PromptDigest != pair.Candidate.PromptDigest ||
			pair.Baseline.PromptDigest != value.Identity.PromptDigests[index%len(value.Identity.PromptDigests)] ||
			pair.Baseline.OutputTokens != pair.Candidate.OutputTokens ||
			pair.Baseline.OutputTokens != value.Baseline.Tokens*expectedSequences ||
			!slices.Equal(pair.Baseline.TokenDigests, pair.Candidate.TokenDigests) {
			return fmt.Errorf("run record: benchmark pair %d is incomparable or changes lossless output", index)
		}
		delta := pair.Candidate.TotalMilliseconds - pair.Baseline.TotalMilliseconds
		deltas[index] = delta
		if delta < 0 {
			faster++
		} else if delta > 0 {
			slower++
		}
	}
	slices.Sort(deltas)
	median := deltas[len(deltas)/2]
	if len(deltas)%2 == 0 {
		median = (deltas[len(deltas)/2-1] + median) / 2
	}
	outcome := "mixed"
	switch {
	case faster == len(deltas):
		outcome = "all-pairs-faster"
	case slower == len(deltas):
		outcome = "all-pairs-slower"
	case faster == 0 && slower == 0:
		outcome = "tie"
	}
	if value.Outcome != "" && value.Outcome != outcome || value.MedianDeltaMilliseconds != 0 && value.MedianDeltaMilliseconds != median {
		return errors.New("run record: benchmark comparison verdict differs from raw pairs")
	}
	value.Outcome = outcome
	value.MedianDeltaMilliseconds = median
	return nil
}

func validBenchmarkIdentity(identity BenchmarkIdentity) bool {
	return identity.Model.Kind() == artifact.KindModel && identity.Recipe.Kind() == artifact.KindRecipe &&
		identity.TokenizerContainer == identity.Model && identity.Environment.Kind() == artifact.KindEvidence &&
		validCodeCommit(identity.CodeCommit) &&
		validLowerDigest(identity.ModuleDigest, 32) &&
		validLowerDigest(identity.WorkloadDigest, 32) &&
		identity.RealizedResidency.Valid() && validBenchmarkPromptDigests(identity.PromptDigests)
}

func validBenchmarkConfiguration(config BenchmarkConfiguration) bool {
	if config.Tokens <= 0 || config.Runs <= 0 || config.Runs > 1000 || config.Warmup < 0 || config.Warmup > 100 ||
		config.BatchSequences < 0 || config.BatchSequences > 1024 || !finiteNonnegative(config.Temperature) || config.TopK < 0 {
		return false
	}
	for _, adapter := range config.Adapters {
		if adapter.Kind() != artifact.KindAdapter {
			return false
		}
	}
	return true
}

func singleBenchmarkFactor(baseline, candidate BenchmarkConfiguration, factor string) bool {
	switch factor {
	case "cache_prompt":
		if baseline.CachePrompt || !candidate.CachePrompt || baseline.BatchSequences != 0 {
			return false
		}
		candidate.CachePrompt = baseline.CachePrompt
	case "speculative":
		if baseline.Speculative || !candidate.Speculative || baseline.BatchSequences != 0 {
			return false
		}
		candidate.Speculative = baseline.Speculative
	default:
		return false
	}
	return reflect.DeepEqual(baseline, candidate)
}

func validBenchmarkObservation(sample BenchmarkObservation, sequences int) bool {
	if len(sample.PromptDigest) != 64 || !validLowerDigest(sample.PromptDigest, 32) ||
		len(sample.TokenDigests) != sequences || sample.OutputTokens <= 0 || sample.PromptTokens <= 0 ||
		sample.CachedPromptTokens < 0 || sample.CachedPromptTokens > sample.PromptTokens ||
		sample.HostLogitTokens < 0 || sample.DeviceSelectedTokens < 0 || sample.DeviceTopKTokens < 0 ||
		sample.HostLogitTokens+sample.DeviceSelectedTokens+sample.DeviceTopKTokens != sample.OutputTokens ||
		!finitePositive(sample.TotalMilliseconds) || !finiteNonnegative(sample.TTFTMilliseconds) ||
		sample.TTFTMilliseconds > sample.TotalMilliseconds || !finiteNonnegative(sample.DecodeMilliseconds) ||
		!finiteNonnegative(sample.PromptMilliseconds) || !finiteNonnegative(sample.PromptTokensPerSecond) ||
		!finiteNonnegative(sample.EndToEndTokensPerSecond) || !finiteNonnegative(sample.DecodeTokensPerSecond) {
		return false
	}
	for _, digest := range sample.TokenDigests {
		if len(digest) != 64 || !validLowerDigest(digest, 32) {
			return false
		}
	}
	return true
}

func finitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
func finiteNonnegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
func validBenchmarkPromptDigests(digests []string) bool {
	if len(digests) == 0 || len(digests) > 1024 {
		return false
	}
	for _, digest := range digests {
		if len(digest) != 64 || !validLowerDigest(digest, 32) {
			return false
		}
	}
	return true
}

func validLowerDigest(value string, byteCount int) bool {
	if len(value) != 2*byteCount || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validCodeCommit(value string) bool {
	return validLowerDigest(value, 20) || validLowerDigest(value, 32)
}
