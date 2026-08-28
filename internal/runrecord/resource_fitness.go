package runrecord

import (
	"cmp"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

// ResourceFitnessVersion identifies the common resource observation contract.
const ResourceFitnessVersion uint16 = 1

// ResourceMetric is a closed, integer-valued resource dimension. Units live in
// the metric name so consumers cannot compare seconds with nanoseconds or bytes
// with allocation counts.
type ResourceMetric string

const (
	// ResourceInputTokens counts uncached input tokens.
	ResourceInputTokens ResourceMetric = "input_tokens"
	// ResourceOutputTokens counts generated output tokens.
	ResourceOutputTokens ResourceMetric = "output_tokens"
	// ResourceCacheReadTokens counts input tokens supplied by a model cache.
	ResourceCacheReadTokens ResourceMetric = "cache_read_tokens"
	// ResourceCacheWriteTokens counts tokens newly written to a model cache.
	ResourceCacheWriteTokens ResourceMetric = "cache_write_tokens"
	// ResourceInputBytes counts admitted input payload bytes.
	ResourceInputBytes ResourceMetric = "input_bytes"
	// ResourceOutputBytes counts produced output payload bytes.
	ResourceOutputBytes ResourceMetric = "output_bytes"
	// ResourceCacheLookups counts cache lookup operations.
	ResourceCacheLookups ResourceMetric = "cache_lookups"
	// ResourceCacheHits counts successful cache lookups.
	ResourceCacheHits ResourceMetric = "cache_hits"
	// ResourceCacheWrites counts cache write operations.
	ResourceCacheWrites ResourceMetric = "cache_writes"
	// ResourceWallNS records elapsed wall time in nanoseconds.
	ResourceWallNS ResourceMetric = "wall_ns"
	// ResourceActiveComputeNS records measured active compute time in nanoseconds.
	// Concurrent work may make this greater than wall time.
	ResourceActiveComputeNS ResourceMetric = "active_compute_ns"
	// ResourceCPUNS records measured CPU time in nanoseconds.
	ResourceCPUNS ResourceMetric = "cpu_ns"
	// ResourceGPUNS records measured GPU active time in nanoseconds.
	ResourceGPUNS ResourceMetric = "gpu_ns"
	// ResourcePeakHostBytes records peak observed host memory bytes.
	ResourcePeakHostBytes ResourceMetric = "peak_host_bytes"
	// ResourcePeakDeviceBytes records peak observed device memory bytes.
	ResourcePeakDeviceBytes ResourceMetric = "peak_device_bytes"
	// ResourceDiskReadBytes counts bytes read from durable storage.
	ResourceDiskReadBytes ResourceMetric = "disk_read_bytes"
	// ResourceDiskWriteBytes counts bytes written to durable storage.
	ResourceDiskWriteBytes ResourceMetric = "disk_write_bytes"
	// ResourceNetworkReceiveBytes counts bytes received over a network.
	ResourceNetworkReceiveBytes ResourceMetric = "network_receive_bytes"
	// ResourceNetworkSendBytes counts bytes sent over a network.
	ResourceNetworkSendBytes ResourceMetric = "network_send_bytes"
	// ResourceHostToDeviceBytes counts host-to-accelerator transfer bytes.
	ResourceHostToDeviceBytes ResourceMetric = "host_to_device_bytes"
	// ResourceDeviceToHostBytes counts accelerator-to-host transfer bytes.
	ResourceDeviceToHostBytes ResourceMetric = "device_to_host_bytes"
	// ResourceCostUnits records exact integer units from the bound provider.
	ResourceCostUnits ResourceMetric = "cost_units"
)

// ResourceMeasure is one observed unsigned integer. Presence in the measure
// list means observed, including when Value is zero; absence means unknown.
type ResourceMeasure struct {
	Metric ResourceMetric `json:"metric"`
	Value  uint64         `json:"value"`
}

// ResourceScope keeps aggregation axes separate until an evidence-backed
// projection chooses which axes to group. Optional axes stay absent when they
// do not apply; fake identities are never synthesized to make a rectangular
// record.
type ResourceScope struct {
	Surface  InteractionSurface `json:"surface"`
	Model    artifact.ID        `json:"model,omitzero"`
	Hardware artifact.ID        `json:"hardware,omitzero"`
	Provider artifact.ID        `json:"provider,omitzero"`
	Workload artifact.ID        `json:"workload"`
	Attempt  artifact.ID        `json:"attempt"`
}

// ResourceFitness is the common resource and interaction observation value.
// High-volume samples remain outside this value; they are summarized into its
// exact integer dimensions by their owning producer.
type ResourceFitness struct {
	Version  uint16            `json:"version"`
	Scope    ResourceScope     `json:"scope"`
	Measures []ResourceMeasure `json:"measures,omitempty"`
	// Interactions reuses the exact InteractionWork owner. A non-nil value
	// declares every InteractionWork counter observed; its zero fields are
	// known zeros. Nil means the interaction counters are unknown.
	Interactions *InteractionWork `json:"interactions,omitempty"`
}

// NewResourceFitness validates and canonicalizes one resource observation.
func NewResourceFitness(value ResourceFitness) (ResourceFitness, error) {
	if value.Version != 0 && value.Version != ResourceFitnessVersion {
		return ResourceFitness{}, errors.New("run record: unsupported resource fitness version")
	}
	value.Version = ResourceFitnessVersion
	value = cloneResourceFitness(value)
	if err := canonicalizeResourceFitness(&value); err != nil {
		return ResourceFitness{}, err
	}
	return value, nil
}

// Validate verifies that value is already in canonical form.
func (value ResourceFitness) Validate() error {
	canonical := cloneResourceFitness(value)
	if err := canonicalizeResourceFitness(&canonical); err != nil {
		return err
	}
	if !slices.Equal(value.Measures, canonical.Measures) {
		return errors.New("run record: resource measures are not canonical")
	}
	return nil
}

// Measure returns an observed metric and its presence. Callers must retain the
// boolean: a missing measurement is not an observed zero.
func (value ResourceFitness) Measure(metric ResourceMetric) (uint64, bool) {
	return resourceMeasure(value.Measures, metric)
}

// Authorities returns the exact identities a containing evidence record must
// bind through lineage. Axis order is stable and semantic; absent optional axes
// are omitted.
func (value ResourceFitness) Authorities() []artifact.ID {
	authorities := [...]artifact.ID{
		value.Scope.Model, value.Scope.Hardware, value.Scope.Provider,
		value.Scope.Workload, value.Scope.Attempt,
	}
	result := make([]artifact.ID, 0, len(authorities))
	for _, id := range authorities {
		if id.Valid() {
			result = append(result, id)
		}
	}
	return result
}

func canonicalizeResourceFitness(value *ResourceFitness) error {
	if value == nil || value.Version != ResourceFitnessVersion || !slices.Contains(interactionSurfaces, value.Scope.Surface) {
		return errors.New("run record: invalid resource fitness envelope")
	}
	if value.Scope.Model.Valid() && value.Scope.Model.Kind() != artifact.KindModel ||
		value.Scope.Hardware.Valid() && value.Scope.Hardware.Kind() != artifact.KindEvidence ||
		value.Scope.Provider.Valid() && value.Scope.Provider.Kind() != artifact.KindProfile ||
		!validResourceWorkload(value.Scope.Workload) ||
		value.Scope.Attempt.Kind() != artifact.KindRun && value.Scope.Attempt.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid resource fitness scope")
	}
	authorities := value.Authorities()
	for index, id := range authorities {
		if slices.Contains(authorities[:index], id) {
			return errors.New("run record: resource fitness identity axes collapse")
		}
	}
	if len(value.Measures) == 0 {
		value.Measures = nil
	}
	for _, measure := range value.Measures {
		if !validResourceMetric(measure.Metric) {
			return errors.New("run record: foreign resource metric")
		}
	}
	sort.Slice(value.Measures, func(i, j int) bool { return value.Measures[i].Metric < value.Measures[j].Metric })
	if len(slices.CompactFunc(slices.Clone(value.Measures), func(left, right ResourceMeasure) bool {
		return left.Metric == right.Metric
	})) != len(value.Measures) {
		return errors.New("run record: duplicate resource metric")
	}
	if len(value.Measures) == 0 && value.Interactions == nil {
		return errors.New("run record: resource fitness records no observations")
	}
	if lookups, lookupsKnown := resourceMeasure(value.Measures, ResourceCacheLookups); lookupsKnown {
		if hits, hitsKnown := resourceMeasure(value.Measures, ResourceCacheHits); hitsKnown && hits > lookups {
			return errors.New("run record: cache hits exceed lookups")
		}
	}
	return nil
}

func validResourceWorkload(id artifact.ID) bool {
	switch id.Kind() {
	case artifact.KindRecipe, artifact.KindProfile, artifact.KindDataset, artifact.KindDatasetShard, artifact.KindEvidence:
		return true
	default:
		return false
	}
}

func validResourceMetric(metric ResourceMetric) bool {
	switch metric {
	case ResourceInputTokens, ResourceOutputTokens, ResourceCacheReadTokens, ResourceCacheWriteTokens,
		ResourceInputBytes, ResourceOutputBytes, ResourceCacheLookups, ResourceCacheHits, ResourceCacheWrites,
		ResourceWallNS, ResourceActiveComputeNS, ResourceCPUNS, ResourceGPUNS,
		ResourcePeakHostBytes, ResourcePeakDeviceBytes, ResourceDiskReadBytes, ResourceDiskWriteBytes,
		ResourceNetworkReceiveBytes, ResourceNetworkSendBytes, ResourceHostToDeviceBytes,
		ResourceDeviceToHostBytes, ResourceCostUnits:
		return true
	default:
		return false
	}
}

func resourceMeasure(measures []ResourceMeasure, metric ResourceMetric) (uint64, bool) {
	index, found := slices.BinarySearchFunc(measures, metric, func(measure ResourceMeasure, target ResourceMetric) int {
		return cmp.Compare(measure.Metric, target)
	})
	if !found {
		var unknown uint64
		return unknown, false
	}
	return measures[index].Value, true
}

func cloneResourceFitness(value ResourceFitness) ResourceFitness {
	value.Measures = slices.Clone(value.Measures)
	if value.Interactions != nil {
		work := *value.Interactions
		value.Interactions = &work
	}
	return value
}
