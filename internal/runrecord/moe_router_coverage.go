package runrecord

import (
	"cmp"
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// MoERouterObservationCoverageMediaType identifies a complete bounded set
	// of exact router observations.
	MoERouterObservationCoverageMediaType = "application/vnd.overgo.moe-router-observation-coverage+json"
	// MoERouterObservationCoverageSchema identifies the coverage contract version.
	MoERouterObservationCoverageSchema = "overgo/moe-router-observation-coverage/v1"
	// MoERouterObservationCoverageAlias names the single current complete set.
	// Compaction therefore retains at most one bounded router-observation set;
	// the append-only source remains an archive until an operator compacts it.
	MoERouterObservationCoverageAlias = "observation/moe-router/head"
)

// MoERouterObservationCoverage proves that every declared routed layer was
// observed exactly once at every step in a contiguous training interval.
// ObservationBytes is the exact sum of the retained child document sizes.
type MoERouterObservationCoverage struct {
	Version          uint16        `json:"version"`
	Run              artifact.ID   `json:"run"`
	FirstStep        uint64        `json:"first_step"`
	Steps            uint64        `json:"steps"`
	Layers           []uint32      `json:"layers"`
	Observations     []artifact.ID `json:"observations"`
	ObservationBytes uint64        `json:"observation_bytes"`
	ID               artifact.ID   `json:"-"`
}

var moeRouterObservationCoverageCodec = artifact.JSONDocumentCodec(
	"moe router observation coverage", artifact.KindEvidence,
	MoERouterObservationCoverageMediaType, MoERouterObservationCoverageSchema,
	canonicalizeMoERouterObservationCoverage,
	func(value MoERouterObservationCoverage) artifact.ID { return value.ID },
	func(value *MoERouterObservationCoverage, id artifact.ID) { value.ID = id },
	func(value MoERouterObservationCoverage) MoERouterObservationCoverage {
		value.Layers = slices.Clone(value.Layers)
		value.Observations = slices.Clone(value.Observations)
		return value
	},
)

// NewMoERouterObservationCoverage validates exact step-by-layer cardinality
// and refuses a retained set whose canonical bytes exceed the repository's
// immutable content bound. It never samples or truncates observations.
func NewMoERouterObservationCoverage(
	observations []MoERouterObservation,
	firstStep, steps uint64,
	layers []uint32,
) (MoERouterObservationCoverage, error) {
	if len(observations) == 0 {
		return MoERouterObservationCoverage{}, errors.New("run record: moe router observations absent")
	}
	ordered := slices.Clone(observations)
	slices.SortFunc(ordered, func(left, right MoERouterObservation) int {
		if order := cmp.Compare(left.Step, right.Step); order != 0 {
			return order
		}
		return cmp.Compare(left.Layer, right.Layer)
	})
	canonicalLayers := slices.Clone(layers)
	slices.Sort(canonicalLayers)
	if len(canonicalLayers) == 0 || len(slices.Compact(slices.Clone(canonicalLayers))) != len(canonicalLayers) {
		return MoERouterObservationCoverage{}, errors.New("run record: invalid moe router coverage layers")
	}
	expected, ok := checked.Mul64(steps, uint64(len(canonicalLayers)))
	if !ok || expected != uint64(len(ordered)) {
		return MoERouterObservationCoverage{}, errors.New("run record: moe router observation cardinality differs")
	}
	end, ok := checked.Add64(firstStep, steps)
	if !ok || end <= firstStep {
		return MoERouterObservationCoverage{}, errors.New("run record: invalid moe router coverage interval")
	}
	coverage := MoERouterObservationCoverage{
		Version: artifact.InitialDocumentVersion, Run: ordered[0].Run,
		FirstStep: firstStep, Steps: steps, Layers: canonicalLayers,
		Observations: make([]artifact.ID, len(ordered)),
	}
	scope := ordered[0]
	for index, observation := range ordered {
		if err := observation.ValidateIdentity(); err != nil {
			return MoERouterObservationCoverage{}, err
		}
		wantStep := firstStep + uint64(index/len(canonicalLayers))
		wantLayer := canonicalLayers[index%len(canonicalLayers)]
		if !sameMoERouterObservationCoverageScope(scope, observation) ||
			observation.Step != wantStep || observation.Layer != wantLayer {
			return MoERouterObservationCoverage{}, errors.New("run record: moe router observation coverage has a gap, duplicate, or scope change")
		}
		content, err := observation.Content()
		if err != nil {
			return MoERouterObservationCoverage{}, err
		}
		coverage.ObservationBytes, ok = checked.Add64(coverage.ObservationBytes, content.Descriptor.Size)
		if !ok {
			return MoERouterObservationCoverage{}, errors.New("run record: moe router retained byte count overflows")
		}
		coverage.Observations[index] = observation.ID
	}
	identified, err := moeRouterObservationCoverageCodec.New(coverage)
	if err != nil {
		return MoERouterObservationCoverage{}, err
	}
	content, err := identified.Content()
	if err != nil {
		return MoERouterObservationCoverage{}, err
	}
	if err := validateMoERouterRetentionBytes(identified.ObservationBytes, content.Descriptor.Size); err != nil {
		return MoERouterObservationCoverage{}, err
	}
	return identified, nil
}

func sameMoERouterObservationCoverageScope(left, right MoERouterObservation) bool {
	return left.Run == right.Run && left.Model == right.Model && left.Dataset == right.Dataset && left.Split == right.Split &&
		left.Recipe == right.Recipe && left.Code == right.Code && left.Checkpoint == right.Checkpoint && left.Policy == right.Policy &&
		left.Rows == right.Rows && left.Experts == right.Experts && left.TopK == right.TopK
}

// Content returns the canonical coverage document.
func (value MoERouterObservationCoverage) Content() (artifact.Content, error) {
	return moeRouterObservationCoverageCodec.Content(value)
}

// ValidateIdentity verifies the canonical coverage identity.
func (value MoERouterObservationCoverage) ValidateIdentity() error {
	return moeRouterObservationCoverageCodec.ValidateIdentity(value)
}

// Lineage roots every exact observation under the bounded current-set alias.
func (value MoERouterObservationCoverage) Lineage() []artifact.Lineage {
	lineage := make([]artifact.Lineage, 0, len(value.Observations)+1)
	lineage = append(lineage, artifact.Lineage{
		Child: value.ID, Parent: value.Run, Relation: artifact.RelationDependsOn,
	})
	for _, observation := range value.Observations {
		lineage = append(lineage, artifact.Lineage{
			Child: value.ID, Parent: observation, Relation: artifact.RelationContains,
		})
	}
	return lineage
}

// Batch publishes a complete set and atomically advances the single retained
// head. Previous sets remain in the append-only source but become reclaimable
// by normal OvergoDB compaction.
func (value MoERouterObservationCoverage) Batch(ctx context.Context, reader artifact.Reader) (artifact.Batch, error) {
	if ctx == nil || reader == nil {
		return artifact.Batch{}, errors.New("run record: moe router coverage dependencies are absent")
	}
	content, err := value.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	alias := artifact.AliasBinding{Name: MoERouterObservationCoverageAlias, Target: value.ID}
	previous, found, err := artifact.ResolveAlias(ctx, reader, MoERouterObservationCoverageAlias)
	if err != nil {
		return artifact.Batch{}, err
	}
	if found {
		alias.Previous = artifact.IDPointer(previous)
	}
	return artifact.NewDocumentBatch(
		"observation/moe-router/coverage/"+value.ID.String(),
		[]artifact.Content{content}, value.Lineage(), []artifact.AliasBinding{alias},
	)
}

func canonicalizeMoERouterObservationCoverage(value *MoERouterObservationCoverage) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun ||
		value.Steps == 0 || len(value.Layers) == 0 || len(value.Observations) == 0 || value.ObservationBytes == 0 ||
		!slices.IsSorted(value.Layers) || len(slices.Compact(slices.Clone(value.Layers))) != len(value.Layers) {
		return errors.New("run record: invalid moe router observation coverage")
	}
	expected, ok := checked.Mul64(value.Steps, uint64(len(value.Layers)))
	if !ok || expected != uint64(len(value.Observations)) {
		return errors.New("run record: invalid moe router observation cardinality")
	}
	if _, ok := checked.Add64(value.FirstStep, value.Steps); !ok {
		return errors.New("run record: moe router coverage interval overflows")
	}
	for _, observation := range value.Observations {
		if observation.Kind() != artifact.KindEvidence {
			return errors.New("run record: invalid moe router observation identity")
		}
	}
	return nil
}

func validateMoERouterRetentionBytes(observationBytes, coverageBytes uint64) error {
	total, ok := checked.Add64(observationBytes, coverageBytes)
	if !ok || total > artifact.MaxContentBytes {
		return errors.New("run record: moe router observations exceed bounded retention")
	}
	return nil
}
