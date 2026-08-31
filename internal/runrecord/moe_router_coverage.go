package runrecord

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/strictjson"
)

const (
	// MoERouterObservationChunkMediaType identifies exact raw router chunks.
	MoERouterObservationChunkMediaType = "application/vnd.overgo.moe-router-observation-chunk+json"
	// MoERouterObservationCoverageMediaType identifies compact router coverage.
	MoERouterObservationCoverageMediaType = "application/vnd.overgo.moe-router-observation-coverage+json"
	// MoERouterObservationCoverageSchema identifies the compact v2 contract.
	MoERouterObservationCoverageSchema = "overgo/moe-router-observation-coverage/v2"
	// MoERouterObservationCoverageAlias names the retained router evidence head.
	MoERouterObservationCoverageAlias = "observation/moe-router/head"
)

// MoERouterObservationChunk is one bounded immutable raw payload. Full router
// facts stay out of the metadata journal and decode only on an explicit read.
type MoERouterObservationChunk struct {
	Version      uint16                 `json:"version"`
	Previous     artifact.ID            `json:"previous,omitzero"`
	Observations []MoERouterObservation `json:"observations"`
	ID           artifact.ID            `json:"-"`
}

// MoERouterObservationCoverage is the compact typed journal fact for a chunk.
type MoERouterObservationCoverage struct {
	Version          uint16      `json:"version"`
	Run              artifact.ID `json:"run"`
	FirstStep        uint64      `json:"first_step"`
	Steps            uint64      `json:"steps"`
	Layers           []uint32    `json:"layers"`
	Chunk            artifact.ID `json:"chunk"`
	Previous         artifact.ID `json:"previous,omitzero"`
	ObservationCount uint64      `json:"observation_count"`
	ObservationBytes uint64      `json:"observation_bytes"`
	ID               artifact.ID `json:"-"`
}

var moeRouterObservationCoverageCodec = artifact.JSONDocumentCodec(
	"moe router observation coverage", artifact.KindEvidence,
	MoERouterObservationCoverageMediaType, MoERouterObservationCoverageSchema,
	canonicalizeMoERouterObservationCoverage,
	func(value MoERouterObservationCoverage) artifact.ID { return value.ID },
	func(value *MoERouterObservationCoverage, id artifact.ID) { value.ID = id },
	func(value MoERouterObservationCoverage) MoERouterObservationCoverage {
		value.Layers = slices.Clone(value.Layers)
		return value
	},
)

// NewMoERouterObservationChunk identifies one canonical bounded raw chunk.
func NewMoERouterObservationChunk(observations []MoERouterObservation, previous artifact.ID) (MoERouterObservationChunk, error) {
	canonical, data, err := canonicalMoERouterObservationChunk(MoERouterObservationChunk{
		Version: artifact.InitialDocumentVersion, Previous: previous,
		Observations: cloneMoERouterObservations(observations),
	})
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	canonical.ID = id
	return canonical, nil
}

// Content returns the exact raw content blob.
func (value MoERouterObservationChunk) Content() (artifact.Content, error) {
	canonical, data, err := canonicalMoERouterObservationChunk(value)
	if err != nil {
		return artifact.Content{}, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return artifact.Content{}, err
	}
	canonical.ID = id
	if value.ID != id || !reflect.DeepEqual(value, canonical) {
		return artifact.Content{}, errors.New("run record: moe router observation chunk is not canonical")
	}
	content := artifact.Content{Descriptor: artifact.Descriptor{
		ID: id, Size: uint64(len(data)), MediaType: MoERouterObservationChunkMediaType,
	}, Data: data}
	return content, content.Validate()
}

// ParseMoERouterObservationChunk accepts only canonical bounded raw bytes.
func ParseMoERouterObservationChunk(data []byte) (MoERouterObservationChunk, error) {
	if len(data) == 0 || len(data) > artifact.MaxContentBytes {
		return MoERouterObservationChunk{}, errors.New("run record: invalid moe router observation chunk size")
	}
	var decoded MoERouterObservationChunk
	if err := strictjson.DecodeBytes(data, &decoded); err != nil {
		return MoERouterObservationChunk{}, err
	}
	canonical, encoded, err := canonicalMoERouterObservationChunk(decoded)
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	if !bytes.Equal(data, encoded) {
		return MoERouterObservationChunk{}, errors.New("run record: non-canonical moe router observation chunk")
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, encoded)
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	canonical.ID = id
	return canonical, nil
}

// RequireMoERouterObservationChunk loads and verifies one raw chunk.
func RequireMoERouterObservationChunk(ctx context.Context, reader artifact.Reader, id artifact.ID) (MoERouterObservationChunk, error) {
	if ctx == nil || reader == nil || id.Kind() != artifact.KindFile {
		return MoERouterObservationChunk{}, errors.New("run record: invalid moe router observation chunk read")
	}
	descriptor, stream, found, err := reader.OpenContent(ctx, id)
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	if !found {
		return MoERouterObservationChunk{}, errors.New("run record: moe router observation chunk is absent")
	}
	content, err := artifact.ReadContentFrom(descriptor, stream)
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	if descriptor.MediaType != MoERouterObservationChunkMediaType || descriptor.Schema != "" {
		return MoERouterObservationChunk{}, errors.New("run record: incompatible moe router observation chunk")
	}
	value, err := ParseMoERouterObservationChunk(content.Data)
	if err != nil {
		return MoERouterObservationChunk{}, err
	}
	if value.ID != id {
		return MoERouterObservationChunk{}, errors.New("run record: moe router observation chunk identity differs")
	}
	return value, nil
}

// NewMoERouterObservationCoverage derives compact exact coverage from a chunk.
func NewMoERouterObservationCoverage(chunk MoERouterObservationChunk, firstStep, steps uint64, layers []uint32) (MoERouterObservationCoverage, error) {
	content, err := chunk.Content()
	if err != nil {
		return MoERouterObservationCoverage{}, err
	}
	canonicalLayers := slices.Clone(layers)
	slices.Sort(canonicalLayers)
	observations := chunk.Observations
	if len(canonicalLayers) == 0 || len(slices.Compact(slices.Clone(canonicalLayers))) != len(canonicalLayers) {
		return MoERouterObservationCoverage{}, errors.New("run record: invalid moe router coverage layers")
	}
	expected, ok := checked.Mul64(steps, uint64(len(canonicalLayers)))
	if !ok || expected == 0 || expected != uint64(len(observations)) {
		return MoERouterObservationCoverage{}, errors.New("run record: moe router observation cardinality differs")
	}
	end, ok := checked.Add64(firstStep, steps)
	if !ok || end <= firstStep {
		return MoERouterObservationCoverage{}, errors.New("run record: invalid moe router coverage interval")
	}
	scope := observations[0]
	for index, observation := range observations {
		wantStep := firstStep + uint64(index/len(canonicalLayers))
		wantLayer := canonicalLayers[index%len(canonicalLayers)]
		if !sameMoERouterObservationCoverageScope(scope, observation) || observation.Step != wantStep || observation.Layer != wantLayer {
			return MoERouterObservationCoverage{}, errors.New("run record: moe router observation coverage has a gap, duplicate, or scope change")
		}
	}
	return moeRouterObservationCoverageCodec.New(MoERouterObservationCoverage{
		Version: artifact.InitialDocumentVersion, Run: scope.Run,
		FirstStep: firstStep, Steps: steps, Layers: canonicalLayers,
		Chunk: chunk.ID, Previous: chunk.Previous,
		ObservationCount: expected, ObservationBytes: content.Descriptor.Size,
	})
}

func sameMoERouterObservationCoverageScope(left, right MoERouterObservation) bool {
	return left.Run == right.Run && left.Model == right.Model && left.Dataset == right.Dataset && left.Split == right.Split &&
		left.Recipe == right.Recipe && left.Code == right.Code && left.Checkpoint == right.Checkpoint && left.Policy == right.Policy &&
		left.Rows == right.Rows && left.Experts == right.Experts && left.TopK == right.TopK
}

// Content returns the canonical compact coverage document.
func (value MoERouterObservationCoverage) Content() (artifact.Content, error) {
	return moeRouterObservationCoverageCodec.Content(value)
}

// ValidateIdentity verifies the compact coverage identity.
func (value MoERouterObservationCoverage) ValidateIdentity() error {
	return moeRouterObservationCoverageCodec.ValidateIdentity(value)
}

// Lineage binds one summary to its run and raw chunk only.
func (value MoERouterObservationCoverage) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: value.ID, Parent: value.Run, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Chunk, Relation: artifact.RelationContains},
	}
}

// Batch atomically binds raw bytes, compact summary, lineage, and retained head.
func (value MoERouterObservationCoverage) Batch(ctx context.Context, reader artifact.Reader, chunk MoERouterObservationChunk) (artifact.Batch, error) {
	want, err := NewMoERouterObservationCoverage(chunk, value.FirstStep, value.Steps, value.Layers)
	if err != nil {
		return artifact.Batch{}, err
	}
	if !reflect.DeepEqual(value, want) {
		return artifact.Batch{}, errors.New("run record: moe router coverage differs from raw chunk")
	}
	raw, err := chunk.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	summary, err := value.Content()
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
	return artifact.NewDocumentBatch("observation/moe-router/coverage/"+value.ID.String(),
		[]artifact.Content{raw, summary}, value.Lineage(), []artifact.AliasBinding{alias})
}

// RequireMoERouterObservationCoverage loads a summary and verifies its raw chunk.
func RequireMoERouterObservationCoverage(ctx context.Context, reader artifact.Reader, id artifact.ID) (MoERouterObservationCoverage, MoERouterObservationChunk, error) {
	value, err := moeRouterObservationCoverageCodec.Require(ctx, reader, id)
	if err != nil {
		return MoERouterObservationCoverage{}, MoERouterObservationChunk{}, err
	}
	chunk, err := RequireMoERouterObservationChunk(ctx, reader, value.Chunk)
	if err != nil {
		return MoERouterObservationCoverage{}, MoERouterObservationChunk{}, err
	}
	want, err := NewMoERouterObservationCoverage(chunk, value.FirstStep, value.Steps, value.Layers)
	if err != nil {
		return MoERouterObservationCoverage{}, MoERouterObservationChunk{}, err
	}
	if !reflect.DeepEqual(value, want) {
		return MoERouterObservationCoverage{}, MoERouterObservationChunk{}, errors.New("run record: moe router coverage summary differs")
	}
	return value, chunk, nil
}

func canonicalizeMoERouterObservationCoverage(value *MoERouterObservationCoverage) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun ||
		value.Chunk.Kind() != artifact.KindFile || value.Steps == 0 || len(value.Layers) == 0 ||
		value.ObservationCount == 0 || value.ObservationBytes == 0 || value.ObservationBytes > artifact.MaxContentBytes ||
		!slices.IsSorted(value.Layers) || len(slices.Compact(slices.Clone(value.Layers))) != len(value.Layers) {
		return errors.New("run record: invalid moe router observation coverage")
	}
	expected, ok := checked.Mul64(value.Steps, uint64(len(value.Layers)))
	if !ok || expected != value.ObservationCount {
		return errors.New("run record: invalid moe router observation cardinality")
	}
	if _, ok := checked.Add64(value.FirstStep, value.Steps); !ok {
		return errors.New("run record: moe router coverage interval overflows")
	}
	return nil
}

func canonicalMoERouterObservationChunk(value MoERouterObservationChunk) (MoERouterObservationChunk, []byte, error) {
	if value.Version != artifact.InitialDocumentVersion || len(value.Observations) == 0 ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return MoERouterObservationChunk{}, nil, errors.New("run record: invalid moe router observation chunk")
	}
	value.ID = artifact.ID{}
	value.Observations = cloneMoERouterObservations(value.Observations)
	for index := range value.Observations {
		priorID := value.Observations[index].ID
		identified, err := NewMoERouterObservation(value.Observations[index])
		if err != nil || priorID.Valid() && priorID != identified.ID {
			return MoERouterObservationChunk{}, nil, errors.Join(err, errors.New("run record: invalid moe router observation identity"))
		}
		value.Observations[index] = identified
	}
	slices.SortFunc(value.Observations, func(left, right MoERouterObservation) int {
		if order := cmp.Compare(left.Step, right.Step); order != 0 {
			return order
		}
		return cmp.Compare(left.Layer, right.Layer)
	})
	data, err := json.Marshal(value)
	if err != nil {
		return MoERouterObservationChunk{}, nil, err
	}
	if len(data) > artifact.MaxContentBytes {
		return MoERouterObservationChunk{}, nil, errors.New("run record: moe router observation chunk exceeds bound")
	}
	return value, data, nil
}

func cloneMoERouterObservations(values []MoERouterObservation) []MoERouterObservation {
	cloned := slices.Clone(values)
	for index := range cloned {
		cloned[index].Selections = slices.Clone(cloned[index].Selections)
		cloned[index].CombineWeights = slices.Clone(cloned[index].CombineWeights)
		cloned[index].Accepted = slices.Clone(cloned[index].Accepted)
		cloned[index].Margins = slices.Clone(cloned[index].Margins)
	}
	return cloned
}
