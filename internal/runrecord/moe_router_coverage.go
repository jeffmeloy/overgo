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
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

const (
	// MoERouterObservationChunkMediaType identifies exact raw router chunks.
	MoERouterObservationChunkMediaType = "application/vnd.overgo.moe-router-observation-chunk+json"
	// MoERouterObservationCoverageMediaType identifies compact router coverage.
	MoERouterObservationCoverageMediaType = "application/vnd.overgo.moe-router-observation-coverage+json"
	// MoERouterObservationCoverageSchema identifies the indexed compact contract.
	MoERouterObservationCoverageSchema = "overgo/moe-router-observation-coverage/v3"
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
	Model            artifact.ID `json:"model"`
	Dataset          artifact.ID `json:"dataset"`
	Split            artifact.ID `json:"split"`
	Recipe           artifact.ID `json:"recipe"`
	FirstStep        uint64      `json:"first_step"`
	Steps            uint64      `json:"steps"`
	Layers           []uint32    `json:"layers"`
	Chunk            artifact.ID `json:"chunk"`
	Previous         artifact.ID `json:"previous,omitzero"`
	ObservationCount uint64      `json:"observation_count"`
	ObservationBytes uint64      `json:"observation_bytes"`
	ID               artifact.ID `json:"-"`
}

// MoERouterObservationQuery selects compact coverage by exact indexed
// authorities. Head selects only the retained alias. Raw chunks are decoded
// only when IncludeSamples is explicit.
type MoERouterObservationQuery struct {
	Run            artifact.ID
	Model          artifact.ID
	Dataset        artifact.ID
	Split          artifact.ID
	Recipe         artifact.ID
	Head           bool
	IncludeSamples bool
	Limit          int
}

// MoERouterObservationQueryWork reports exact query and decode work.
type MoERouterObservationQueryWork struct {
	SummariesInspected uint64 `json:"summaries_inspected"`
	SummariesMatched   uint64 `json:"summaries_matched"`
	SummariesReturned  uint64 `json:"summaries_returned"`
	SummaryBlobsLoaded uint64 `json:"summary_blobs_loaded"`
	SummaryBytesLoaded uint64 `json:"summary_bytes_loaded"`
	RawChunksLoaded    uint64 `json:"raw_chunks_loaded"`
	RawBytesLoaded     uint64 `json:"raw_bytes_loaded"`
}

// MoERouterObservationQueryResult is one auditable indexed evidence read.
type MoERouterObservationQueryResult struct {
	Summaries []MoERouterObservationCoverage `json:"summaries"`
	Chunks    []MoERouterObservationChunk    `json:"chunks,omitempty"`
	Plan      overgodb.QueryPlanReport       `json:"plan"`
	Work      MoERouterObservationQueryWork  `json:"work"`
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
		Version: artifact.InitialDocumentVersion, Run: scope.Run, Model: scope.Model,
		Dataset: scope.Dataset, Split: scope.Split, Recipe: scope.Recipe,
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

// Lineage binds one summary to its indexed scope authorities and raw chunk.
func (value MoERouterObservationCoverage) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: value.ID, Parent: value.Run, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Model, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Dataset, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Split, Relation: artifact.RelationDependsOn},
		{Child: value.ID, Parent: value.Recipe, Relation: artifact.RelationDependsOn},
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

// RequireMoERouterObservationCoverageSummary loads one exact compact summary
// without opening its raw chunk.
func RequireMoERouterObservationCoverageSummary(ctx context.Context, reader artifact.Reader, id artifact.ID) (MoERouterObservationCoverage, error) {
	value, err := moeRouterObservationCoverageCodec.Require(ctx, reader, id)
	if err != nil {
		return MoERouterObservationCoverage{}, err
	}
	stored, err := reader.Parents(ctx, id)
	if err != nil {
		return MoERouterObservationCoverage{}, err
	}
	expected := value.Lineage()
	if len(stored) != len(expected) {
		return MoERouterObservationCoverage{}, errors.New("run record: moe router coverage stored lineage differs")
	}
	for _, edge := range expected {
		if !slices.Contains(stored, edge) {
			return MoERouterObservationCoverage{}, errors.New("run record: moe router coverage stored lineage differs")
		}
		if _, found, parentErr := reader.Artifact(ctx, edge.Parent); parentErr != nil || !found {
			return MoERouterObservationCoverage{}, errors.Join(errors.New("run record: moe router coverage lineage parent is absent"), parentErr)
		}
	}
	return value, nil
}

// RequireMoERouterObservationCoverage loads a summary and verifies its raw chunk.
func RequireMoERouterObservationCoverage(ctx context.Context, reader artifact.Reader, id artifact.ID) (MoERouterObservationCoverage, MoERouterObservationChunk, error) {
	value, err := RequireMoERouterObservationCoverageSummary(ctx, reader, id)
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

// QueryMoERouterObservationCoverage selects summaries through the retained
// alias or one exact lineage index, then applies remaining scope filters to
// compact documents. Raw payloads remain unopened unless requested.
func QueryMoERouterObservationCoverage(
	ctx context.Context,
	store *overgodb.Store,
	query MoERouterObservationQuery,
) (MoERouterObservationQueryResult, error) {
	seed, err := validateMoERouterObservationQuery(query)
	if ctx == nil || store == nil || err != nil {
		return MoERouterObservationQueryResult{}, errors.Join(errors.New("run record: invalid moe router observation query"), err)
	}
	request := overgodb.Query{
		MediaType: MoERouterObservationCoverageMediaType, Schema: MoERouterObservationCoverageSchema,
		MaxResults: MaximumAttemptPopulation,
		Projection: overgodb.ProjectContentPresence, RequireIndex: true,
	}
	if query.Head {
		request.Alias = MoERouterObservationCoverageAlias
	} else {
		request.Artifact = &seed
		request.Follow = overgodb.FollowChildren
		request.MaxDepth = uint32(len([...]overgodb.FollowDirection{overgodb.FollowChildren}))
		request.Relation = artifact.RelationDependsOn
	}
	selected, err := store.Query(ctx, request)
	if err != nil {
		return MoERouterObservationQueryResult{}, err
	}
	result := MoERouterObservationQueryResult{Plan: selected.Plan}
	for _, content := range selected.Contents {
		result.Work.SummariesInspected++
		descriptor, found, descriptorErr := store.Artifact(ctx, content.Artifact)
		if descriptorErr != nil || !found {
			return MoERouterObservationQueryResult{}, errors.Join(errors.New("run record: moe router coverage descriptor is absent"), descriptorErr)
		}
		summary, summaryErr := RequireMoERouterObservationCoverageSummary(ctx, store, content.Artifact)
		if summaryErr != nil {
			return MoERouterObservationQueryResult{}, summaryErr
		}
		result.Work.SummaryBlobsLoaded++
		result.Work.SummaryBytesLoaded += descriptor.Size
		if !matchesMoERouterObservationQuery(summary, query) {
			continue
		}
		result.Work.SummariesMatched++
		if len(result.Summaries) == query.Limit {
			continue
		}
		result.Summaries = append(result.Summaries, summary)
		result.Work.SummariesReturned++
		if !query.IncludeSamples {
			continue
		}
		chunk, chunkErr := RequireMoERouterObservationChunk(ctx, store, summary.Chunk)
		if chunkErr != nil {
			return MoERouterObservationQueryResult{}, chunkErr
		}
		want, deriveErr := NewMoERouterObservationCoverage(chunk, summary.FirstStep, summary.Steps, summary.Layers)
		if deriveErr != nil || !reflect.DeepEqual(summary, want) {
			return MoERouterObservationQueryResult{}, errors.Join(errors.New("run record: moe router coverage summary differs"), deriveErr)
		}
		result.Chunks = append(result.Chunks, chunk)
		result.Work.RawChunksLoaded++
		result.Work.RawBytesLoaded += summary.ObservationBytes
	}
	return result, nil
}

func validateMoERouterObservationQuery(query MoERouterObservationQuery) (artifact.ID, error) {
	if query.Limit <= 0 || query.Limit >= MaximumAttemptPopulation {
		return artifact.ID{}, errors.New("run record: invalid moe router observation query bound")
	}
	filters := []struct {
		id   artifact.ID
		kind artifact.Kind
	}{{query.Run, artifact.KindRun}, {query.Model, artifact.KindModel}, {query.Dataset, artifact.KindDataset},
		{query.Split, artifact.KindDatasetShard}, {query.Recipe, artifact.KindRecipe}}
	var seed artifact.ID
	for _, filter := range filters {
		if !filter.id.Valid() {
			continue
		}
		if filter.id.Kind() != filter.kind {
			return artifact.ID{}, errors.New("run record: invalid moe router observation query authority")
		}
		if !seed.Valid() {
			seed = filter.id
		}
	}
	if !query.Head && !seed.Valid() {
		return artifact.ID{}, errors.New("run record: query requires a head or indexed scope seed")
	}
	return seed, nil
}

func matchesMoERouterObservationQuery(value MoERouterObservationCoverage, query MoERouterObservationQuery) bool {
	return (!query.Run.Valid() || value.Run == query.Run) && (!query.Model.Valid() || value.Model == query.Model) &&
		(!query.Dataset.Valid() || value.Dataset == query.Dataset) && (!query.Split.Valid() || value.Split == query.Split) &&
		(!query.Recipe.Valid() || value.Recipe == query.Recipe)
}

func canonicalizeMoERouterObservationCoverage(value *MoERouterObservationCoverage) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun ||
		value.Model.Kind() != artifact.KindModel || value.Dataset.Kind() != artifact.KindDataset ||
		value.Split.Kind() != artifact.KindDatasetShard || value.Recipe.Kind() != artifact.KindRecipe ||
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
