package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	// RolloutProjectionMediaType identifies rollout projection reports.
	RolloutProjectionMediaType = "application/vnd.overgo.rollout-projection+json"
	// RolloutProjectionSchema identifies the projection report contract.
	RolloutProjectionSchema = "overgo/rollout-projection/v1"
	// RolloutProjectorVersion is the registered projector implementation
	// version; the digest binds it, so a changed projector produces new
	// evidence instead of silently overwriting old readings.
	RolloutProjectorVersion = 1
	// RolloutProjectionQuery names the one registered query contract: the
	// lineage children of the rollout plan, excluding prior projection
	// reports, at an exact store head.
	RolloutProjectionQuery = "overgo/rollout-projection-query/children-of-plan/v1"
)

// RolloutProjection is the immutable report promotion and rollback consume
// — never a dashboard value or mutable projection row. It binds the exact
// store head it was read at, the registered projector version, the query
// contract, every source document consumed, the coverage count, and the
// result digest, so rebuilding the same projector at the same head
// reproduces the digest bit for bit and any drift is a different report.
type RolloutProjection struct {
	Version   uint16        `json:"version"`
	Plan      artifact.ID   `json:"plan"`
	Head      string        `json:"head"`
	Sequence  uint64        `json:"sequence"`
	Projector uint16        `json:"projector"`
	Query     string        `json:"query"`
	Sources   []artifact.ID `json:"sources,omitempty"`
	Coverage  uint64        `json:"coverage"`
	Digest    string        `json:"digest"`
	ID        artifact.ID   `json:"-"`
}

var rolloutProjectionCodec = artifact.JSONDocumentCodec(
	"rollout projection", artifact.KindEvidence, RolloutProjectionMediaType, RolloutProjectionSchema,
	canonicalizeRolloutProjection,
	func(value RolloutProjection) artifact.ID { return value.ID },
	func(value *RolloutProjection, id artifact.ID) { value.ID = id },
	func(value RolloutProjection) RolloutProjection {
		value.Sources = slices.Clone(value.Sources)
		return value
	},
)

func canonicalizeRolloutProjection(value *RolloutProjection) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("evaluation: invalid rollout projection version")
	}
	if value.Plan.Kind() != artifact.KindProfile || value.Head == "" {
		return errors.New("evaluation: rollout projection requires its plan identity and exact head")
	}
	if value.Projector != RolloutProjectorVersion {
		return fmt.Errorf("evaluation: unregistered rollout projector version %d", value.Projector)
	}
	if value.Query != RolloutProjectionQuery {
		return fmt.Errorf("evaluation: unregistered rollout projection query %q", value.Query)
	}
	value.Sources = uniqueArtifactIDs(value.Sources)
	if value.Coverage != uint64(len(value.Sources)) {
		return errors.New("evaluation: rollout projection coverage must equal its consumed sources")
	}
	if value.Digest != rolloutProjectionDigest(*value) {
		return errors.New("evaluation: rollout projection digest does not reproduce from its bindings")
	}
	return nil
}

// rolloutProjectionDigest derives the result digest from every binding
// that determines the reading: query contract, projector version, plan,
// head, and the canonical source list.
func rolloutProjectionDigest(value RolloutProjection) string {
	inputs := []string{
		value.Query, fmt.Sprintf("projector=%d", value.Projector),
		value.Plan.String(), value.Head, fmt.Sprintf("sequence=%d", value.Sequence),
	}
	for _, source := range value.Sources {
		inputs = append(inputs, source.String())
	}
	digest := sha256.Sum256([]byte(strings.Join(inputs, "\n")))
	return hex.EncodeToString(digest[:])
}

// ProjectRolloutEvidence materializes one rollout reading at the store's
// current head without committing anything: the registered query
// enumerates the plan's lineage children — the observations and decisions
// citing it — excluding prior projection reports, and the digest derives
// from the complete binding set. Projection is pure, so projecting twice
// at one head reproduces one identity.
func ProjectRolloutEvidence(
	ctx context.Context,
	store artifact.Repository,
	planID artifact.ID,
) (RolloutProjection, error) {
	content, found, err := artifact.ReadContent(ctx, store, planID)
	if err != nil {
		return RolloutProjection{}, err
	}
	if !found {
		return RolloutProjection{}, fmt.Errorf("evaluation: rollout plan %s is not committed", planID)
	}
	if _, err := runrecord.ParseRolloutPlan(content.Data); err != nil {
		return RolloutProjection{}, err
	}
	children, err := store.Children(ctx, planID)
	if err != nil {
		return RolloutProjection{}, err
	}
	sources := make([]artifact.ID, 0, len(children))
	for _, edge := range children {
		descriptor, present, err := store.Artifact(ctx, edge.Child)
		if err != nil {
			return RolloutProjection{}, err
		}
		if !present {
			return RolloutProjection{}, fmt.Errorf("evaluation: rollout source %s is not committed", edge.Child)
		}
		if descriptor.MediaType == RolloutProjectionMediaType {
			continue
		}
		sources = append(sources, edge.Child)
	}
	sources = uniqueArtifactIDs(sources)
	head, sequence := store.Head()
	projection := RolloutProjection{
		Version: artifact.InitialDocumentVersion, Plan: planID,
		Head: head.String(), Sequence: sequence,
		Projector: RolloutProjectorVersion, Query: RolloutProjectionQuery,
		Sources: sources, Coverage: uint64(len(sources)),
	}
	projection.Digest = rolloutProjectionDigest(projection)
	return rolloutProjectionCodec.New(projection)
}

// PublishRolloutProjection commits one materialized reading as immutable
// evidence citing its plan and every consumed source.
func PublishRolloutProjection(
	ctx context.Context,
	store artifact.Repository,
	projection RolloutProjection,
) (artifact.CommitID, error) {
	lineage := artifact.UniqueDependencyLineage(
		projection.ID, append([]artifact.ID{projection.Plan}, projection.Sources...)...,
	)
	batch, err := rolloutProjectionCodec.Batch(
		"rollout/projection/"+projection.ID.String(), projection, lineage, nil,
	)
	if err != nil {
		return artifact.CommitID{}, err
	}
	return artifact.CommitBatch(ctx, store, batch)
}

// ParseRolloutProjection decodes one canonical projection report and
// proves its content identity — including that the digest reproduces from
// the report's own bindings.
func ParseRolloutProjection(content []byte) (RolloutProjection, error) {
	return rolloutProjectionCodec.Parse(content)
}
