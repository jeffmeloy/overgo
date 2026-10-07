package tensorstats

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"

	"overgo/internal/artifact"
)

// ResamplingMethod names the exact operation on declared whole units.
type ResamplingMethod string

// BlockBootstrap draws the original number of units with replacement within
// each declared stratum. Unequal member counts never change unit probabilities.
const BlockBootstrap ResamplingMethod = "stratified-block-bootstrap"

// SamplingUnit is one declared independent source unit and its ordered members.
// Member order preserves temporal/paired structure. Distinct IDs do not prove
// independence; the plan's assumption and evidence must justify the grouping.
type SamplingUnit struct {
	ID      artifact.ID   `json:"id"`
	Stratum artifact.ID   `json:"stratum"`
	Members []artifact.ID `json:"members"`
}

// ResamplingPlan binds population, unit partition, statistical assumptions,
// evidence, method, explicit PCG seeds, and a finite streamed draw count.
// Version one uses Go math/rand/v2 PCG with IntN in canonical stratum and unit
// order. Seeds may be zero. No global RNG, confidence level, or default number
// of replicates is supplied. Declaration is not statistical admission.
type ResamplingPlan struct {
	Version    uint16           `json:"version"`
	Population artifact.ID      `json:"population"`
	Assumption artifact.ID      `json:"assumption"`
	Evidence   artifact.ID      `json:"evidence"`
	Method     ResamplingMethod `json:"method"`
	SeedHigh   uint64           `json:"seed_high"`
	SeedLow    uint64           `json:"seed_low"`
	Draws      uint64           `json:"draws"`
	Units      []SamplingUnit   `json:"units"`
	ID         artifact.ID      `json:"-"`
}

var resamplingCodec = artifact.JSONDocumentCodec(
	"block resampling plan", artifact.KindProfile,
	"application/vnd.overgo.block-resampling+json", "overgo/block-resampling/v1",
	canonicalizeResamplingPlan, func(value ResamplingPlan) artifact.ID { return value.ID },
	func(value *ResamplingPlan, id artifact.ID) { value.ID = id }, cloneResamplingPlan,
)

// NewResamplingPlan canonicalizes an explicit, evidence-bound unit partition.
func NewResamplingPlan(value ResamplingPlan) (ResamplingPlan, error) {
	return resamplingCodec.NewInitial(value)
}

// Batch constructs one caller-owned publication with the plan's authorities:
// its population, assumption, evidence, strata, units and members.
func (value ResamplingPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{value.Population, value.Assumption, value.Evidence}
	for _, unit := range value.Units {
		parents = append(parents, unit.ID, unit.Stratum)
		parents = append(parents, unit.Members...)
	}
	return resamplingCodec.Batch(key, value, artifact.UniqueDependencyLineage(value.ID, parents...), nil)
}

func cloneResamplingPlan(value ResamplingPlan) ResamplingPlan {
	value.Units = slices.Clone(value.Units)
	for index := range value.Units {
		value.Units[index].Members = slices.Clone(value.Units[index].Members)
	}
	return value
}

func canonicalizeResamplingPlan(value *ResamplingPlan) error {
	if value.Version != artifact.InitialDocumentVersion || !value.Population.Valid() ||
		value.Assumption.Kind() != artifact.KindProfile || value.Evidence.Kind() != artifact.KindEvidence ||
		value.Draws == 0 || len(value.Units) == 0 {
		return errors.New("tensorstats: resampling requires population, assumption, evidence, units, and positive draws")
	}
	if value.Method != BlockBootstrap {
		return errors.New("tensorstats: unsupported block resampling method")
	}
	slices.SortFunc(value.Units, func(left, right SamplingUnit) int {
		if order := artifact.CompareID(left.Stratum, right.Stratum); order != 0 {
			return order
		}
		return artifact.CompareID(left.ID, right.ID)
	})
	units := make(map[artifact.ID]bool, len(value.Units))
	members := make(map[artifact.ID]bool)
	for _, unit := range value.Units {
		if !unit.ID.Valid() || unit.Stratum.Kind() != artifact.KindProfile || len(unit.Members) == 0 || units[unit.ID] {
			return errors.New("tensorstats: units require unique identities, explicit strata, and ordered members")
		}
		units[unit.ID] = true
		for _, member := range unit.Members {
			if !member.Valid() || members[member] {
				return errors.New("tensorstats: member belongs to more than one position or unit")
			}
			members[member] = true
		}
	}
	return nil
}

// BlockDraw records ordered whole-unit selections for one zero-based replicate.
// Bootstrap repetitions are intentional. The referenced plan supplies membership
// and fixed stratum slots; no per-member duplication creates new independent units.
type BlockDraw struct {
	Plan    artifact.ID
	Ordinal uint64
	Units   []artifact.ID
}

// ResampleBlocks streams reproducible whole-unit selections in canonical strata.
// Memory depends on the partition and one draw, not draws times member count.
// Each callback owns its unit slice. Errors/cancellation stop further callbacks;
// already visited draws remain partial work and do not claim a completed run.
func ResampleBlocks(ctx context.Context, plan ResamplingPlan, visit func(BlockDraw) error) error {
	if ctx == nil || visit == nil {
		return errors.New("tensorstats: resampling requires context and consumer")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := resamplingCodec.Content(plan); err != nil {
		return err
	}
	plan = cloneResamplingPlan(plan)
	random := rand.New(rand.NewPCG(plan.SeedHigh, plan.SeedLow))
	for ordinal := range plan.Draws {
		if err := ctx.Err(); err != nil {
			return err
		}
		draw := BlockDraw{Plan: plan.ID, Ordinal: ordinal, Units: make([]artifact.ID, len(plan.Units))}
		for start := 0; start < len(plan.Units); {
			end := start
			for end < len(plan.Units) && plan.Units[end].Stratum == plan.Units[start].Stratum {
				end++
			}
			for index := start; index < end; index++ {
				draw.Units[index] = plan.Units[start+random.IntN(end-start)].ID
			}
			start = end
		}
		if err := visit(draw); err != nil {
			return err
		}
	}
	return nil
}
