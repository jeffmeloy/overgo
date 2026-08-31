package evaluation

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// CapabilityGapKind names the three typed gap findings the composition
// driver consumes.
type CapabilityGapKind string

const (
	// GapFrontier marks a recipe whose latest measurement trails the best
	// latest measurement of the same metric across the store.
	GapFrontier CapabilityGapKind = "frontier-gap"
	// GapRegression marks a recipe whose latest measurement fell below its
	// own best earlier measurement of the same metric.
	GapRegression CapabilityGapKind = "regression"
	// GapWeakestDomain marks, per recipe, the metric with its largest
	// frontier shortfall — the domain to improve first.
	GapWeakestDomain CapabilityGapKind = "weakest-domain"
)

// CapabilityGapTarget is one typed improvement target derived from
// admitted evaluation evidence: the exact recipe and model, the metric,
// the direction-oriented measured and reference values, the shortfall,
// and the evidence documents behind both sides.
type CapabilityGapTarget struct {
	Kind      CapabilityGapKind `json:"kind"`
	Metric    string            `json:"metric"`
	Model     artifact.ID       `json:"model"`
	Recipe    artifact.ID       `json:"recipe"`
	Measured  float64           `json:"measured"`
	Reference float64           `json:"reference"`
	Gap       float64           `json:"gap"`
	Evidence  []artifact.ID     `json:"evidence"`
}

// EvaluationStoreReader is the store surface gap derivation needs: the
// ordinary reader plus introduction ordering, which decides which
// measurement of a recipe is its latest.
type EvaluationStoreReader interface {
	artifact.Reader
	ArtifactIntroduction(context.Context, artifact.ID) (overgodb.ArtifactIntroduction, bool, error)
}

type gapMeasurement struct {
	evidence artifact.ID
	model    artifact.ID
	sequence uint64
	quality  float64
}

// DeriveCapabilityGapTargets turns admitted evaluation evidence into the
// typed targets the composition driver consumes. Every cited document
// admits through RequireEvaluationEvidence — the same ordered gate owner
// evidence routing uses; there is no second domain-mapping path — and
// quality orients in each metric's improvement direction exactly as
// routing does, so maximize and minimize metrics compare through one
// inequality. Per recipe and metric, the latest admitted measurement
// stands; a latest below the store frontier is a frontier gap, a latest
// below the recipe's own best earlier measurement is a regression, and
// each recipe's largest frontier shortfall is its weakest domain. Targets
// order deterministically and every target cites both sides' evidence.
func DeriveCapabilityGapTargets(
	ctx context.Context,
	store EvaluationStoreReader,
	evidence []artifact.ID,
) ([]CapabilityGapTarget, error) {
	if len(evidence) == 0 {
		return nil, errors.New("evaluation: gap derivation requires admitted evaluation evidence")
	}
	type seriesKey struct {
		recipe artifact.ID
		metric string
	}
	series := make(map[seriesKey][]gapMeasurement)
	directions := make(map[string]runrecord.Direction)
	for _, id := range evidence {
		document, err := RequireEvaluationEvidence(ctx, store, id)
		if err != nil {
			return nil, err
		}
		introduction, found, err := store.ArtifactIntroduction(ctx, id)
		if err != nil || !found {
			return nil, errors.Join(fmt.Errorf("evaluation: evidence %s has no introduction record", id), err)
		}
		for _, metric := range document.Metrics {
			if !validMetricDirection(metric.Direction) {
				continue
			}
			if known, seen := directions[metric.Name]; seen && known != metric.Direction {
				return nil, fmt.Errorf(
					"evaluation: metric %q is measured in conflicting directions; the gap is not comparable", metric.Name,
				)
			}
			directions[metric.Name] = metric.Direction
			key := seriesKey{recipe: document.Recipe, metric: metric.Name}
			series[key] = append(series[key], gapMeasurement{
				evidence: id, model: document.ModelDefinition,
				sequence: introduction.Sequence,
				quality:  signedQuality(metric.Value, metric.Direction),
			})
		}
	}
	type latestReading struct {
		key    seriesKey
		latest gapMeasurement
	}
	latests := make([]latestReading, 0, len(series))
	targets := make([]CapabilityGapTarget, 0)
	for key, measurements := range series {
		slices.SortFunc(measurements, func(a, b gapMeasurement) int {
			if a.sequence != b.sequence {
				return int(a.sequence) - int(b.sequence)
			}
			return strings.Compare(a.evidence.String(), b.evidence.String())
		})
		latest := measurements[len(measurements)-1]
		latests = append(latests, latestReading{key: key, latest: latest})
		bestEarlier := latest
		for _, earlier := range measurements[:len(measurements)-1] {
			if earlier.quality > bestEarlier.quality || bestEarlier == latest {
				bestEarlier = earlier
			}
		}
		if bestEarlier != latest && latest.quality < bestEarlier.quality {
			targets = append(targets, CapabilityGapTarget{
				Kind: GapRegression, Metric: key.metric,
				Model: latest.model, Recipe: key.recipe,
				Measured: latest.quality, Reference: bestEarlier.quality,
				Gap:      bestEarlier.quality - latest.quality,
				Evidence: []artifact.ID{latest.evidence, bestEarlier.evidence},
			})
		}
	}
	frontier := make(map[string]gapMeasurement)
	for _, reading := range latests {
		best, seen := frontier[reading.key.metric]
		if !seen || reading.latest.quality > best.quality {
			frontier[reading.key.metric] = reading.latest
		}
	}
	weakest := make(map[artifact.ID]CapabilityGapTarget)
	for _, reading := range latests {
		best := frontier[reading.key.metric]
		if reading.latest.quality >= best.quality {
			continue
		}
		target := CapabilityGapTarget{
			Kind: GapFrontier, Metric: reading.key.metric,
			Model: reading.latest.model, Recipe: reading.key.recipe,
			Measured: reading.latest.quality, Reference: best.quality,
			Gap:      best.quality - reading.latest.quality,
			Evidence: []artifact.ID{reading.latest.evidence, best.evidence},
		}
		targets = append(targets, target)
		current, seen := weakest[reading.key.recipe]
		if !seen || target.Gap > current.Gap ||
			(target.Gap == current.Gap && target.Metric < current.Metric) {
			domain := target
			domain.Kind = GapWeakestDomain
			weakest[reading.key.recipe] = domain
		}
	}
	targets = append(targets, slices.Collect(maps.Values(weakest))...)
	slices.SortFunc(targets, func(a, b CapabilityGapTarget) int {
		if by := strings.Compare(string(a.Kind), string(b.Kind)); by != 0 {
			return by
		}
		if by := strings.Compare(a.Metric, b.Metric); by != 0 {
			return by
		}
		return strings.Compare(a.Recipe.String(), b.Recipe.String())
	})
	return targets, nil
}
