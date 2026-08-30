package loop

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
)

const (
	// CircuitBreakerMediaType identifies breaker transition evidence.
	CircuitBreakerMediaType = "application/vnd.overgo.circuit-breaker-transition+json"
	// CircuitBreakerSchema identifies the transition contract.
	CircuitBreakerSchema = "overgo/circuit-breaker-transition/v1"
	// BreakerClear records a judged window with no sustained regression;
	// it resets the confirmation count.
	BreakerClear = "clear"
	// BreakerAdvisory is the first confirmation step: a single breached
	// window can advise and nothing more.
	BreakerAdvisory = "advisory"
	// BreakerFinding is the second consecutive breached window.
	BreakerFinding = "finding"
	// BreakerQuarantine is the third consecutive breached window.
	BreakerQuarantine = "quarantine"
	// BreakerOperatorStop is the fourth and every later consecutive
	// breached window: automation halts and a person decides.
	BreakerOperatorStop = "operator-stop"
	// BreakerCauseRegression marks a window whose comparable observations
	// all crossed the alarm boundary on the regression side.
	BreakerCauseRegression = "regression"
	// BreakerCauseCoverage marks a window whose comparable observations
	// fell below the derived requirement while work kept arriving —
	// a coverage regression, not a data gap.
	BreakerCauseCoverage = "coverage"
)

// LiveObservation is one unit of live work: its measured metric value and
// whether it is comparable. Skipped and inapplicable work sets Applicable
// false and never enters success or failure ratios.
type LiveObservation struct {
	Value      float64 `json:"value"`
	Applicable bool    `json:"applicable"`
}

// CircuitBreakerTransition is the published verdict of judging one live
// window against its derived safety boundaries: the level reached, the
// exact cause, the counts behind the ratio, and the confirmation state
// the next judgment continues from.
type CircuitBreakerTransition struct {
	Metric              string      `json:"metric"`
	Level               string      `json:"level"`
	Cause               string      `json:"cause,omitempty"`
	Window              artifact.ID `json:"window"`
	Comparable          uint64      `json:"comparable"`
	Excluded            uint64      `json:"excluded"`
	Breached            uint64      `json:"breached"`
	ConsecutiveBreaches uint64      `json:"consecutive_breaches"`
}

// breakerEscalation is the recorded confirmation policy: each additional
// consecutive breached window climbs one step, and the ladder never skips.
func breakerEscalation(confirmed uint64) string {
	switch confirmed {
	case 1:
		return BreakerAdvisory
	case 2:
		return BreakerFinding
	case 3:
		return BreakerQuarantine
	default:
		return BreakerOperatorStop
	}
}

// JudgeCircuitBreaker judges one live window of observations against a
// derived safety window. It requires enough total observations to fill
// the derived comparison window; comparable observations short of that
// requirement while work kept arriving is itself a coverage regression. A
// regression breach requires every comparable observation in the window
// to cross the boundary on the regression side — the derivation already
// sized the window so that joint luck stays under the registered bound —
// and escalation follows the recorded confirmation ladder from the
// caller's prior consecutive-breach count.
func JudgeCircuitBreaker(
	window evaluation.LiveSafetyWindow,
	windowEvidence artifact.ID,
	observations []LiveObservation,
	priorBreaches uint64,
) (CircuitBreakerTransition, error) {
	if window.RequiredObservations == 0 || window.Metric == "" {
		return CircuitBreakerTransition{}, errors.New("loop: circuit breaker requires a derived safety window")
	}
	if windowEvidence.Kind() != artifact.KindEvidence {
		return CircuitBreakerTransition{}, errors.New("loop: circuit breaker requires the published window derivation")
	}
	if uint64(len(observations)) < window.RequiredObservations {
		return CircuitBreakerTransition{}, fmt.Errorf(
			"loop: %d observations cannot fill the derived comparison window of %d",
			len(observations), window.RequiredObservations,
		)
	}
	transition := CircuitBreakerTransition{
		Metric: window.Metric, Window: windowEvidence,
	}
	for _, observation := range observations {
		if !observation.Applicable {
			transition.Excluded++
			continue
		}
		transition.Comparable++
		regressed := observation.Value < window.Lower
		if window.Direction == runrecord.DirectionMinimize {
			regressed = observation.Value > window.Upper
		}
		if regressed {
			transition.Breached++
		}
	}
	breached := false
	if transition.Comparable < window.RequiredObservations {
		transition.Cause = BreakerCauseCoverage
		breached = true
	} else if transition.Breached == transition.Comparable {
		transition.Cause = BreakerCauseRegression
		breached = true
	}
	if !breached {
		transition.Level = BreakerClear
		transition.ConsecutiveBreaches = 0
		return transition, nil
	}
	transition.ConsecutiveBreaches = priorBreaches + 1
	transition.Level = breakerEscalation(transition.ConsecutiveBreaches)
	return transition, nil
}

// PublishCircuitBreakerTransition commits one judged transition as
// immutable evidence citing the window derivation it was judged against.
func PublishCircuitBreakerTransition(
	ctx context.Context,
	store artifact.Repository,
	transition CircuitBreakerTransition,
) (artifact.ID, error) {
	if transition.Level == "" || transition.Metric == "" {
		return artifact.ID{}, errors.New("loop: only a judged breaker transition can publish")
	}
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: CircuitBreakerMediaType, Schema: CircuitBreakerSchema,
	}
	content, err := artifact.JSONContent(contract, transition)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"live-safety/breaker/"+content.Descriptor.ID.String(),
		[]artifact.Content{content},
		artifact.DependencyLineage(content.Descriptor.ID, transition.Window),
		nil,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return content.Descriptor.ID, nil
}
