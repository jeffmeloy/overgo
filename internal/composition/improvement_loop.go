package composition

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

const (
	// DriverAttemptMediaType identifies driver attempt records.
	DriverAttemptMediaType = "application/vnd.overgo.composition-driver-attempt+json"
	// DriverAttemptSchema identifies the attempt record contract.
	DriverAttemptSchema = "overgo/composition-driver-attempt/v1"
	// DriverStopBudget marks a run that spent its derived attempt budget.
	DriverStopBudget = "budget"
	// DriverStopSaturation marks a run stopped by the derived
	// consecutive-failure streak.
	DriverStopSaturation = "saturation"
	// DriverStopOperator marks a run stopped by external operator authority.
	DriverStopOperator = "operator-stop"
	// DriverStopExhausted marks a run that worked every target.
	DriverStopExhausted = "targets-exhausted"
)

// CompositionTarget is one derived improvement target handed to the
// driver: the label of the capability gap it answers and the catalog
// component to improve.
type CompositionTarget struct {
	Label     string
	Component CatalogComponent
}

// CompositionDriverStages binds the tested owners the driver sequences.
// Each stage is a pure seam: enumeration, realization, scoring, and
// promotion own their own refusals, and the driver owns only the order,
// the derived bounds, and the record.
type CompositionDriverStages struct {
	Enumerate func(context.Context, CompositionTarget) ([]CompositionCandidate, error)
	Realize   func(context.Context, CompositionCandidate) (CandidateRealization, error)
	Score     func(context.Context, CandidateRealization) (CompositeScore, error)
	Promote   func(context.Context, CompositeFitnessVerdict, CandidateRealization) (artifact.ID, error)
	// OperatorStop reports external stop authority; the driver checks it
	// before every attempt and never overrides it.
	OperatorStop func() bool
}

// CompositionDriverAttempt is the published record of one attempt: the
// target, the candidate, how far the pipeline ran, the fitness verdict,
// and the exact refusal when a stage refused.
type CompositionDriverAttempt struct {
	Target    string      `json:"target"`
	Donor     artifact.ID `json:"donor,omitzero"`
	Component string      `json:"component,omitempty"`
	Composite artifact.ID `json:"composite,omitzero"`
	Promotion artifact.ID `json:"promotion,omitzero"`
	Fit       bool        `json:"fit"`
	Promoted  bool        `json:"promoted"`
	Refusal   string      `json:"refusal,omitempty"`
}

// CompositionDriverReport aggregates one closed run.
type CompositionDriverReport struct {
	Attempts   []CompositionDriverAttempt `json:"attempts"`
	Budget     uint64                     `json:"budget"`
	Saturation uint64                     `json:"saturation"`
	Stop       string                     `json:"stop"`
}

// DeriveDriverBounds derives the run's attempt budget and saturation
// streak from the measured attempt history — no free knobs. The budget is
// the measured cost of one promotion: the ceiling of attempts per
// historical promotion, so a run may spend what a success has historically
// cost and no more. Saturation is the longest historical failure streak
// that was still followed by a promotion, plus one: a streak longer than
// anything that ever recovered is evidence of exhaustion. An empty or
// promotionless history derives the most conservative bounds — one
// attempt, one-failure saturation — because nothing measured supports
// spending more.
func DeriveDriverBounds(history []bool) (budget, saturation uint64) {
	budget, saturation = 1, 1
	attempts, promotions := uint64(len(history)), uint64(0)
	longestRecovered, streak := uint64(0), uint64(0)
	for _, promoted := range history {
		if promoted {
			promotions++
			if streak > longestRecovered {
				longestRecovered = streak
			}
			streak = 0
			continue
		}
		streak++
	}
	if promotions > 0 {
		budget = (attempts + promotions - 1) / promotions
		saturation = longestRecovered + 1
	}
	return budget, saturation
}

// RunCompositionImprovementLoop closes the driver: for each target it
// enumerates, realizes the best-ranked candidate, scores it on the
// unscalarized fitness, and promotes or refuses through the ablation
// gate, publishing every attempt to the store before moving on. The run
// stops at its derived attempt budget, at the derived consecutive-failure
// saturation, on operator stop, or when the targets are exhausted — and
// the report names which.
func RunCompositionImprovementLoop(
	ctx context.Context,
	store artifact.Repository,
	targets []CompositionTarget,
	stages CompositionDriverStages,
	history []bool,
) (CompositionDriverReport, error) {
	if store == nil || stages.Enumerate == nil || stages.Realize == nil ||
		stages.Score == nil || stages.Promote == nil || stages.OperatorStop == nil {
		return CompositionDriverReport{}, errors.New("composition: the driver requires the store and every stage owner")
	}
	if len(targets) == 0 {
		return CompositionDriverReport{}, errors.New("composition: the driver requires derived targets")
	}
	budget, saturation := DeriveDriverBounds(history)
	report := CompositionDriverReport{Budget: budget, Saturation: saturation, Stop: DriverStopExhausted}
	streak := uint64(0)
	for _, target := range targets {
		if uint64(len(report.Attempts)) >= budget {
			report.Stop = DriverStopBudget
			break
		}
		if streak >= saturation {
			report.Stop = DriverStopSaturation
			break
		}
		if stages.OperatorStop() {
			report.Stop = DriverStopOperator
			break
		}
		attempt := CompositionDriverAttempt{Target: target.Label}
		candidates, err := stages.Enumerate(ctx, target)
		if err != nil || len(candidates) == 0 {
			attempt.Refusal = driverRefusal("enumerate", err)
		} else {
			candidate := candidates[0]
			attempt.Donor, attempt.Component = candidate.Donor, candidate.Component
			realization, err := stages.Realize(ctx, candidate)
			if err != nil {
				attempt.Refusal = driverRefusal("realize", err)
			} else {
				attempt.Composite = realization.Composite.ID
				score, err := stages.Score(ctx, realization)
				if err != nil {
					attempt.Refusal = driverRefusal("score", err)
				} else if verdicts, err := ScoreCompositeSelection([]CompositeScore{score}); err != nil {
					attempt.Refusal = driverRefusal("select", err)
				} else if verdict := verdicts[0]; !verdict.Fit {
					attempt.Refusal = fmt.Sprintf("select: regressed=%v", verdict.Regressed)
				} else {
					attempt.Fit = true
					promotion, err := stages.Promote(ctx, verdict, realization)
					if err != nil {
						attempt.Refusal = driverRefusal("promote", err)
					} else {
						attempt.Promotion, attempt.Promoted = promotion, true
					}
				}
			}
		}
		if err := publishDriverAttempt(ctx, store, attempt); err != nil {
			return report, err
		}
		report.Attempts = append(report.Attempts, attempt)
		if attempt.Promoted {
			streak = 0
		} else {
			streak++
		}
	}
	if report.Stop == DriverStopExhausted {
		switch {
		case uint64(len(report.Attempts)) >= budget:
			report.Stop = DriverStopBudget
		case streak >= saturation:
			report.Stop = DriverStopSaturation
		}
	}
	return report, nil
}

func driverRefusal(stage string, err error) string {
	if err == nil {
		return stage + ": no candidates"
	}
	return stage + ": " + err.Error()
}

func publishDriverAttempt(
	ctx context.Context, store artifact.Repository, attempt CompositionDriverAttempt,
) error {
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: DriverAttemptMediaType, Schema: DriverAttemptSchema,
	}
	content, err := artifact.JSONContent(contract, attempt)
	if err != nil {
		return err
	}
	var parents []artifact.ID
	for _, id := range []artifact.ID{attempt.Donor, attempt.Composite, attempt.Promotion} {
		if id.Valid() {
			parents = append(parents, id)
		}
	}
	batch, err := artifact.NewDocumentBatch(
		"composition/driver-attempt/"+content.Descriptor.ID.String(),
		[]artifact.Content{content},
		artifact.UniqueDependencyLineage(content.Descriptor.ID, parents...),
		nil,
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	return err
}
