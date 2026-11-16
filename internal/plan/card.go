package plan

import (
	"slices"

	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
)

// CardCheckSetting is how a step's check asks for the graphics card: the setting cudatest.Require reads, without which
// it skips every test that reaches the card.
const CardCheckSetting = "OVERGO_CUDA_TEST=1"

// OnTheCard says a step's check runs on the graphics card: it sets CardCheckSetting. A check that needs model files
// alone (the modeltest tag) runs on the host (cheap-first/form).
func OnTheCard(verify string) bool {
	parsed, err := runrecord.ParseVerify(verify)
	return err == nil && slices.ContainsFunc(parsed.Segments, func(segment runrecord.VerifySegment) bool {
		return slices.Contains(segment.Env, CardCheckSetting)
	})
}

// NeedsTheCard says the step's work needs the graphics card: it is marked so (NeedsCard), or its check runs on the
// card. Of 54 steps marked on 2026-11-14, 49 had a check on the card, and one such check carried no mark.
func (step Step) NeedsTheCard() bool { return step.NeedsCard || OnTheCard(step.Verify) }

// cardHolder names who holds the graphics card at this reading: the measurement run while it measures
// (MeasurementRun.HoldsTheCard), else another process that holds a card's turn, whose name begins with turn, as its
// record says; nothing while the card is free. Read from the measurement run alone, the card counted free while a
// render, a run of the lanes or a landing held its turn for hours (card-free-dispatch/do).
func cardHolder(run MeasurementRun, recorded bool, turn string) (string, error) {
	if recorded && run.HoldsTheCard() {
		return "the measurement run", nil
	}
	held, err := processcontrol.TurnsHeld(turn)
	if err != nil || len(held) == 0 {
		return "", err
	}
	return held[0], nil
}

// cardWaits says the step waits for the card at this reading: it needs the card and another holds it.
func (document Plan) cardWaits(step Step) bool {
	return document.cardHolder != "" && step.NeedsTheCard()
}
