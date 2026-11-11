package plan

import (
	"errors"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// cardHoldFile is the record a session's own model run keeps in the process state of the checkout it works from while
// it holds the graphics card: a campaign of renders and judged readings, a training run. Dispatch reads it beside the
// measurement run's record, so through the hours such a run takes the step offered next is one the host can do
// (fail-early-runs/card-held): through guidance-ops' readings, about two days of the card on 2026-11-11, the stop
// check would otherwise have sent each turn back to a step that waited on the card.
const cardHoldFile = "card_hold.json"

// CardHold is a session's run that holds the card: its process, what it does, and when it began. It holds the card
// while its process lives.
type CardHold struct {
	Process int       `json:"process"`
	Purpose string    `json:"purpose"`
	Started time.Time `json:"started"`
}

// HoldCard records that the run of process holds the card for purpose, from now, in the process state of the checkout
// at root. A run that ends needs no release: the hold lapses with its process.
func HoldCard(root string, process int, purpose string) error {
	if !processcontrol.ProcessAlive(process) || strings.TrimSpace(purpose) == "" {
		return errors.New("plan: a card hold names a live process and what it does")
	}
	return jsonfile.Write(filepath.Join(root, processlock.StateDirectory, cardHoldFile), CardHold{Process: process, Purpose: purpose, Started: time.Now()}, runRecordMode)
}

// ReadCardHold is the card hold recorded at root, and whether one is.
func ReadCardHold(root string) (CardHold, bool) {
	var hold CardHold
	if jsonfile.Decode(filepath.Join(root, processlock.StateDirectory, cardHoldFile), &hold) != nil {
		return CardHold{}, false
	}
	return hold, hold.Process != 0
}

// HoldsTheCard says the run holds the card now: its process lives.
func (hold CardHold) HoldsTheCard() bool { return processcontrol.ProcessAlive(hold.Process) }

// cardHeld says a run holds the card at root: a measurement run, or a session's own (CardHold).
func cardHeld(root string) bool {
	run, recorded := ReadMeasurementRun(root)
	hold, held := ReadCardHold(root)
	return recorded && run.HoldsTheCard() || held && hold.HoldsTheCard()
}
