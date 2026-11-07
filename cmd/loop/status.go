package main

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/plan"
)

// statusFile is where go run ./cmd/loop -status keeps what it last read, in the checkout's process state, and percent
// turns the refused share into the percentage it prints.
const (
	statusFile = "status.json"
	percent    = 100
)

// workStatus is the state of the work in a checkout, as the owner asks for it: the dispatched step or why none is,
// the last commit, the decisions waiting on the owner's answer while the other rows run, and the week's pace.
type workStatus struct {
	Read            time.Time    `json:"read"`
	Dispatched      string       `json:"dispatched"`
	LastCommit      string       `json:"last_commit"`
	WaitingForOwner []string     `json:"waiting_for_owner,omitempty"`
	Week            landing.Pace `json:"week"`
}

// statusOf is the status of a checkout whose dispatch line, last commit, plan and pace are given.
func statusOf(read time.Time, dispatched, lastCommit string, document plan.Plan, pace landing.Pace) workStatus {
	status := workStatus{Read: read, Dispatched: dispatched, LastCommit: lastCommit, Week: pace}
	for _, decision := range document.Decisions {
		if decision.Answer == "" && (decision.Doctrine || decision.OwnersAct) {
			status.WaitingForOwner = append(status.WaitingForOwner, decision.ID+": "+decision.Question)
		}
	}
	slices.Sort(status.WaitingForOwner)
	return status
}

// writeStatus reads the status of the checkout at root, keeps it in the process state and says it.
func writeStatus(root string, said io.Writer) error {
	ctx := context.Background()
	dispatch, err := plan.ResolveDispatch(ctx, root, plan.DispatchRequest{})
	dispatched := dispatch.Line
	if err != nil {
		dispatched = "no dispatch: " + err.Error()
	}
	last, err := gitauthority.Query(ctx, root, "log", "-1", "--format=%h %cI %s")
	if err != nil {
		return err
	}
	document, err := plan.Load("")
	if err != nil {
		return err
	}
	now := time.Now()
	pace, err := landing.WeekPace(root, now)
	if err != nil {
		return err
	}
	status := statusOf(now, dispatched, strings.TrimSpace(string(last)), document, pace)
	path, err := statePath(root, statusFile)
	if err != nil {
		return err
	}
	if err := jsonfile.Write(path, status, startupFileMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(said, "dispatched: %s\nlast commit: %s\nwaiting for the owner: %d\nweek: %.1f commits a day, %.0f%% of attempts refused, %.1f idle hours\n",
		status.Dispatched, status.LastCommit, len(status.WaitingForOwner), status.Week.LandingsPerDay, percent*status.Week.RefusedShare, status.Week.IdleHours)
	return err
}
