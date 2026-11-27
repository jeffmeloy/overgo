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
// the last commit, the decisions waiting on the owner's answer (an act only the owner takes; one that changes a standing
// rule is the lead's to answer, doctrine-answer) while the other rows run, and the week's pace.
type workStatus struct {
	Read            time.Time    `json:"read"`
	Dispatched      string       `json:"dispatched"`
	LastCommit      string       `json:"last_commit"`
	WaitingForOwner []string     `json:"waiting_for_owner,omitempty"`
	Week            landing.Pace `json:"week"`
	// Day is the day's delivery, the five measures the session start prints too (review-fixes/status-measures).
	Day landing.Delivery `json:"day"`
	// Worklist is the refusals by the rule or test that made each, most first (refusal-advice/worklist).
	Worklist []landing.WorkItem `json:"worklist,omitzero"`
	// Lanes are where each cycle lane stands (lane-passes/do).
	Lanes []laneStanding `json:"lanes,omitzero"`
}

// laneStanding is a cycle lane as the status says it, from the results its tests keep wherever they ran
// (plan.RecordTestResults, lanes-record): how many passed and failed, and the newest.
type laneStanding struct {
	Lane   string          `json:"lane"`
	Passed int             `json:"passed"`
	Failed int             `json:"failed"`
	Newest plan.TestResult `json:"newest,omitzero"`
}

// laneStandings reads where each of the cycle's declared lanes stands in the results kept at root, in the cycle's order:
// a lane run whole by its own result, a lane run test by test by its packages' tests'; a result no declared lane holds
// goes unread, as the seven fixture lanes a test of the runner had written into the trunk's record were printed
// (lane-record/do).
func laneStandings(root string) []laneStanding {
	results := plan.ReadTestResults(root)
	var standings []laneStanding
	for _, lane := range landing.CycleLanes() {
		standing := laneStanding{Lane: lane.Name}
		packages := map[string]bool{}
		for _, arguments := range lane.Each {
			packages["overgo/"+strings.TrimPrefix(arguments[len(arguments)-1], "./")] = true
		}
		for key, result := range results {
			if pkg, _, _ := strings.Cut(key, " "); key != plan.LaneTest(lane.Name) && !packages[pkg] {
				continue
			}
			if result.Failed {
				standing.Failed++
			} else {
				standing.Passed++
			}
			if result.Ended.After(standing.Newest.Ended) {
				standing.Newest = result
			}
		}
		standings = append(standings, standing)
	}
	return standings
}

// String is a lane's standing in a line: how many of its results passed and failed and when the newest was taken, or
// that it never ran.
func (standing laneStanding) String() string {
	if standing.Newest.Commit == "" {
		return "lane " + standing.Lane + ": never run"
	}
	return fmt.Sprintf("lane %s: %d passed, %d failed; the newest at %.12s, %s", standing.Lane, standing.Passed, standing.Failed,
		standing.Newest.Commit, standing.Newest.Ended.Local().Format("Jan 2 15:04"))
}

// statusOf is the status of a checkout whose dispatch line, last commit, plan and pace are given.
func statusOf(read time.Time, dispatched, lastCommit string, document plan.Plan, pace landing.Pace) workStatus {
	status := workStatus{Read: read, Dispatched: dispatched, LastCommit: lastCommit, Week: pace}
	for _, decision := range document.Decisions {
		if decision.Answer == "" && decision.OwnersAct {
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
	if status.Day, err = landing.DayDelivery(ctx, root, now, landing.ReportFiling); err != nil {
		return err
	}
	if status.Worklist, err = landing.Worklist(root); err != nil {
		return err
	}
	status.Lanes = laneStandings(root)
	path, err := statePath(root, statusFile)
	if err != nil {
		return err
	}
	if err := jsonfile.Write(path, status, startupFileMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(said, "dispatched: %s\nlast commit: %s\nwaiting for the owner: %d\nweek: %.1f commits a day, %.0f%% of attempts failed, %.1f idle hours\n",
		status.Dispatched, status.LastCommit, len(status.WaitingForOwner), status.Week.LandingsPerDay, percent*status.Week.RefusedShare, status.Week.IdleHours)
	if err != nil {
		return err
	}
	if _, err = fmt.Fprintln(said, status.Day); err != nil {
		return err
	}
	for _, standing := range status.Lanes {
		if _, err := fmt.Fprintln(said, standing); err != nil {
			return err
		}
	}
	for _, item := range status.Worklist {
		if _, err := fmt.Fprintf(said, "refused by %s: %d (%d said its edit); latest: %s\n", item.By, item.Refusals, item.Edits, item.Latest); err != nil {
			return err
		}
	}
	return nil
}
