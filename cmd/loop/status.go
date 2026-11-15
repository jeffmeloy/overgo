package main

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strconv"
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
	// Day is the day's delivery, the five measures the session start prints too (review-fixes/status-measures).
	Day landing.Delivery `json:"day"`
	// Worklist is the refusals by the rule or test that made each, most first (refusal-advice/worklist).
	Worklist []landing.WorkItem `json:"worklist,omitzero"`
	// Lanes are where each cycle lane stands (lane-passes/do).
	Lanes []laneStanding `json:"lanes,omitzero"`
}

// laneStanding is a cycle lane as the status says it: the commit it last passed at, the trunk's commits since, and its
// last run, kept by validate -run wherever it ran (plan.RecordLaneRun).
type laneStanding struct {
	Lane   string       `json:"lane"`
	Passed string       `json:"passed,omitzero"`
	Since  int          `json:"since"`
	Last   plan.LaneRun `json:"last"`
}

// laneStandings reads where each of the cycle's declared lanes stands in the record kept at root, in the cycle's order:
// a lane that never ran says so, and a name the record holds that no lane is goes unread, as the seven fixture lanes a
// test of the runner had written into the trunk's record were printed (lane-record/do).
func laneStandings(ctx context.Context, root string) []laneStanding {
	runs := plan.ReadLaneRuns(root)
	var standings []laneStanding
	for _, declared := range landing.CycleLanes() {
		lane := declared.Name
		standing := laneStanding{Lane: lane, Last: runs[lane].Last}
		if passed := runs[lane].Passed.Commit; passed != "" {
			if count, err := gitauthority.Query(ctx, root, "rev-list", "--count", passed+"..HEAD"); err == nil {
				standing.Passed = passed
				standing.Since, _ = strconv.Atoi(strings.TrimSpace(string(count)))
			}
		}
		standings = append(standings, standing)
	}
	return standings
}

// String is a lane's standing in a line: where it last passed and the commits since, and how its last run ended, or
// that it never ran.
func (standing laneStanding) String() string {
	if standing.Last.Commit == "" {
		return "lane " + standing.Lane + ": never run"
	}
	said := "lane " + standing.Lane + ": never passed"
	if standing.Passed != "" {
		said = fmt.Sprintf("lane %s: passed at %.12s, %d commits since", standing.Lane, standing.Passed, standing.Since)
	}
	verdict := "failed"
	if standing.Last.Passed {
		verdict = "passed"
	}
	return fmt.Sprintf("%s; last run %s at %.12s, %s", said, verdict, standing.Last.Commit, standing.Last.Ended.Local().Format("Jan 2 15:04"))
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
	if status.Day, err = landing.DayDelivery(ctx, root, now, landing.ReportFiling); err != nil {
		return err
	}
	if status.Worklist, err = landing.Worklist(root); err != nil {
		return err
	}
	status.Lanes = laneStandings(ctx, root)
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
