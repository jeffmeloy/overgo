package landing

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/processlock"
)

// ReportFiling is the commit that filed the report's rows (guidance-ops, comfy-recorder and those after them): a row it
// added to the plan is one of them, by its identity, never by a commit's wording.
const ReportFiling = "9479f5815"

// deliveryWindow is the span the status reads the work's delivery over, a day, and unknownCount the count of a measure
// no record holds.
const (
	deliveryWindow = 24 * time.Hour
	unknownCount   = -1
)

// planOnlyFiles are the files a commit may touch and still be a change of the plan alone.
var planOnlyFiles = []string{"docs/plan.json", "docs/sqa_findings.json"}

// Delivery is what the day before now says of the work's delivery, read from Git and the gate's run records of a
// checkout and nothing else (review-fixes/status-measures, SQA-376): the commits, those changing the plan alone apart;
// the steps they landed, the steps added to the plan and those retired; how many landings were of the report's rows;
// the commit attempts the gate refused, by the stage that refused them, with the refused runs of unknown kind (records
// from before runs said their kind) and the preflights apart; and the gate's hours over its runs, of which the waits.
// A measure no record holds is unknown (unknownCount), not zero.
type Delivery struct {
	Since             time.Time      `json:"since"`
	Revision          string         `json:"revision"`
	Commits           int            `json:"commits"`
	PlanOnly          int            `json:"plan_only_commits"`
	Landed            []string       `json:"landed,omitempty"`
	Added             []string       `json:"added,omitempty"`
	Retired           []string       `json:"retired,omitempty"`
	ReportLandings    int            `json:"report_landings"`
	Refused           map[string]int `json:"refused_by_stage,omitempty"`
	UnknownKind       map[string]int `json:"refused_of_unknown_kind,omitempty"`
	Preflights        int            `json:"preflights"`
	PreflightsRefused int            `json:"preflights_refused"`
	GateHours         float64        `json:"gate_hours"`
	WaitHours         float64        `json:"wait_hours"`
}

// DayDelivery reads the delivery of the day before now in the checkout at root, its report's rows those reportFiling
// added to the plan. It writes nothing.
func DayDelivery(ctx context.Context, root string, now time.Time, reportFiling string) (Delivery, error) {
	delivery := Delivery{Since: now.Add(-deliveryWindow), ReportLandings: unknownCount, Refused: map[string]int{}, UnknownKind: map[string]int{}}
	head, err := gitauthority.Query(ctx, root, "rev-parse", "--short", "HEAD")
	if err != nil {
		return Delivery{}, err
	}
	delivery.Revision = strings.TrimSpace(string(head))
	since := delivery.Since.Format(time.RFC3339)
	log, err := gitauthority.Query(ctx, root, "log", "--since="+since, "--format=%x1e%B%x1f", "--name-only", "HEAD")
	if err != nil {
		return Delivery{}, err
	}
	for entry := range strings.SplitSeq(string(log), "\x1e") {
		body, files, found := strings.Cut(entry, "\x1f")
		if !found {
			continue
		}
		delivery.Commits++
		touched := strings.Fields(files)
		if len(touched) != 0 && !slices.ContainsFunc(touched, func(file string) bool { return !slices.Contains(planOnlyFiles, file) }) {
			delivery.PlanOnly++
		}
		for line := range strings.Lines(body) {
			if row, landed := strings.CutPrefix(strings.TrimSpace(line), "Landed-Row: "); landed {
				delivery.Landed = append(delivery.Landed, row)
			}
			if row, retired := strings.CutPrefix(strings.TrimSpace(line), "Retired-Row: "); retired {
				delivery.Retired = append(delivery.Retired, row)
			}
		}
	}
	slices.Sort(delivery.Landed)
	slices.Sort(delivery.Retired)
	if delivery.Added, err = addedSteps(ctx, root, since, delivery.Landed, delivery.Retired); err != nil {
		return Delivery{}, err
	}
	if rows, err := filedRows(ctx, root, reportFiling); err == nil {
		delivery.ReportLandings = 0
		for _, landed := range delivery.Landed {
			item, _, _ := strings.Cut(landed, "/")
			if slices.Contains(rows, item) {
				delivery.ReportLandings++
			}
		}
	}
	return delivery, delivery.readRuns(root)
}

// readRuns counts the gate's runs that started in the window, from its run records, read and never created.
func (delivery *Delivery) readRuns(root string) error {
	data, err := os.ReadFile(filepath.Join(root, processlock.StateDirectory, spentRecord))
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var runs []spent
	if err := json.Unmarshal(data, &runs); err != nil {
		return err
	}
	for _, run := range runs {
		if run.Started.Before(delivery.Since) {
			continue
		}
		delivery.GateHours += run.Ended.Sub(run.Started).Hours()
		delivery.WaitHours += (run.Queue + run.Measurement + run.Turn + run.Device).Hours()
		switch {
		case run.Outcome == outcomeLanded:
		case run.Outcome == outcomePreflight:
			delivery.Preflights++
		case run.Kind == kindPreflight:
			delivery.Preflights++
			delivery.PreflightsRefused++
		case run.Kind == kindCommit:
			delivery.Refused[run.Outcome]++
		default:
			delivery.UnknownKind[run.Outcome]++
		}
	}
	return nil
}

// planSteps are the steps of the plan at a revision, item/step, read for their identities alone.
func planSteps(ctx context.Context, root, revision string) ([]string, error) {
	data, err := gitauthority.Query(ctx, root, "show", revision+":docs/plan.json")
	if err != nil {
		return nil, err
	}
	var document struct {
		Items []struct {
			ID    string `json:"id"`
			Steps []struct {
				ID string `json:"id"`
			} `json:"steps"`
		} `json:"items"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, err
	}
	var steps []string
	for _, item := range document.Items {
		for _, step := range item.Steps {
			steps = append(steps, item.ID+"/"+step.ID)
		}
	}
	return steps, nil
}

// addedSteps are the steps the window added to the plan: those of the head's plan, or landed or retired in the window,
// that the plan before the window did not hold. With no commit before the window, every one is added.
func addedSteps(ctx context.Context, root, since string, landed, retired []string) ([]string, error) {
	held, err := planSteps(ctx, root, "HEAD")
	if err != nil {
		return nil, err
	}
	var before []string
	if base, err := gitauthority.Query(ctx, root, "rev-list", "-1", "--before="+since, "HEAD"); err == nil && strings.TrimSpace(string(base)) != "" {
		if before, err = planSteps(ctx, root, strings.TrimSpace(string(base))); err != nil {
			return nil, err
		}
	}
	var added []string
	for _, step := range slices.Concat(held, landed, retired) {
		if !slices.Contains(before, step) && !slices.Contains(added, step) {
			added = append(added, step)
		}
	}
	slices.Sort(added)
	return added, nil
}

// filedRows are the rows a commit added to the plan.
func filedRows(ctx context.Context, root, filing string) ([]string, error) {
	after, err := planSteps(ctx, root, filing)
	if err != nil {
		return nil, err
	}
	before, err := planSteps(ctx, root, filing+"^")
	if err != nil {
		return nil, err
	}
	var rows []string
	for _, step := range after {
		item, _, _ := strings.Cut(step, "/")
		if !slices.ContainsFunc(before, func(held string) bool { return strings.HasPrefix(held, item+"/") }) && !slices.Contains(rows, item) {
			rows = append(rows, item)
		}
	}
	return rows, nil
}

// String is the delivery as the session start and the status say it, in one line.
func (delivery Delivery) String() string {
	stages := func(counts map[string]int) string {
		var parts []string
		for _, stage := range slices.Sorted(func(yield func(string) bool) {
			for stage := range counts {
				if !yield(stage) {
					return
				}
			}
		}) {
			parts = append(parts, fmt.Sprintf("%s %d", stage, counts[stage]))
		}
		return strings.Join(parts, ", ")
	}
	refused := 0
	for _, count := range delivery.Refused {
		refused += count
	}
	unknown := 0
	for _, count := range delivery.UnknownKind {
		unknown += count
	}
	report := "unknown"
	if delivery.ReportLandings != unknownCount {
		report = fmt.Sprint(delivery.ReportLandings)
	}
	return fmt.Sprintf("day to %s: %d commits (%d plan-only); steps %d landed, %d added, %d retired; report rows landed %s; "+
		"%d commit attempts refused (%s), %d refused of unknown kind (%s), %d preflights (%d refused); gate %.1f h, %.1f h of it waiting",
		delivery.Revision, delivery.Commits, delivery.PlanOnly, len(delivery.Landed), len(delivery.Added), len(delivery.Retired), report,
		refused, stages(delivery.Refused), unknown, stages(delivery.UnknownKind), delivery.Preflights, delivery.PreflightsRefused,
		delivery.GateHours, delivery.WaitHours)
}
