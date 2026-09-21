package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Gate durations may overlap. Missing intervals cannot establish elapsed savings.
type gatePhaseReport struct {
	Gates       []retainedGate `json:"gates"`
	Unfinalized []pendingGate  `json:"unfinalized_preparations"`
	Limitations string         `json:"limitations"`
}

type pendingGate struct {
	ID      artifact.ID `json:"preparation_id"`
	TreeKey string      `json:"tree_key"`
	Started string      `json:"started"`
}

type retainedGate struct {
	ID             artifact.ID                   `json:"result_id"`
	Result         runrecord.GateResult          `json:"result"`
	Attempts       []retainedAttempt             `json:"attempts"`
	Finalizations  []artifact.ID                 `json:"finalizations"`
	SummedStepNS   uint64                        `json:"summed_step_ns"`
	Outcomes       map[runrecord.StepOutcome]int `json:"named_check_outcomes"`
	WaitingNS      *uint64                       `json:"waiting_ns"`
	UnattributedNS *uint64                       `json:"unattributed_wall_ns"`
	// SelectionCauses explains the executed package cost when the gate
	// retained its selection-cause record; earlier gates retained none.
	SelectionCauses *runrecord.SelectionHistogram `json:"selection_causes,omitempty"`
}

type retainedAttempt struct {
	ID     artifact.ID             `json:"attempt_id"`
	Record runrecord.AttemptRecord `json:"record"`
}

func printGatePhaseHistory(c cli, output io.Writer) error {
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		return err
	}
	defer store.Close()
	report, err := loadGatePhaseHistory(context.Background(), store, c)
	if err != nil {
		return err
	}
	if c.json {
		return json.NewEncoder(output).Encode(report)
	}
	return writeGatePhaseHistory(output, report)
}

func loadGatePhaseHistory(ctx context.Context, store *overgodb.Store, c cli) (gatePhaseReport, error) {
	report := gatePhaseReport{
		Gates: []retainedGate{}, Unfinalized: []pendingGate{},
		Limitations: "Named checks are not packages or tests. Durations may overlap; their sum is work, not wall. Waiting and unattributed wall are unavailable without intervals. Zero step duration is unmeasured, not free. Cancelled checks do not establish whether execution started. Unfinalized preparations are store-wide: they lack commit and plan binding and do not prove process death. Missing finalization is not completion.",
	}
	var selectedResult artifact.ID
	if c.historyResult != "" {
		var err error
		selectedResult, err = artifact.ParseID(c.historyResult)
		if err != nil || selectedResult.Kind() != artifact.KindEvidence {
			return report, fmt.Errorf("-result requires an exact evidence artifact ID")
		}
	}
	attempts := map[artifact.ID][]retainedAttempt{}
	causes := map[artifact.ID]runrecord.SelectionCauseRecord{}
	var results []runrecord.GateResult
	var lifecycles []runrecord.GateLifecycle
	contracts := []artifact.DocumentContract{
		{Kind: artifact.KindEvidence, MediaType: runrecord.AttemptMediaType, Schema: runrecord.AttemptSchema},
		{Kind: artifact.KindEvidence, MediaType: runrecord.GateMediaType, Schema: runrecord.GateSchema},
		{Kind: artifact.KindEvidence, MediaType: runrecord.GateLifecycleMediaType, Schema: runrecord.GateLifecycleSchema},
		runrecord.SelectionCauseCodec.Contract,
	}
	_, err := store.VisitDocuments(ctx, overgodb.DocumentQuery{Contracts: contracts, Order: overgodb.DocumentOldestFirst}, func(view overgodb.DocumentView) error {
		switch view.Content.Descriptor.MediaType {
		case runrecord.AttemptMediaType:
			attempt, err := runrecord.RequireAttemptRecord(ctx, store, view.Content.Descriptor.ID)
			if err != nil {
				return err
			}
			if (c.history == "all" || attempt.PlanItem == c.history) && (c.historyCommit == "" || attempt.CodeCommit == c.historyCommit) {
				attempts[attempt.Result] = append(attempts[attempt.Result], retainedAttempt{ID: attempt.ID, Record: attempt})
			}
		case runrecord.GateMediaType:
			result, err := runrecord.ParseGateResult(view.Content.Data)
			if err != nil {
				return err
			}
			if (c.historyCommit == "" || result.CodeCommit == c.historyCommit) && (!selectedResult.Valid() || result.ID == selectedResult) {
				results = append(results, result)
			}
		case runrecord.GateLifecycleMediaType:
			lifecycle, err := runrecord.ParseGateLifecycle(view.Content.Data)
			if err != nil {
				return err
			}
			lifecycles = append(lifecycles, lifecycle)
		case runrecord.SelectionCauseCodec.Contract.MediaType:
			record, err := runrecord.SelectionCauseCodec.Parse(view.Content.Data)
			if err != nil {
				return err
			}
			causes[record.Result] = record
		}
		return nil
	})
	if err != nil {
		return report, err
	}
	finalizations := map[artifact.ID][]runrecord.GateLifecycle{}
	for _, lifecycle := range lifecycles {
		if lifecycle.State == runrecord.GateFinalized && lifecycle.Result != nil {
			finalizations[*lifecycle.Result] = append(finalizations[*lifecycle.Result], lifecycle)
		}
	}
	foundResults := map[artifact.ID]bool{}
	for _, result := range results {
		foundResults[result.ID] = true
		bound := attempts[result.ID]
		if c.history != "all" && len(bound) == 0 {
			continue
		}
		for _, attempt := range bound {
			if attempt.Record.CodeCommit != result.CodeCommit || attempt.Record.Recipe != result.Recipe || attempt.Record.Outcome != result.Outcome || attempt.Record.Failure != result.Failure {
				return report, fmt.Errorf("attempt %s contradicts result %s", attempt.ID, result.ID)
			}
		}
		gate := retainedGate{ID: result.ID, Result: result, Attempts: bound, Outcomes: map[runrecord.StepOutcome]int{}}
		for _, finalization := range finalizations[result.ID] {
			if finalization.CodeCommit != result.CodeCommit || finalization.Environment != result.Environment || finalization.Outcome != result.Outcome {
				return report, fmt.Errorf("finalization %s contradicts result %s", finalization.ID, result.ID)
			}
			gate.Finalizations = append(gate.Finalizations, finalization.ID)
		}
		for _, step := range result.Steps {
			gate.SummedStepNS += step.DurationNS
			gate.Outcomes[step.Outcome]++
		}
		if record, retained := causes[result.ID]; retained {
			histogram, err := runrecord.SelectionCauseHistogram(result, record)
			if err != nil {
				return report, fmt.Errorf("selection causes %s: %w", record.ID, err)
			}
			gate.SelectionCauses = &histogram
		}
		report.Gates = append(report.Gates, gate)
	}
	for id := range attempts {
		if (!selectedResult.Valid() || id == selectedResult) && !foundResults[id] {
			return report, fmt.Errorf("retained attempt cites unavailable or mismatched gate result %s", id)
		}
	}
	debt, err := runrecord.OutstandingGateDebt(lifecycles)
	if err != nil {
		return report, err
	}
	for _, preparation := range debt {
		report.Unfinalized = append(report.Unfinalized, pendingGate{ID: preparation.ID, TreeKey: preparation.TreeKey, Started: preparation.Started})
	}
	return report, nil
}

func writeGatePhaseHistory(target io.Writer, report gatePhaseReport) error {
	output := &bytes.Buffer{}
	if len(report.Gates) == 0 {
		fmt.Fprintln(output, "no gate records match")
	}
	for _, gate := range report.Gates {
		fmt.Fprintf(output, "result=%s commit=%s outcome=%s summed_step_work=%s finalizations=%d\n", gate.ID, gate.Result.CodeCommit, gate.Result.Outcome, time.Duration(gate.SummedStepNS), len(gate.Finalizations))
		if len(gate.Attempts) == 0 {
			fmt.Fprintln(output, "  attempt unavailable; outer wall unavailable")
		}
		for _, attempt := range gate.Attempts {
			fmt.Fprintf(output, "  attempt=%s step=%s/%s outer_wall=%s\n", attempt.ID, attempt.Record.PlanItem, attempt.Record.PlanStep, time.Duration(attempt.Record.WallNS))
			selection := attempt.Record.Selection
			if selection == (runrecord.AttemptSelection{}) {
				fmt.Fprintln(output, "  selection measurements unavailable")
			} else {
				fmt.Fprintf(output, "  selection defined=%d selected=%d excluded=%d uncertain=%d cache_hits=%d/%d planning=%s\n",
					selection.Defined, selection.Selected, selection.Excluded, selection.Uncertainty,
					selection.CacheHits, selection.CacheEligible, time.Duration(selection.PlanningNS))
			}
		}
		for _, step := range gate.Result.Steps {
			duration := "unmeasured"
			if step.DurationNS != 0 {
				duration = time.Duration(step.DurationNS).String()
			}
			fmt.Fprintf(output, "  %-24s %-12s %s\n", step.Name, step.Outcome, duration)
		}
		writeSelectionCauses(output, gate.SelectionCauses)
	}
	for _, preparation := range report.Unfinalized {
		fmt.Fprintf(output, "unfinalized=%s tree=%s started=%s commit=unbound\n", preparation.ID, preparation.TreeKey, preparation.Started)
	}
	fmt.Fprintln(output, report.Limitations)
	_, err := io.Copy(target, output)
	return err
}

// writeSelectionCauses prints the retained histogram: packages by primary
// cause, every sufficient cause, and the changed inputs that selected most.
func writeSelectionCauses(output io.Writer, histogram *runrecord.SelectionHistogram) {
	if histogram == nil {
		fmt.Fprintln(output, "  selection causes not retained for this gate")
		return
	}
	for _, step := range histogram.Steps {
		fmt.Fprintf(output, "  causes %-16s packages=%d executed=%d failed=%d skipped=%d unstarted=%d\n", step.Name, step.Packages, step.Executed, step.Failed, step.Skipped, step.Unstarted)
	}
	var unknown, missing, failed, passed, isolated int
	var isolatedSeconds float64
	for _, entry := range histogram.Packages {
		if entry.ShadowIsolated {
			isolated++
			if entry.ElapsedSeconds != nil {
				isolatedSeconds += *entry.ElapsedSeconds
			}
		}
		switch {
		case !entry.Reuse.Obligation.Valid():
			unknown++
		case !entry.Reuse.Receipt.Valid():
			missing++
		case entry.Reuse.Passed:
			passed++
		default:
			failed++
		}
	}
	fmt.Fprintf(output, "  receipts before execution: unrecorded=%d no_matching=%d prior_failed=%d prior_passed=%d\n", unknown, missing, failed, passed)
	fmt.Fprintf(output, "  shadow isolation, measured and not enforced: would leave out packages=%d of %d elapsed=%.1fs\n", isolated, len(histogram.Packages), isolatedSeconds)
	for _, bar := range histogram.Primary {
		fmt.Fprintf(output, "  primary %-14s packages=%d executed=%d skipped=%d unstarted=%d elapsed=%.1fs\n", bar.Cause, bar.Packages, bar.Executed, bar.Skipped, bar.Unstarted, bar.ElapsedSeconds)
	}
	for _, bar := range histogram.Sufficient {
		fmt.Fprintf(output, "  sufficient %-11s packages=%d\n", bar.Cause, bar.Packages)
	}
	for _, bar := range histogram.Inputs {
		fmt.Fprintf(output, "  input %s packages=%d executed=%d elapsed=%.1fs\n", bar.Cause, bar.Packages, bar.Executed, bar.ElapsedSeconds)
	}
	// An unnamed reach binds its reader to every root, so the readers that
	// select the most packages are the inputs most worth naming.
	selected, seconds := map[string]int{}, map[string]float64{}
	for _, entry := range histogram.Packages {
		for _, cause := range entry.Causes {
			if cause.Kind == runrecord.SelectionCauseReader {
				selected[cause.Detail]++
				seconds[cause.Detail] += *cmp.Or(entry.ElapsedSeconds, new(float64))
			}
		}
	}
	for _, reader := range slices.SortedFunc(maps.Keys(selected), func(left, right string) int {
		return cmp.Or(cmp.Compare(selected[right], selected[left]), strings.Compare(left, right))
	}) {
		fmt.Fprintf(output, "  reader packages=%d elapsed=%.1fs %s\n", selected[reader], seconds[reader], reader)
	}
	fmt.Fprintln(output, "  "+histogram.Limitations)
}
