package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

// rowDiagnosis: one row's observed state and the facts the next action keys on.
type rowDiagnosis struct {
	state    string
	waits    []string
	attempts []runrecord.AttemptRecord
}

// diagnoseRow renders one row's state, its attempt records in order, the
// bounded evidence tails of the latest failure, and the next action keyed
// on the observed state.
func diagnoseRow(root, reference string, output io.Writer) error {
	item, step, bound := strings.Cut(reference, "/")
	if !bound || item == "" || step == "" {
		return errors.New("usage: plan -diagnose <item>/<step>")
	}
	document, err := plan.Load(filepath.Join(root, filepath.FromSlash(plan.Path)))
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	authority, err := plan.ResolveCompletionAuthority(ctx, root, "HEAD", document, store)
	if err != nil {
		return err
	}
	diagnosis, err := observeRow(document, authority, item, step)
	if err != nil {
		return err
	}
	history, err := runrecord.LoadAttemptHistory(ctx, store, runrecord.AttemptFilter{PlanItem: item})
	if err != nil {
		return err
	}
	for _, attempt := range history.Attempts {
		if attempt.PlanStep == step {
			diagnosis.attempts = append(diagnosis.attempts, attempt)
		}
	}
	fmt.Fprintf(output, "row %s: state=%s\n", reference, diagnosis.state)
	for _, wait := range diagnosis.waits {
		fmt.Fprintf(output, "waits on %s\n", wait)
	}
	fmt.Fprintf(output, "attempts: %d\n", len(diagnosis.attempts))
	var failed []string
	for index, attempt := range diagnosis.attempts {
		outcome := string(attempt.Outcome)
		if attempt.Failure != "" {
			outcome += ":" + attempt.Failure
		}
		fmt.Fprintf(output, "#%d %s wall=%s commit=%.8s\n", index+1, outcome,
			time.Duration(attempt.WallNS).Round(time.Millisecond), attempt.CodeCommit)
	}
	if last := len(diagnosis.attempts); last != 0 && diagnosis.attempts[last-1].Outcome != runrecord.OutcomeSucceeded {
		failed, err = printFailureEvidence(ctx, store, diagnosis.attempts[last-1], output)
		if err != nil {
			return err
		}
	}
	fmt.Fprintf(output, "next: %s\n", nextActionFor(reference, diagnosis, failed))
	return nil
}

// observeRow: the row's retained status, its completion evidence, and
// whether the frontier dispatches it or holds it.
func observeRow(document plan.Plan, authority plan.CompletionAuthority, item, step string) (rowDiagnosis, error) {
	reference := item + "/" + step
	var retained *plan.Step
	for _, candidate := range document.Items {
		if candidate.ID != item {
			continue
		}
		for index := range candidate.Steps {
			if candidate.Steps[index].ID == step {
				retained = &candidate.Steps[index]
			}
		}
	}
	if retained == nil {
		if _, completed := authority.CompletedOutcomes()[reference]; completed {
			return rowDiagnosis{state: "completed"}, nil
		}
		frontier, err := plan.ReadyFrontier(document, authority)
		if err != nil {
			return rowDiagnosis{}, err
		}
		if slicesContainsRef(frontier, reference) {
			return rowDiagnosis{state: "completed"}, nil
		}
		return rowDiagnosis{state: "absent"}, nil
	}
	if retained.Refusal != nil {
		return rowDiagnosis{state: fmt.Sprintf("%s: %s", retained.Status, retained.Refusal.Reason)}, nil
	}
	if retained.Status != plan.StatusOpen {
		return rowDiagnosis{state: retained.Status}, nil
	}
	frontier, err := plan.ReadyFrontier(document, authority)
	if err != nil {
		return rowDiagnosis{}, err
	}
	if !slicesContainsRef(frontier, reference) {
		diagnosis := rowDiagnosis{state: "open-blocked"}
		for _, blocked := range plan.BlockedRows(document, frontier, authority) {
			if blocked.Ref.String() == reference {
				diagnosis.waits = blocked.Waits
			}
		}
		return diagnosis, nil
	}
	dispositions, err := plan.Dispositions(document, frontier, plan.DocumentConditionFacts(document, authority))
	if err != nil {
		return rowDiagnosis{}, err
	}
	for _, row := range dispositions {
		if row.Ref.String() == reference && row.Disposition != plan.DispositionProceed {
			return rowDiagnosis{state: "open-" + string(row.Disposition), waits: []string{row.Reason}}, nil
		}
	}
	return rowDiagnosis{state: "open-ready"}, nil
}

func slicesContainsRef(frontier []plan.Ref, reference string) bool {
	for _, ref := range frontier {
		if ref.String() == reference {
			return true
		}
	}
	return false
}

// printFailureEvidence: the failed steps of the attempt's gate result with
// bounded evidence tails; returns the failed step names.
func printFailureEvidence(ctx context.Context, store *overgodb.Store, attempt runrecord.AttemptRecord, output io.Writer) ([]string, error) {
	if !attempt.Result.Valid() {
		return nil, nil
	}
	result, err := runrecord.RequireGateResult(ctx, store, attempt.Result)
	if err != nil {
		return nil, err
	}
	var failed []string
	for _, step := range result.Steps {
		if step.Outcome != runrecord.StepFailed {
			continue
		}
		failed = append(failed, step.Name)
		fmt.Fprintf(output, "failed step %s (%s):\n%s\n", step.Name, step.Phase, clioptions.Tail(step.Evidence, clioptions.DiagnosticTailBytes))
	}
	if result.Failure != "" {
		fmt.Fprintf(output, "gate failure: %s\n", result.Failure)
	}
	return failed, nil
}

// nextActionFor keys the operator's next action on the observed state.
func nextActionFor(reference string, diagnosis rowDiagnosis, failed []string) string {
	switch {
	case diagnosis.state == "completed":
		return "the row completed through the gate; nothing remains"
	case diagnosis.state == "absent":
		return "the row is neither retained nor completed; check the reference or add it with plan -add"
	case strings.HasPrefix(diagnosis.state, plan.StatusSkipped) || strings.HasPrefix(diagnosis.state, plan.StatusCancelled):
		return "the row was refused; dependents that accept the refusal proceed, others stay held until the work is re-added"
	case diagnosis.state == "open-blocked":
		return "work the rows it waits on first: go run ./cmd/plan -next"
	case strings.HasPrefix(diagnosis.state, "open-") && diagnosis.state != "open-ready":
		return "resolve the condition it waits on, or refuse the row: go run ./cmd/plan -refuse " + reference + " -disposition skip -reason <text>"
	case len(diagnosis.attempts) == 0:
		return "dispatch it: go run ./cmd/plan -next, implement, go run ./cmd/plan -verify, then gate"
	case len(failed) != 0:
		return "fix the failed step(s) " + strings.Join(failed, ",") + " and gate again: go run ./cmd/gate -plan " + reference
	case diagnosis.attempts[len(diagnosis.attempts)-1].Outcome != runrecord.OutcomeSucceeded:
		return "the last attempt did not succeed and its gate result names no failed step; read the gate failure above and gate again"
	default:
		return "the last attempt succeeded but the row is still open; run go run ./cmd/plan -verify and gate again"
	}
}
