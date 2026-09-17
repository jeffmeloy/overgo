package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// Called only inside withPlanMutation's process-owned critical section.
func savePlanMutation(root string, document plan.Plan) error {
	path := filepath.Join(root, filepath.FromSlash(plan.Path))
	before, err := plan.Load(path)
	if err != nil {
		return err
	}
	return saveCapturedPlanMutation(root, before, document)
}

// Called only while holding withPlanMutation's lock, with before captured under
// that same lock. Merge preparation can replace the live text with conflicts;
// its claim authority remains the captured pre-merge document.
func saveCapturedPlanMutation(root string, before, document plan.Plan) error {
	store, err := overgodb.OpenReadOnly(filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return err
	}
	defer store.Close()
	if err := plan.ValidateClaimedPlan(context.Background(), store, before, document); err != nil {
		return err
	}
	return plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document)
}

func releaseDispatchClaim(root, rawID, worker, reason string, output io.Writer) (err error) {
	if reason != "cancelled" && reason != "handoff" {
		return errors.New("plan: -release-claim requires -release-reason cancelled or handoff; completion belongs to the gate")
	}
	id, err := artifact.ParseID(rawID)
	if err != nil {
		return err
	}
	lock, err := authoritylock.Acquire(root)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	store, err := overgodb.Open(filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, store.Close()) }()
	lease, found, err := worklease.Read(context.Background(), store, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("plan: claim %s is unavailable", id)
	}
	worktree, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	if !strings.EqualFold(filepath.ToSlash(worktree), lease.Worktree) {
		return errors.New("plan: release from the claimed worktree so its gate and plan mutation lock protects the handoff")
	}
	batch, err := lease.ReleaseBatch(worker, reason)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "claim=%s task=%s state=released reason=%s; evidence and outstanding checks retained\n", id, lease.Task, reason)
	return err
}

// Stop controls remain usable when the live plan cannot load. They publish
// through the control owner without waiting for a running gate's mutation lock.
func recordStopControlCommand(root string, c cli, args []string, output io.Writer) error {
	role, err := plan.AutomationRole(c.role)
	if err != nil {
		return err
	}
	worker := cmp.Or(strings.TrimSpace(c.worker), strings.TrimSpace(os.Getenv(plan.AutomationWorkerEnvironment)))
	if worker == "" {
		return errors.New("plan: stop control requires -worker or OVERGO_AUTOMATION_WORKER")
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return err
	}
	head, err := gitOutput(root, "rev-parse", "HEAD")
	if err != nil {
		return err
	}
	store, err := overgodb.Open(filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return err
	}
	defer store.Close()
	current, err := plan.ReadStop(context.Background(), store, root, strings.TrimSpace(string(head)), plan.ExecutionInteractive)
	if err != nil {
		return err
	}
	event := plan.ControlEvent{Kind: plan.ControlStop, Lane: role, Worker: worker, Worktree: filepath.ToSlash(root), Mode: cmp.Or(c.stopMode, plan.ExecutionAll), CodeCommit: strings.TrimSpace(string(head)), Previous: current.ID}
	if c.stop {
		reason := strings.TrimSpace(strings.Join(args, commandWordSeparator))
		code, detail, err := plan.ParseStopReason(reason)
		if err != nil {
			return err
		}
		event.ReasonCode = strings.TrimSpace(code)
		event.Detail = cmp.Or(strings.TrimSpace(detail), event.ReasonCode)
	} else {
		expected, err := artifact.ParseID(cmp.Or(c.resumeStop, c.maintenanceStop))
		if err != nil {
			return err
		}
		if expected != current.ID || current.Event == nil {
			return errors.New("plan: requested stop is not the current scoped authority")
		}
		event.Mode = current.Event.Mode
		if c.stopMode != "" && c.stopMode != event.Mode {
			return errors.New("plan: resume cannot change stop scope")
		}
		event.Kind = plan.ControlResume
		event.ReasonCode = "operator-resume"
		if c.maintenanceStop != "" {
			if len(args) == 0 {
				return errors.New("plan: maintenance requires item/step and an operator reason")
			}
			event.Kind = plan.ControlMaintenance
			event.ReasonCode = "operator-maintenance"
			event.Task = args[0]
			args = args[1:]
		}
		event.Detail = strings.TrimSpace(strings.Join(args, commandWordSeparator))
		if event.Detail == "" {
			return errors.New("plan: explicit resume requires an operator reason")
		}
	}
	recorded, err := plan.RecordControlEvent(context.Background(), store, event)
	if err != nil {
		return err
	}
	if c.json {
		return json.NewEncoder(output).Encode(struct {
			ID    artifact.ID       `json:"id"`
			Event plan.ControlEvent `json:"event"`
		}{recorded.ID, recorded})
	}
	fmt.Fprintf(output, "recorded %s %s owner=%s scope=%s worktree=%s\n", recorded.Kind, recorded.ID, recorded.Lane, recorded.Mode, recorded.Worktree)
	if recorded.Kind == plan.ControlMaintenance {
		fmt.Fprintf(output, "For task %s only: inherit %s=%s with worker=%s; unattended work remains stopped.\n", recorded.Task, plan.AutomationMaintenanceEnvironment, recorded.ID, recorded.Worker)
	}
	return nil
}

// Inspection and executable prompts share one typed dispatch and stop projection.
func printDispatch(c cli, args []string, output io.Writer) error {
	if len(args) > 1 || len(args) != 0 && !c.prompt {
		return errors.New("plan: only -prompt accepts one explicit item/step")
	}
	reference := ""
	if len(args) == 1 {
		reference = args[0]
	}
	document, err := loadCommandPlan(false)
	var dispatch plan.Dispatch
	if err != nil {
		stop, stopErr := plan.ReadStop(context.Background(), nil, commandWorktree, "", c.mode)
		if stopErr != nil || !stop.Blocked {
			return err
		}
		dispatch = plan.Dispatch{Stop: &stop, Waiting: stop.String(), Line: "plan waiting: " + stop.String()}
		err = nil
	} else {
		dispatch, err = plan.ResolveDispatch(context.Background(), commandWorktree, plan.DispatchRequest{Role: c.role, Worker: c.worker, Mode: c.mode, Acquire: c.prompt || c.verify, Reference: reference})
	}
	if err != nil {
		return err
	}
	if c.json && !c.verify {
		return json.NewEncoder(output).Encode(dispatch)
	}
	if c.next || dispatch.Waiting != "" || dispatch.Complete {
		fmt.Fprintln(output, dispatch.Line)
		if c.next && dispatch.Stop != nil && dispatch.Waiting == "" {
			fmt.Fprintln(output, dispatch.Stop.String())
		}
		return nil
	}
	document, err = plan.Load("")
	if err != nil {
		return err
	}
	it, st, found := locateStep(document, dispatch)
	if !found {
		return errors.New("plan: claimed task changed before prompt projection")
	}
	if c.verify {
		return runVerify(it, st)
	}
	if dispatch.Claim != nil {
		fmt.Fprintf(output, "Claim: %s worker=%s worktree=%s\n", dispatch.Claim.ID, dispatch.Claim.Worker, dispatch.Claim.Worktree)
	}
	if dispatch.Stop != nil {
		fmt.Fprintln(output, dispatch.Stop.String())
	}
	printPromptStep(it, st, output)
	return nil
}

// CLI argument prose separates already-parsed words with one space.
const commandWordSeparator = " "

// editPlanStep shares mutation ownership, dependency authority and atomic Save.
func editPlanStep(root, path string, output io.Writer) error {
	var edit plan.StepEdit
	if err := jsonfile.DecodeStrict(path, &edit); err != nil {
		return err
	}
	return withPlanMutation(root, false, func(document plan.Plan) error {
		updated, err := document.EditStep(edit)
		if err != nil {
			return err
		}
		if _, err := resolveCompletionAuthority(root, updated); err != nil {
			return err
		}
		var prior plan.Step
		for _, item := range document.Items {
			if item.ID == edit.Item {
				for _, step := range item.Steps {
					if step.ID == edit.Step.ID {
						prior = step
					}
				}
			}
		}
		if browserVerifyOutsideLane(edit.Step.Verify) {
			return errors.New("plan: browser acceptance requires the existing webui lane")
		}
		previousAcceptance, err := json.Marshal(struct {
			Verify string
			Batch  *plan.VerificationBatch
		}{prior.Verify, prior.VerificationBatch})
		if err != nil {
			return err
		}
		nextAcceptance, err := json.Marshal(struct {
			Verify string
			Batch  *plan.VerificationBatch
		}{edit.Step.Verify, edit.Step.VerificationBatch})
		if err != nil {
			return err
		}
		if err := savePlanMutation(root, updated); err != nil {
			return err
		}
		return json.NewEncoder(output).Encode(struct {
			Ref                 plan.Ref `json:"ref"`
			Before              string   `json:"before"`
			After               string   `json:"after"`
			Created             bool     `json:"created"`
			AcceptanceChanged   bool     `json:"acceptance_changed"`
			PublicationRequired bool     `json:"publication_required"`
		}{plan.Ref{Item: edit.Item, Step: edit.Step.ID}, document.Digest(), updated.Digest(), edit.Create, string(previousAcceptance) != string(nextAcceptance), true})
	})
}
