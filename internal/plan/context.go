package plan

import (
	"errors"
	"path/filepath"
	"strings"

	"overgo/internal/repoanalysis"
)

const AutomationContextVersion = 2

// AutomationContext is the compact machine grounding for one automation turn.
// It intentionally exposes one current task; campaign strategy remains in the
// plan and does not compete with dispatch under another "rank-1" name.
type AutomationContext struct {
	SchemaVersion int             `json:"schema_version"`
	Head          string          `json:"head"`
	Branch        string          `json:"branch"`
	Worktree      string          `json:"worktree"`
	Role          string          `json:"role"`
	PlanState     string          `json:"plan_state"`
	CurrentTask   *TaskContext    `json:"current_task,omitempty"`
	Workflow      WorkflowContext `json:"workflow"`
	Dirty         []DirtyPath     `json:"dirty"`
	EvidenceDebt  EvidenceDebt    `json:"evidence_debt"`
}

type TaskContext struct {
	ItemID    string `json:"item_id"`
	ItemTitle string `json:"item_title"`
	StepID    string `json:"step_id"`
	StepTitle string `json:"step_title"`
	Verify    string `json:"verify,omitempty"`
}

type DirtyPath = repoanalysis.DirtyPath

// EvidenceDebt is deliberately conservative and derives from authoritative
// RepoDB prepared/finalized lifecycle records.
type EvidenceDebt struct {
	State    string `json:"state"`
	Source   string `json:"source"`
	Reason   string `json:"reason,omitempty"`
	ResultID string `json:"result_id,omitempty"`
}

// WorkflowContext is a Git-HEAD-derived implementation -> SQA -> priority
// guide. CurrentTask remains the sole executable work owner.
type WorkflowContext struct {
	Phase       string `json:"phase"`
	Source      string `json:"source"`
	Reason      string `json:"reason,omitempty"`
	CandidateID string `json:"candidate_id,omitempty"`
	VerdictID   string `json:"verdict_id,omitempty"`
}

type ContextFacts struct {
	Head         string
	Branch       string
	Worktree     string
	Role         string
	Dirty        []DirtyPath
	EvidenceDebt EvidenceDebt
	Workflow     WorkflowContext
}

func BuildAutomationContext(document Plan, facts ContextFacts) (AutomationContext, error) {
	if err := ValidateOpenWork(document); err != nil {
		return AutomationContext{}, err
	}
	facts.Head = strings.TrimSpace(facts.Head)
	facts.Branch = strings.TrimSpace(facts.Branch)
	facts.Worktree = filepath.ToSlash(strings.TrimSpace(facts.Worktree))
	facts.Role = strings.TrimSpace(facts.Role)
	if facts.Head == "" || facts.Branch == "" || facts.Worktree == "" {
		return AutomationContext{}, errors.New("automation context requires head, branch, and worktree")
	}
	if strings.ContainsAny(facts.Role, "\r\n") {
		return AutomationContext{}, errors.New("automation context role contains a newline")
	}
	if facts.Role == "" {
		facts.Role = "unassigned"
	}
	ctx := AutomationContext{
		SchemaVersion: AutomationContextVersion,
		Head:          facts.Head, Branch: facts.Branch, Worktree: facts.Worktree,
		Role: facts.Role, PlanState: "complete", Workflow: facts.Workflow,
		Dirty: repoanalysis.NormalizeDirty(facts.Dirty), EvidenceDebt: facts.EvidenceDebt,
	}
	if ctx.Dirty == nil {
		ctx.Dirty = []DirtyPath{}
	}
	if ctx.EvidenceDebt.State == "" || ctx.EvidenceDebt.Source == "" {
		return AutomationContext{}, errors.New("automation context requires evidence-debt state and source")
	}
	if ctx.Workflow.Source == "" || ctx.Workflow.Phase != "implementation" && ctx.Workflow.Phase != "sqa" && ctx.Workflow.Phase != "priority" {
		return AutomationContext{}, errors.New("automation context requires a valid workflow phase and source")
	}
	if item, step, ok := Current(document); ok {
		ctx.PlanState = "active"
		ctx.CurrentTask = &TaskContext{
			ItemID: item.ID, ItemTitle: item.Title,
			StepID: step.ID, StepTitle: step.Title, Verify: step.Verify,
		}
	}
	return ctx, nil
}
