package plan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	AutomationContextVersion = 1
	gateStatusMirrorPath     = "bin/gate_status.json"
)

// AutomationContext is the compact machine grounding for one automation turn.
// It intentionally exposes one current task; campaign strategy remains in the
// plan and does not compete with dispatch under another "rank-1" name.
type AutomationContext struct {
	SchemaVersion int          `json:"schema_version"`
	Head          string       `json:"head"`
	Branch        string       `json:"branch"`
	Worktree      string       `json:"worktree"`
	Role          string       `json:"role"`
	PlanState     string       `json:"plan_state"`
	CurrentTask   *TaskContext `json:"current_task,omitempty"`
	Dirty         []DirtyPath  `json:"dirty"`
	EvidenceDebt  EvidenceDebt `json:"evidence_debt"`
}

type TaskContext struct {
	ItemID    string `json:"item_id"`
	ItemTitle string `json:"item_title"`
	StepID    string `json:"step_id"`
	StepTitle string `json:"step_title"`
	Verify    string `json:"verify,omitempty"`
}

type DirtyPath struct {
	Path           string `json:"path"`
	IndexStatus    string `json:"index_status"`
	WorktreeStatus string `json:"worktree_status"`
	OriginalPath   string `json:"original_path,omitempty"`
}

// EvidenceDebt is deliberately conservative in context version 1. The gate
// status file is an advisory mirror, so anything except an exact successful
// HEAD match is possible debt, never proof that authoritative RepoDB evidence
// is absent. The gate-lifecycle plan slice will replace this mirror inference
// with authoritative prepared/finalized debt records.
type EvidenceDebt struct {
	State          string `json:"state"`
	Source         string `json:"source"`
	Reason         string `json:"reason,omitempty"`
	ObservedCommit string `json:"observed_commit,omitempty"`
	ResultID       string `json:"result_id,omitempty"`
	Outcome        string `json:"outcome,omitempty"`
}

type ContextFacts struct {
	Head         string
	Branch       string
	Worktree     string
	Role         string
	Dirty        []DirtyPath
	EvidenceDebt EvidenceDebt
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
		Role: facts.Role, PlanState: "complete",
		Dirty: normalizeDirty(facts.Dirty), EvidenceDebt: facts.EvidenceDebt,
	}
	if ctx.Dirty == nil {
		ctx.Dirty = []DirtyPath{}
	}
	if ctx.EvidenceDebt.State == "" || ctx.EvidenceDebt.Source == "" {
		return AutomationContext{}, errors.New("automation context requires evidence-debt state and source")
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

func normalizeDirty(paths []DirtyPath) []DirtyPath {
	out := make([]DirtyPath, 0, len(paths))
	for _, path := range paths {
		path.Path = filepath.ToSlash(strings.TrimSpace(path.Path))
		path.OriginalPath = filepath.ToSlash(strings.TrimSpace(path.OriginalPath))
		if path.Path == "" {
			continue
		}
		out = append(out, path)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].OriginalPath < out[j].OriginalPath
	})
	return out
}

// ParseDirtyStatus parses `git status --porcelain=v1 -z`. The -z form keeps
// filenames literal; rename/copy records carry the original path as the next
// NUL-delimited field.
func ParseDirtyStatus(raw []byte) ([]DirtyPath, error) {
	fields := strings.Split(string(raw), "\x00")
	var paths []DirtyPath
	for i := 0; i < len(fields); i++ {
		entry := fields[i]
		if entry == "" {
			continue
		}
		if len(entry) < 4 || entry[2] != ' ' {
			return nil, fmt.Errorf("malformed porcelain status entry %q", entry)
		}
		path := DirtyPath{
			IndexStatus: string(entry[0]), WorktreeStatus: string(entry[1]),
			Path: entry[3:],
		}
		if strings.ContainsAny(entry[:2], "RC") {
			i++
			if i >= len(fields) || fields[i] == "" {
				return nil, fmt.Errorf("rename/copy status for %q lacks original path", path.Path)
			}
			path.OriginalPath = fields[i]
		}
		paths = append(paths, path)
	}
	return normalizeDirty(paths), nil
}

type gateStatusMirror struct {
	ResultID   string `json:"result_id"`
	CodeCommit string `json:"code_commit"`
	Outcome    string `json:"outcome"`
}

// EvidenceDebtFromMirror classifies only the cheap advisory mirror. It cannot
// claim authoritative debt because a RepoDB write may exist when its mirror is
// missing or stale.
func EvidenceDebtFromMirror(head string, raw []byte, readErr error) EvidenceDebt {
	debt := EvidenceDebt{State: "possible", Source: gateStatusMirrorPath}
	if readErr != nil {
		if errors.Is(readErr, os.ErrNotExist) {
			debt.Reason = "advisory gate status mirror is absent; authoritative RepoDB was not inspected"
		} else {
			debt.Reason = "advisory gate status mirror is unreadable: " + readErr.Error()
		}
		return debt
	}
	var mirror gateStatusMirror
	if err := json.Unmarshal(raw, &mirror); err != nil {
		debt.Reason = "advisory gate status mirror is invalid JSON: " + err.Error()
		return debt
	}
	debt.ObservedCommit = strings.TrimSpace(mirror.CodeCommit)
	debt.ResultID = strings.TrimSpace(mirror.ResultID)
	debt.Outcome = strings.TrimSpace(mirror.Outcome)
	switch {
	case debt.ObservedCommit != strings.TrimSpace(head):
		debt.Reason = "advisory gate status mirror names a different commit"
	case debt.Outcome != "succeeded":
		debt.Reason = "advisory gate status mirror does not report success"
	case debt.ResultID == "":
		debt.Reason = "advisory gate status mirror lacks a result identity"
	default:
		debt.State = "none_observed"
	}
	return debt
}
