package plan

import (
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

const AutomationContextVersion = 1

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

// EvidenceDebt is deliberately conservative and derives from authoritative
// RepoDB prepared/finalized lifecycle records.
type EvidenceDebt struct {
	State    string `json:"state"`
	Source   string `json:"source"`
	Reason   string `json:"reason,omitempty"`
	ResultID string `json:"result_id,omitempty"`
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
