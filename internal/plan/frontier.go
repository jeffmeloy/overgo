package plan

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// Ref names one plan row as an exact item/step identity.
type Ref struct {
	Item string `json:"item"`
	Step string `json:"step"`
}

// String renders the canonical item/step reference.
func (r Ref) String() string { return r.Item + "/" + r.Step }

// ReadyFrontier returns every dispatchable open step in deterministic
// document order: each open step whose dependencies are all proven by gated
// completion evidence or retained done state. Current remains the
// single-lane compatibility selector; its selection is always a member of
// this frontier, so a single-lane driver and a leased multi-lane driver can
// never disagree about what is dispatchable.
func ReadyFrontier(d Plan, authority CompletionAuthority) ([]Ref, error) {
	if !authority.resolves(d) {
		return nil, errors.New("plan: ready frontier requires a resolved completion authority for this exact plan")
	}
	frontier := []Ref{}
	for _, item := range d.Items {
		if item.Status != StatusOpen {
			continue
		}
		for _, step := range item.Steps {
			if step.Status != StatusOpen {
				continue
			}
			if dependenciesSatisfied(d, step, authority) {
				frontier = append(frontier, Ref{Item: item.ID, Step: step.ID})
			}
		}
	}
	return frontier, nil
}

// ValidateFrontierLeases requires distinct ready rows and isolated ownership.
// Worker claims allow shared roles across worktrees. Legacy advisory leases
// retain their conservative role and repository-relative overlap checks.
func ValidateFrontierLeases(frontier []Ref, leases []WorkLease) error {
	ready := make(map[string]bool, len(frontier))
	for _, ref := range frontier {
		ready[ref.String()] = true
	}
	taskOwners := make(map[string]string, len(leases))
	roleOwners := make(map[string]string, len(leases))
	for index, lease := range leases {
		if err := lease.ValidateIdentity(); err != nil {
			return fmt.Errorf("plan: frontier lease %q: %w", lease.Worktree, err)
		}
		if !ready[lease.Task] {
			return fmt.Errorf("plan: lease %q claims %q outside the ready frontier", lease.Worktree, lease.Task)
		}
		if owner, taken := taskOwners[lease.Task]; taken {
			return fmt.Errorf("plan: leases %q and %q both own frontier row %q", owner, lease.Worktree, lease.Task)
		}
		taskOwners[lease.Task] = lease.Worktree
		role := normalizedRole(lease.Role)
		if lease.Worker != "" {
			role = "worker:" + lease.Worker
		} else {
			role = "role:" + role
		}
		if owner, taken := roleOwners[role]; taken {
			return fmt.Errorf("plan: role %q holds leases %q and %q; one role owns one row at a time", role, owner, lease.Worktree)
		}
		roleOwners[role] = lease.Worktree
		for _, other := range leases[:index] {
			overlap := frontierClaimsOverlap(lease.Claims, other.Claims)
			if lease.Worker != "" && other.Worker != "" {
				overlap = WorkspaceClaimsConflict(lease, other)
			}
			if overlap {
				return fmt.Errorf(
					"plan: leases %q and %q hold overlapping workspace claims", other.Worktree, lease.Worktree,
				)
			}
		}
	}
	return nil
}

// frontierClaimsOverlap compares repo-relative claims across lanes: distinct
// worktrees still collide when they claim the same relative surface, because
// their gated commits meet again at merge time.
func frontierClaimsOverlap(left, right WorkspaceClaims) bool {
	if left.WholeWorktree || right.WholeWorktree {
		return true
	}
	for _, write := range left.Write {
		for _, path := range slices.Concat(right.Read, right.Write) {
			if claimPathsOverlap(write, path) {
				return true
			}
		}
	}
	for _, write := range right.Write {
		for _, read := range left.Read {
			if claimPathsOverlap(write, read) {
				return true
			}
		}
	}
	return false
}

// FormatFrontier renders one row per line for operator and driver consumption.
func FormatFrontier(frontier []Ref) string {
	lines := make([]string, 0, len(frontier))
	for _, ref := range frontier {
		lines = append(lines, ref.String())
	}
	return strings.Join(lines, "\n")
}
