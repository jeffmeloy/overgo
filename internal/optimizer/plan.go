package optimizer

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
)

const planIdentityDomain = "overgo.optimizer.plan.v2"

// GroupSpec: named flat parameter matrix.
type GroupSpec struct {
	Name   string
	Start  int
	End    int
	Rows   int
	Cols   int
	Frozen bool
}

// Group: validated Muon group.
type Group struct{ GroupSpec }

// Plan: immutable flat parameter layout.
type Plan struct {
	parameterCount int
	groups         []Group
	identity       string
	maxMatrix      int
	maxSquare      int
}

// CompilePlan validates a complete, ordered parameter partition.
func CompilePlan(parameterCount int, specs []GroupSpec) (Plan, error) {
	if parameterCount < 0 {
		return Plan{}, errors.New("optimizer plan: negative parameter count")
	}
	if parameterCount == 0 {
		if len(specs) != 0 {
			return Plan{}, errors.New("optimizer plan: non-empty groups for empty parameters")
		}
		return newPlan(0, nil), nil
	}
	if len(specs) == 0 {
		return Plan{}, errors.New("optimizer plan: parameter groups are required")
	}

	groups := make([]Group, len(specs))
	names := make(map[string]struct{}, len(specs))
	next := 0
	for index, spec := range specs {
		if spec.Name == "" {
			return Plan{}, fmt.Errorf("optimizer plan: group %d has no name", index)
		}
		if _, exists := names[spec.Name]; exists {
			return Plan{}, fmt.Errorf("optimizer plan: duplicate group %q", spec.Name)
		}
		if spec.Start != next || spec.End < spec.Start || spec.End > parameterCount {
			return Plan{}, fmt.Errorf("optimizer plan: group %q range [%d,%d) does not continue at %d", spec.Name, spec.Start, spec.End, next)
		}
		if spec.Rows <= 0 || spec.Cols <= 0 || spec.Rows > int(^uint(0)>>1)/spec.Cols || spec.Rows*spec.Cols != spec.End-spec.Start {
			return Plan{}, fmt.Errorf("optimizer plan: group %q shape %dx%d does not match range [%d,%d)", spec.Name, spec.Rows, spec.Cols, spec.Start, spec.End)
		}
		groups[index] = Group{GroupSpec: spec}
		names[spec.Name] = struct{}{}
		next = spec.End
	}
	if next != parameterCount {
		return Plan{}, fmt.Errorf("optimizer plan: groups cover %d of %d parameters", next, parameterCount)
	}
	return newPlan(parameterCount, groups), nil
}

func newPlan(parameterCount int, groups []Group) Plan {
	plan := Plan{parameterCount: parameterCount, groups: groups}
	for _, group := range groups {
		if group.Frozen {
			continue
		}
		matrix := group.End - group.Start
		square := min(group.Rows, group.Cols)
		square *= square
		plan.maxMatrix = max(plan.maxMatrix, matrix)
		plan.maxSquare = max(plan.maxSquare, square)
	}
	plan.identity = planIdentity(plan)
	return plan
}

func (p Plan) ParameterCount() int { return p.parameterCount }
func (p Plan) GroupCount() int     { return len(p.groups) }
func (p Plan) Identity() string    { return p.identity }

func (p Plan) Group(index int) (Group, bool) {
	if index < 0 || index >= len(p.groups) {
		return Group{}, false
	}
	return p.groups[index], true
}

func planIdentity(plan Plan) string {
	digest := sha256.New()
	digest.Write([]byte(planIdentityDomain))
	writePlanUint(digest, uint64(plan.parameterCount))
	writePlanUint(digest, uint64(len(plan.groups)))
	for _, group := range plan.groups {
		writePlanUint(digest, uint64(len(group.Name)))
		digest.Write([]byte(group.Name))
		writePlanUint(digest, uint64(group.Start))
		writePlanUint(digest, uint64(group.End))
		writePlanUint(digest, uint64(group.Rows))
		writePlanUint(digest, uint64(group.Cols))
		if group.Frozen {
			writePlanUint(digest, 1)
		} else {
			writePlanUint(digest, 0)
		}
	}
	return fmt.Sprintf("%x", digest.Sum(nil))
}

func writePlanUint(dst hash.Hash, value uint64) {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], value)
	dst.Write(encoded[:])
}
