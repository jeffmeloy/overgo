package plan

import (
	"errors"
	"fmt"
	"strings"
)

// MergeProjection is the canonical ownership rule for a two-parent plan
// transition. SemanticUnion is the backward-compatible default. The
// FirstParentTarget value must be selected explicitly and is persisted in the
// completion commit; callers must never infer it from plan similarity.
type MergeProjection string

const (
	// MergeProjectionSemanticUnion requires the child to retain both parent plans.
	MergeProjectionSemanticUnion MergeProjection = ""
	// MergeProjectionFirstParentTarget retains only the target lane's first-parent plan.
	MergeProjectionFirstParentTarget MergeProjection = "first-parent-target"
)

func (projection MergeProjection) validate() error {
	switch projection {
	case MergeProjectionSemanticUnion, MergeProjectionFirstParentTarget:
		return nil
	default:
		return fmt.Errorf("plan: invalid merge projection %q", projection)
	}
}

// ParseMergeProjection accepts the one explicit non-default projection used
// at command boundaries. Empty text selects the existing semantic-union
// behavior. Whitespace and aliases are refused so the commit trailer has one
// representation.
func ParseMergeProjection(value string) (MergeProjection, error) {
	if value == "" {
		return MergeProjectionSemanticUnion, nil
	}
	if value != strings.TrimSpace(value) {
		return MergeProjectionSemanticUnion, errors.New("plan: merge projection must be canonical")
	}
	projection := MergeProjection(value)
	if err := projection.validate(); err != nil {
		return MergeProjectionSemanticUnion, err
	}
	return projection, nil
}
