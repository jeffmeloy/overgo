package plan

import (
	"context"
	"maps"
)

// transitionPlans retains only immutable Git facts within one resolution.
// Nested first-parent proofs share ancestors; reading and parsing those plans
// again adds no evidence. Store-dependent completion evidence remains separate
// and bound to the exact store head; projected merge receipts still revalidate.
func (resolver *completionAuthorityResolver) transitionPlans(ctx context.Context, repository string, commits []gitCompletionMessage) (map[string]completionTransition, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if resolver.transitions == nil {
		resolver.transitions = make(map[string]map[string]completionTransition)
	}
	known := resolver.transitions[repository]
	if known == nil {
		known = make(map[string]completionTransition)
		resolver.transitions[repository] = known
	}
	var missing []gitCompletionMessage
	for _, commit := range commits {
		if _, found := known[commit.hash]; !found {
			missing = append(missing, commit)
		}
	}
	if len(missing) != 0 {
		loaded, err := completionTransitionPlans(ctx, repository, missing)
		if err != nil {
			return nil, err
		}
		maps.Copy(known, loaded)
		resolver.transitionLoads += len(loaded)
	}
	return known, nil
}
