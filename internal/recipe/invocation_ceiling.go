package recipe

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

// RequireInvocationCeiling proves that one stored task or delegated
// invocation grants an exact compiled capability, effect, and scope. It is a
// ceiling only; a mutation still needs inspection and action-bound authority.
func RequireInvocationCeiling(
	ctx context.Context,
	reader artifact.Reader,
	ceiling, manual artifact.ID,
	effect, scope string,
) error {
	if ctx == nil || reader == nil || ceiling.Kind() != artifact.KindRecipe ||
		manual.Kind() != artifact.KindRecipe || effect == "" || scope == "" {
		return errors.New("recipe: invocation ceiling authority is absent")
	}
	task, taskErr := agentTaskContractCodec.RequireExactLineage(
		ctx, reader, ceiling, AgentTaskContract.Lineage,
	)
	if taskErr == nil {
		return requireTaskInvocation(task, effect, scope)
	}
	delegated, delegatedErr := delegatedAgentInvocationCodec.RequireExactLineage(
		ctx, reader, ceiling, DelegatedAgentInvocation.Lineage,
	)
	if delegatedErr != nil {
		return errors.Join(errors.New("recipe: invocation ceiling is not a task or delegation"), taskErr, delegatedErr)
	}
	if delegated.Grant.ReadOnly || !slices.Contains(delegated.Grant.Manuals, manual) ||
		!slices.Contains(delegated.Grant.AllowedEffects, effect) {
		return errors.New("recipe: delegation does not grant the invocation")
	}
	parent, err := agentTaskContractCodec.RequireExactLineage(
		ctx, reader, delegated.Task, AgentTaskContract.Lineage,
	)
	if err != nil {
		return err
	}
	return requireTaskInvocation(parent, effect, scope)
}

func requireTaskInvocation(task AgentTaskContract, effect, scope string) error {
	if !slices.Contains(task.AllowedEffects, effect) || !slices.Contains(task.Scope, scope) {
		return errors.New("recipe: task does not grant the invocation")
	}
	return nil
}
