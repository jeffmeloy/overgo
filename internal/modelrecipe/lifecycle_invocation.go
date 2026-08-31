package modelrecipe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/invocation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func verifiedTransition(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	evidence []artifact.ID,
	terminal runrecord.StageReceipt,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	if ctx == nil || store == nil || len(evidence) != 1 {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: supervised verification requires one typed decision")
	}
	stored, err := recipe.RequireDefinition(ctx, store, definition.ID)
	if err != nil || definition.ValidateIdentity() != nil || stored.ID != definition.ID {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.Join(errors.New("model recipe: verified recipe authority differs"), err)
	}
	decision, err := requireVerificationDecision(ctx, store, definition.ID, evidence[0])
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	if err := requireVerificationInvocation(ctx, store, definition, decision, terminal); err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	completed, prepared, err := runrecord.PrepareStageReceipt(ctx, store, terminal, nil, nil)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	transitionEvidence := append([]artifact.ID{decision.ID}, decision.Evidence...)
	transitionEvidence = append(transitionEvidence, completed.ID)
	return transition(
		ctx, store, key, definition, recipe.StatusVerified, transitionEvidence, nil,
		nil, nil, nil, &prepared,
	)
}

func requireVerificationDecision(
	ctx context.Context,
	reader artifact.Reader,
	subject, id artifact.ID,
) (recipe.Decision, error) {
	decision, err := recipe.RequireDecision(ctx, reader, id)
	if err != nil {
		return recipe.Decision{}, err
	}
	if decision.Subject != subject || decision.Outcome != recipe.DecisionAccepted ||
		decision.Tier == recipe.EvidenceExperimental || !decision.Tier.Valid() ||
		decision.Reason == "" || len(decision.Evidence) == 0 {
		return recipe.Decision{}, errors.New("model recipe: verification decision does not accept exact evaluated recipe")
	}
	stored, err := reader.Parents(ctx, decision.ID)
	if err != nil {
		return recipe.Decision{}, err
	}
	expected := decision.Lineage()
	if len(stored) != len(expected) || slices.ContainsFunc(expected, func(edge artifact.Lineage) bool {
		return !slices.Contains(stored, edge)
	}) {
		return recipe.Decision{}, errors.New("model recipe: verification decision lineage differs")
	}
	for _, parent := range decision.Evidence {
		if _, err := artifact.RequireTypedContent(ctx, reader, parent); err != nil {
			return recipe.Decision{}, errors.Join(errors.New("model recipe: verification evidence is not typed and immutable"), err)
		}
	}
	return decision, nil
}

func requireVerificationInvocation(
	ctx context.Context,
	reader artifact.Reader,
	definition recipe.Definition,
	decision recipe.Decision,
	terminal runrecord.StageReceipt,
) error {
	if terminal.Invocation == nil {
		return errors.New("model recipe: supervised verification lacks invocation receipt")
	}
	binding := *terminal.Invocation
	target := statusAlias(definition.ID)
	if err := binding.Validate(); err != nil || binding.Boundary != invocation.BoundaryInternal ||
		binding.Kind != invocation.MutationPromotion || binding.Action != string(binding.Kind) ||
		binding.Subject != definition.ID || terminal.Recipe != binding.Ceiling ||
		terminal.Node != recipe.NodeID(binding.Kind) || terminal.State != runrecord.StageCompleted ||
		terminal.Previous.Valid() || !stageReceiptCites(terminal, decision.ID) {
		return errors.Join(errors.New("model recipe: supervised verification receipt differs"), err)
	}
	arguments, err := verificationArguments(definition.ID, decision.ID, target)
	if err != nil || arguments.Descriptor.ID != binding.Arguments {
		return errors.Join(errors.New("model recipe: supervised verification arguments differ"), err)
	}
	storedArguments, found, err := artifact.ReadContent(ctx, reader, binding.Arguments)
	if err != nil || !found || storedArguments.Descriptor != arguments.Descriptor ||
		!bytes.Equal(storedArguments.Data, arguments.Data) {
		return errors.Join(errors.New("model recipe: supervised verification arguments are not durable"), err)
	}
	effect, err := invocation.RequireEffect(ctx, reader, binding.Effect)
	if err != nil || !exactVerificationEffect(effect, binding, target) {
		return errors.Join(errors.New("model recipe: supervised verification effect differs"), err)
	}
	stimulus, err := runrecord.RequireAttemptStimulus(ctx, reader, binding.Preflight)
	if err != nil || !exactVerificationStimulus(stimulus, terminal, binding) {
		return errors.Join(errors.New("model recipe: supervised verification preflight differs"), err)
	}
	inspection, err := invocation.RequireEffect(ctx, reader, stimulus.InspectionEffect)
	if err != nil || !invocation.InspectionCovers(inspection, effect) {
		return errors.Join(errors.New("model recipe: supervised verification inspection is irrelevant"), err)
	}
	if _, err := artifact.RequireTypedContent(ctx, reader, binding.Inspection); err != nil {
		return errors.Join(errors.New("model recipe: supervised verification inspection is absent"), err)
	}
	if err := recipe.RequireInvocationCeiling(
		ctx, reader, binding.Ceiling, binding.Subject, binding.Action, target,
	); err != nil {
		return err
	}
	if err := requireVerificationDecisionAuthority(ctx, reader, terminal.Operation, binding); err != nil {
		return err
	}
	previous, found, err := runrecord.ResolveStageReceipt(ctx, reader, terminal.Operation, terminal.Node)
	if err != nil || !found || previous.State != runrecord.StageRunning ||
		previous.Attempt != terminal.Attempt || previous.Invocation == nil || *previous.Invocation != binding {
		return errors.Join(errors.New("model recipe: supervised verification lacks admitted running receipt"), err)
	}
	return nil
}

func verificationArguments(recipeID, decision artifact.ID, target string) (artifact.Content, error) {
	data, err := json.Marshal([]string{recipeID.String(), decision.String(), string(recipe.StatusVerified), target})
	if err != nil {
		return artifact.Content{}, err
	}
	return runrecord.AttemptArgumentContent(data)
}

func exactVerificationEffect(effect invocation.Effect, binding invocation.ReceiptBinding, target string) bool {
	return effect.ID == binding.Effect && effect.Manual == binding.Subject && effect.Arguments == binding.Arguments &&
		effect.ObservedResult == nil && effect.Class == invocation.ClassMutation && effect.Known && !effect.OpaqueMutation &&
		!effect.Network && !effect.Executable && !effect.Destructive && !effect.Privileged && !effect.Irreversible &&
		slices.Equal(effect.Targets, []invocation.Target{{Scope: invocation.ScopeRepository, Value: target}})
}

func exactVerificationStimulus(
	stimulus runrecord.AttemptStimulusBoundary,
	receipt runrecord.StageReceipt,
	binding invocation.ReceiptBinding,
) bool {
	return stimulus.ID == binding.Preflight && stimulus.Operation == receipt.Operation &&
		stimulus.Attempt == receipt.Attempt && stimulus.Manual == binding.Subject &&
		stimulus.Class == invocation.ClassMutation && stimulus.Arguments == binding.Arguments &&
		stimulus.Effect == binding.Effect && stimulus.Inspection == binding.Inspection &&
		stimulus.Ceiling == binding.Ceiling && stimulus.CausalContext == binding.CausalContext &&
		stimulus.Selection.Head == binding.Head
}

func requireVerificationDecisionAuthority(
	ctx context.Context,
	reader artifact.Reader,
	operation artifact.ID,
	binding invocation.ReceiptBinding,
) error {
	decision, err := runrecord.RequireHumanDecision(ctx, reader, binding.Authority)
	if err != nil || decision.Operation != operation || decision.Recipe != binding.Ceiling ||
		decision.Answer != operatoraction.AnswerGrant {
		return errors.Join(errors.New("model recipe: supervised verification authority differs"), err)
	}
	current, found, err := runrecord.ResolveHumanDecision(ctx, reader, operation)
	if err != nil || !found || current.ID != decision.ID {
		return errors.Join(errors.New("model recipe: supervised verification authority is stale"), err)
	}
	action := operatoraction.Action{
		Code: binding.Action, Summary: binding.Action,
		Argv: invocation.ApprovalArguments(binding.Subject, binding.Arguments, binding.Effect, binding.Preflight),
	}
	request, err := operatoraction.NewApprovalRequest(operation, binding.Ceiling, action, decision.Prior)
	if err != nil || request.ID != decision.Request || !decision.Binds(request) {
		return errors.Join(errors.New("model recipe: supervised verification approval arguments differ"), err)
	}
	expected, err := request.Content()
	if err != nil {
		return err
	}
	stored, found, err := artifact.ReadContent(ctx, reader, request.ID)
	if err != nil || !found || stored.Descriptor != expected.Descriptor || !bytes.Equal(stored.Data, expected.Data) {
		return errors.Join(errors.New("model recipe: supervised verification approval request is absent"), err)
	}
	return nil
}

func stageReceiptCites(receipt runrecord.StageReceipt, id artifact.ID) bool {
	return slices.ContainsFunc(receipt.Inputs, func(binding runrecord.StageBinding) bool {
		return slices.Contains(binding.Artifacts, id)
	})
}
