package modelrecipe

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func PublishCandidate(ctx context.Context, store artifact.Repository, key string, definition recipe.Definition) (artifact.CommitID, recipe.LifecycleEvent, error) {
	return publishCandidate(ctx, store, key, definition)
}

func publishCandidate(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	definitionContent, err := Content(definition)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	event, err := recipe.NewLifecycleEvent(definition, "", recipe.StatusCandidate, nil, nil, nil)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	eventContent, err := event.Content()
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	contents := []artifact.Content{definitionContent, eventContent}
	lineage := make([]artifact.Lineage, 0, len(definition.Dependencies))
	for _, dependency := range definition.Dependencies {
		lineage = append(lineage, artifact.Lineage{
			Child: definition.ID, Parent: dependency.Artifact, Relation: artifact.RelationDependsOn,
		})
	}
	batch, err := artifact.NewDocumentBatch(key, contents, lineage, []artifact.AliasBinding{{
		Name: statusAlias(definition.ID), Target: event.ID,
	}})
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	commit, err := artifact.CommitBatch(ctx, store, batch)
	return commit, event, err
}

func Transition(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	to recipe.Status,
	evidence []artifact.ID,
	supersedes *artifact.ID,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	return transition(ctx, store, key, definition, to, evidence, supersedes, nil, nil)
}

// Verification defines immutable verifier gate/run identities.
type Verification struct {
	Gate artifact.ID
	Run  artifact.ID
}

// ActivateCapability advances one definition through verified activation.
func ActivateCapability(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	verification Verification,
	tier recipe.EvidenceTier,
	reason string,
) error {
	if reason == "" {
		return errors.New("model recipe: activation reason is empty")
	}
	state, published, err := Status(ctx, store, definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := PublishCandidate(ctx, store, "recipe/candidate/"+definition.ID.String(), definition); err != nil {
			return fmt.Errorf("model recipe: publish candidate: %w", err)
		}
		state = recipe.StatusCandidate
	}
	if state == recipe.StatusCandidate {
		if _, _, err := Transition(
			ctx, store, "recipe/validated/"+definition.ID.String(), definition,
			recipe.StatusValidated, nil, nil,
		); err != nil {
			return fmt.Errorf("model recipe: transition validated: %w", err)
		}
		state = recipe.StatusValidated
	}
	switch state {
	case recipe.StatusActive:
		return reverifyActiveCapability(ctx, store, definition, verification, tier, reason)
	case recipe.StatusValidated:
	default:
		return fmt.Errorf("model recipe: recipe %s is %q; activation resumes only from candidate or validated", definition.ID, state)
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		return err
	}
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionAccepted, tier, reason,
		recipe.Decider{CodeCommit: verified.Gate.CodeCommit, Derivation: verified.Gate.ID},
		[]artifact.ID{verified.Gate.ID, verified.Run.ID},
	)
	if err != nil {
		return err
	}
	decisionBatch, err := decision.Batch(
		"recipe/activation-decision/" + definition.ID.String() + "/" + decision.ID.String(),
	)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, decisionBatch); err != nil {
		return err
	}
	var supersedes *artifact.ID
	activeID, active, err := artifact.ResolveAlias(ctx, store, activeAlias(definition.Model, definition.Task))
	if err != nil {
		return err
	}
	if active && activeID != definition.ID {
		supersedes = &activeID
	}
	if _, _, err := ActivateVerified(
		ctx, store, "recipe/active/"+definition.ID.String(), definition,
		verification, []artifact.ID{decision.ID}, supersedes,
	); err != nil {
		return fmt.Errorf("model recipe: transition active: %w", err)
	}
	return nil
}

// RetireActiveCapability supersedes an invalid active recipe with failed proof.
func RetireActiveCapability(
	ctx context.Context,
	store artifact.Repository,
	candidate recipe.Definition,
	verification Verification,
	reason string,
) error {
	if strings.TrimSpace(reason) == "" {
		return errors.New("model recipe: retirement reason is empty")
	}
	activeID, active, err := artifact.ResolveAlias(ctx, store, activeAlias(candidate.Model, candidate.Task))
	if err != nil {
		return err
	}
	if !active {
		return errors.New("model recipe: retirement requires an active recipe")
	}
	definition, err := loadDefinition(ctx, store, activeID)
	if err != nil {
		return err
	}
	current, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return err
	}
	if current.To != recipe.StatusActive || definition.Model != candidate.Model || definition.Task != candidate.Task {
		return errors.New("model recipe: retirement subject mismatch")
	}
	failed, err := runrecord.VerifyFailedGateRun(
		ctx, store, candidate.ID, verification.Gate, verification.Run,
	)
	if err != nil {
		return err
	}
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionRefused, recipe.EvidenceExperimental, reason,
		recipe.Decider{CodeCommit: failed.Gate.CodeCommit, Derivation: failed.Gate.ID},
		[]artifact.ID{failed.Gate.ID, failed.Run.ID},
	)
	if err != nil {
		return err
	}
	batch, err := decision.Batch(
		"recipe/retirement-decision/" + definition.ID.String() + "/" + decision.ID.String(),
	)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return err
	}
	_, _, err = Transition(
		ctx, store, "recipe/retired/"+definition.ID.String()+"/"+decision.ID.String(),
		definition, recipe.StatusSuperseded,
		[]artifact.ID{decision.ID, failed.Gate.ID, failed.Run.ID}, nil,
	)
	return err
}

// reverifyActiveCapability: refresh proof after verifier-schema or code change.
func reverifyActiveCapability(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	verification Verification,
	tier recipe.EvidenceTier,
	reason string,
) error {
	current, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return err
	}
	if current.To != recipe.StatusActive {
		return fmt.Errorf("model recipe: recipe %s is not active", definition.ID)
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		return err
	}
	if slices.Contains(current.Evidence, verification.Gate) && slices.Contains(current.Evidence, verification.Run) {
		return nil
	}
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionAccepted, tier, reason,
		recipe.Decider{CodeCommit: verified.Gate.CodeCommit, Derivation: verified.Gate.ID},
		[]artifact.ID{verified.Gate.ID, verified.Run.ID},
	)
	if err != nil {
		return err
	}
	decisionContent, err := decision.Content()
	if err != nil {
		return err
	}
	event, err := recipe.NewLifecycleEvent(
		definition, recipe.StatusActive, recipe.StatusActive, &current.ID, nil,
		[]artifact.ID{decision.ID, verified.Gate.ID, verified.Run.ID},
	)
	if err != nil {
		return err
	}
	eventContent, err := event.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"recipe/reverified/"+definition.ID.String()+"/"+event.ID.String(),
		[]artifact.Content{decisionContent, eventContent},
		decision.Lineage(),
		[]artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID, Previous: &current.ID}},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	return err
}

// ActivateVerified promotes only from a successful recipe-bound verifier run.
func ActivateVerified(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	verification Verification,
	evidence []artifact.ID,
	supersedes *artifact.ID,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	evidence = append(slices.Clone(evidence), verification.Gate, verification.Run)
	return transition(
		ctx, store, key, definition, recipe.StatusActive, evidence, supersedes, nil, &verification,
	)
}

func transition(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	to recipe.Status,
	evidence []artifact.ID,
	supersedes *artifact.ID,
	pending []artifact.Content,
	verification *Verification,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	previous, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	if previous.Recipe != definition.ID || previous.Model != definition.Model || previous.Task != definition.Task {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle subject mismatch")
	}
	if to == recipe.StatusActive {
		if verification == nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: activation requires verifier gate/run identities")
		}
		if _, ok, lookupErr := store.Artifact(ctx, definition.Model); lookupErr != nil || !ok {
			if lookupErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, lookupErr
			}
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: activation model artifact is absent")
		}
		if _, verifyErr := runrecord.VerifyGateRun(
			ctx, store, definition.ID, verification.Gate, verification.Run,
		); verifyErr != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, fmt.Errorf("model recipe: activation verification: %w", verifyErr)
		}
	}
	pendingIDs := make(map[artifact.ID]struct{}, len(pending))
	for _, content := range pending {
		pendingIDs[content.Descriptor.ID] = struct{}{}
	}
	for _, evidenceID := range evidence {
		if _, ok := pendingIDs[evidenceID]; ok {
			continue
		}
		if _, ok, lookupErr := store.Artifact(ctx, evidenceID); lookupErr != nil || !ok {
			if lookupErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, lookupErr
			}
			return artifact.CommitID{}, recipe.LifecycleEvent{}, fmt.Errorf("model recipe: unknown evidence %s", evidenceID)
		}
	}
	if to == recipe.StatusRefused {
		decision, found, decisionErr := findDecision(
			ctx, store, definition.ID, recipe.DecisionRefused, evidence, pending,
		)
		if decisionErr != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, decisionErr
		}
		if !found || decision.Reason == "" {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: refusal requires typed decision evidence")
		}
	}
	event, err := recipe.NewLifecycleEvent(definition, previous.To, to, &previous.ID, supersedes, evidence)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	eventContent, err := event.Content()
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	contents := append(append([]artifact.Content(nil), pending...), eventContent)
	aliases := []artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID, Previous: &previous.ID}}
	if to == recipe.StatusActive {
		activeID, active, lookupErr := artifact.ResolveAlias(ctx, store, activeAlias(definition.Model, definition.Task))
		if lookupErr != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, lookupErr
		}
		if active && (supersedes == nil || *supersedes != activeID) {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: activation must supersede current active recipe")
		}
		if !active && supersedes != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: activation supersedes no active recipe")
		}
		aliases = append(aliases, artifact.AliasBinding{
			Name: activeAlias(definition.Model, definition.Task), Target: definition.ID,
			Previous: artifact.CloneID(supersedes),
		})
		if active {
			oldDefinition, loadErr := loadDefinition(ctx, store, activeID)
			if loadErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, loadErr
			}
			oldEvent, loadErr := currentEvent(ctx, store, activeID)
			if loadErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, loadErr
			}
			superseded, loadErr := recipe.NewLifecycleEvent(
				oldDefinition, oldEvent.To, recipe.StatusSuperseded, &oldEvent.ID, nil, evidence,
			)
			if loadErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, loadErr
			}
			supersededContent, loadErr := superseded.Content()
			if loadErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, loadErr
			}
			contents = append(contents, supersededContent)
			aliases = append(aliases, artifact.AliasBinding{
				Name: statusAlias(activeID), Target: superseded.ID, Previous: &oldEvent.ID,
			})
		}
	}
	batch, err := artifact.NewDocumentBatch(key, contents, nil, aliases)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	commit, err := artifact.CommitBatch(ctx, store, batch)
	return commit, event, err
}

type Activation struct {
	Definition recipe.Definition
	Event      recipe.LifecycleEvent
	Tier       recipe.EvidenceTier
	Decisions  []recipe.Decision
}

// HasActiveRecipe reports declaration only; ActiveRecord validates evidence.
func HasActiveRecipe(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (bool, error) {
	id, ok, err := artifact.ResolveAlias(ctx, store, activeAlias(modelID, task))
	if err != nil || !ok {
		return ok, err
	}
	status, published, err := Status(ctx, store, id)
	if err != nil || !published {
		return false, err
	}
	return status != recipe.StatusSuperseded && status != recipe.StatusRefused, nil
}

func ActiveRecord(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (Activation, bool, error) {
	id, ok, err := artifact.ResolveAlias(ctx, store, activeAlias(modelID, task))
	if err != nil || !ok {
		return Activation{}, ok, err
	}
	definition, err := loadDefinition(ctx, store, id)
	if err != nil {
		return Activation{}, false, err
	}
	if definition.Model != modelID || definition.Task != task {
		return Activation{}, false, errors.New("model recipe: active alias subject mismatch")
	}
	event, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return Activation{}, false, err
	}
	decisions, err := loadDecisions(ctx, store, event.Evidence)
	if err != nil {
		return Activation{}, false, err
	}
	if event.To != recipe.StatusActive {
		reason := ""
		for _, decision := range decisions {
			if decision.Subject == definition.ID && decision.Outcome == recipe.DecisionRefused {
				reason = decision.Reason
				break
			}
		}
		if reason != "" {
			return Activation{}, false, fmt.Errorf("model recipe: active alias names refused recipe: %s", reason)
		}
		return Activation{}, false, fmt.Errorf("model recipe: active alias names recipe in %q state", event.To)
	}
	if _, err := runrecord.VerifyEvidence(ctx, store, definition.ID, event.Evidence); err != nil {
		return Activation{}, false, fmt.Errorf("model recipe: active recipe lacks verified evidence: %w", err)
	}
	tier := recipe.EvidenceExperimental
	for _, decision := range decisions {
		if decision.Subject == definition.ID && decision.Outcome == recipe.DecisionAccepted &&
			decision.Tier.StrongerThan(tier) {
			tier = decision.Tier
		}
	}
	return Activation{
		Definition: definition, Event: event, Tier: tier, Decisions: slices.Clone(decisions),
	}, true, nil
}

// ResolveActiveCapability returns the verified executable capability program.
func ResolveActiveCapability(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (Activation, recipe.Program, error) {
	return resolveActiveCapability(ctx, store, modelID, task)
}

func resolveActiveCapability(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (Activation, recipe.Program, error) {
	activation, active, err := ActiveRecord(ctx, store, modelID, task)
	if err == nil && !active {
		err = fmt.Errorf("model recipe: model %s has no active %s recipe", modelID, task)
	}
	if err != nil {
		return Activation{}, recipe.Program{}, err
	}
	program, err := CompileCapability(activation.Definition)
	return activation, program, err
}

// Status returns the current recipe lifecycle state and whether it was published.
func Status(ctx context.Context, store artifact.Reader, recipeID artifact.ID) (recipe.Status, bool, error) {
	if _, ok, err := artifact.ResolveAlias(ctx, store, statusAlias(recipeID)); err != nil || !ok {
		return "", false, err
	}
	event, err := currentEvent(ctx, store, recipeID)
	if err != nil {
		return "", false, err
	}
	return event.To, true, nil
}

func currentEvent(ctx context.Context, store artifact.Reader, recipeID artifact.ID) (recipe.LifecycleEvent, error) {
	eventID, ok, err := artifact.ResolveAlias(ctx, store, statusAlias(recipeID))
	if err != nil {
		return recipe.LifecycleEvent{}, err
	}
	if !ok {
		return recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle status is absent")
	}
	event, ok, err := recipe.ReadLifecycleEvent(ctx, store, eventID)
	if err != nil {
		return recipe.LifecycleEvent{}, err
	}
	if !ok {
		return recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle content is absent or incompatible")
	}
	return event, nil
}

func findDecision(
	ctx context.Context,
	store artifact.Reader,
	subject artifact.ID,
	outcome recipe.DecisionOutcome,
	evidence []artifact.ID,
	pending []artifact.Content,
) (recipe.Decision, bool, error) {
	pendingByID := make(map[artifact.ID]artifact.Content, len(pending))
	for _, content := range pending {
		pendingByID[content.Descriptor.ID] = content
	}
	for _, id := range evidence {
		if content, ok := pendingByID[id]; ok {
			if content.Descriptor.MediaType != recipe.DecisionMediaType || content.Descriptor.Schema != recipe.DecisionSchema {
				continue
			}
			decision, err := recipe.ParseDecision(content.Data)
			if err != nil {
				return recipe.Decision{}, false, err
			}
			if decision.ID != id {
				return recipe.Decision{}, false, errors.New("model recipe: pending decision identity mismatch")
			}
			if decision.Subject == subject && decision.Outcome == outcome {
				return decision, true, nil
			}
			continue
		}
		content, ok, err := store.Content(ctx, id)
		if err != nil {
			return recipe.Decision{}, false, err
		}
		if !ok || content.Descriptor.MediaType != recipe.DecisionMediaType ||
			content.Descriptor.Schema != recipe.DecisionSchema {
			continue
		}
		decision, err := recipe.ParseDecision(content.Data)
		if err != nil {
			return recipe.Decision{}, false, err
		}
		if decision.Subject == subject && decision.Outcome == outcome {
			return decision, true, nil
		}
	}
	return recipe.Decision{}, false, nil
}

func loadDecisions(ctx context.Context, store artifact.Reader, ids []artifact.ID) ([]recipe.Decision, error) {
	decisions := make([]recipe.Decision, 0)
	for _, id := range ids {
		content, ok, err := store.Content(ctx, id)
		if err != nil {
			return nil, err
		}
		if !ok || content.Descriptor.MediaType != recipe.DecisionMediaType ||
			content.Descriptor.Schema != recipe.DecisionSchema {
			continue
		}
		decision, err := recipe.ParseDecision(content.Data)
		if err != nil {
			return nil, err
		}
		if decision.ID != id {
			return nil, errors.New("model recipe: decision identity mismatch")
		}
		decisions = append(decisions, decision)
	}
	return decisions, nil
}

func loadDefinition(ctx context.Context, store artifact.Reader, id artifact.ID) (recipe.Definition, error) {
	content, ok, err := store.Content(ctx, id)
	if err != nil {
		return recipe.Definition{}, err
	}
	if !ok || content.Descriptor.Schema != recipe.Schema {
		return recipe.Definition{}, errors.New("model recipe: definition content is absent or incompatible")
	}
	definition, err := recipe.ParseDefinition(content.Data)
	if err != nil {
		return recipe.Definition{}, err
	}
	if recipe.DefinitionDocumentContract().ValidateContent(content, id) != nil || definition.ID != id {
		return recipe.Definition{}, errors.New("model recipe: definition content identity differs")
	}
	return definition, nil
}

func statusAlias(recipeID artifact.ID) string {
	return "recipe.status." + recipeID.String()
}

func activeAlias(modelID artifact.ID, task recipe.Task) string {
	return "recipe.active." + string(task) + "." + modelID.String()
}
