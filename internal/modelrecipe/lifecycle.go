package modelrecipe

import (
	"context"
	"encoding/json"
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
	lineage := event.Lineage()
	aliases := []artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID}}
	for _, dependency := range definition.Dependencies {
		lineage = append(lineage, artifact.Lineage{
			Child: definition.ID, Parent: dependency.Artifact, Relation: artifact.RelationDependsOn,
		})
	}
	policyContent, policyAlias, policyFound, err := runtimePolicyBinding(definition)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	if policyFound {
		contents = append(contents, policyContent)
		aliases = append(aliases, policyAlias)
		lineage = append(lineage, artifact.Lineage{
			Child: definition.ID, Parent: policyContent.Descriptor.ID, Relation: artifact.RelationDependsOn,
		})
	}
	batch, err := artifact.NewDocumentBatch(key, contents, lineage, aliases)
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
	terminal ...runrecord.StageReceipt,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	if to == recipe.StatusVerified {
		if len(terminal) != 1 || supersedes != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: verified transition requires one non-serving terminal receipt")
		}
		return verifiedTransition(ctx, store, key, definition, evidence, terminal[0])
	}
	if len(terminal) != 0 {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: terminal receipt is reserved for supervised verification")
	}
	return transition(ctx, store, key, definition, to, evidence, supersedes, nil, nil, nil, nil)
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
	if strings.TrimSpace(reason) == "" || strings.TrimSpace(reason) != reason || !tier.Valid() {
		return errors.New("model recipe: activation decision is invalid")
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
	} else if err := EnsureRuntimePolicy(ctx, store, definition); err != nil {
		return err
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
		verification, tier, reason, nil, supersedes,
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
	definition, err := recipe.RequireDefinition(ctx, store, activeID)
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
		definition.ID, recipe.DecisionRefused, recipe.EvidenceVerified, reason,
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

// RetireOrphanedActivation retires an activation whose own trust check
// fails -- a stale alias left behind when a schema migration changed
// the model identity, or an activation whose verifier evidence never
// stood. The failed trust check IS the evidence: a trusted activation
// is refused here and must retire through the evidence-backed path.
func RetireOrphanedActivation(
	ctx context.Context,
	store artifact.Repository,
	model artifact.ID,
	task recipe.Task,
	revision, reason string,
) error {
	if strings.TrimSpace(reason) == "" {
		return errors.New("model recipe: retirement reason is empty")
	}
	alias := activeAlias(model, task)
	activeID, active, err := artifact.ResolveAlias(ctx, store, alias)
	if err != nil {
		return err
	}
	if !active {
		return errors.New("model recipe: retirement requires an active recipe")
	}
	_, _, trustErr := ActiveRecord(ctx, store, model, task)
	if trustErr == nil {
		// A trusted activation can still be dead: when no recorded
		// location holds the model bytes, the identity resolves from
		// nothing and absence is the measured orphan evidence.
		if _, pathErr := artifact.AvailablePath(ctx, store, model, artifact.LocationFile); pathErr == nil {
			return errors.New("model recipe: the activation is trusted and its bytes are present; retire it through the evidence-backed path")
		} else {
			trustErr = pathErr
		}
	}
	definition, err := recipe.RequireDefinition(ctx, store, activeID)
	if err != nil {
		return err
	}
	// The measured trust failure is the retirement evidence: it is
	// committed verbatim so the refusal cites an observation, not an
	// assertion.
	failureContract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: artifact.JSONMediaType,
		Schema: "overgo/orphan-trust-failure/v1",
	}
	failureBody, err := json.Marshal(struct {
		Model    artifact.ID `json:"model"`
		Task     recipe.Task `json:"task"`
		Recipe   artifact.ID `json:"recipe"`
		Failure  string      `json:"failure"`
		Revision string      `json:"revision"`
	}{Model: model, Task: task, Recipe: definition.ID, Failure: trustErr.Error(), Revision: revision})
	if err != nil {
		return err
	}
	failure, err := failureContract.ContentBytes(failureBody)
	if err != nil {
		return err
	}
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionRefused, recipe.EvidenceVerified,
		reason+"; trust check: "+trustErr.Error(),
		recipe.Decider{CodeCommit: revision, Derivation: definition.ID},
		[]artifact.ID{failure.Descriptor.ID},
	)
	if err != nil {
		return err
	}
	batch, err := decision.Batch(
		"recipe/orphan-retirement/" + definition.ID.String() + "/" + decision.ID.String(),
	)
	if err != nil {
		return err
	}
	batch.Artifacts = append(batch.Artifacts, failure.Descriptor)
	batch.Contents = append(batch.Contents, failure)
	// The orphan has no successor, so the active alias is removed
	// outright: a retired recipe must leave the catalog, not linger as
	// an alias naming a refused definition.
	batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
		Name: alias, Target: activeID, Previous: &activeID, Remove: true,
	})
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return err
	}
	_, _, err = Transition(
		ctx, store, "recipe/retired/"+definition.ID.String()+"/"+decision.ID.String(),
		definition, recipe.StatusSuperseded,
		[]artifact.ID{decision.ID, failure.Descriptor.ID}, nil,
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
		append(decision.Lineage(), event.Lineage()...),
		[]artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID, Previous: &current.ID}},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	return err
}

// ActivateVerified promotes an accepted, successfully verified recipe and its
// alias transition in one repository commit.
func ActivateVerified(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	verification Verification,
	tier recipe.EvidenceTier,
	reason string,
	evidence []artifact.ID,
	supersedes *artifact.ID,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	if strings.TrimSpace(reason) == "" || strings.TrimSpace(reason) != reason {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: activation reason is invalid")
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	decisionEvidence := append(slices.Clone(evidence), verified.Gate.ID, verified.Run.ID)
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionAccepted, tier, reason,
		recipe.Decider{CodeCommit: verified.Gate.CodeCommit, Derivation: verified.Gate.ID},
		decisionEvidence,
	)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	decisionContent, err := decision.Content()
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	evidence = append([]artifact.ID{decision.ID}, decision.Evidence...)
	return transition(
		ctx, store, key, definition, recipe.StatusActive, evidence, supersedes,
		[]artifact.Content{decisionContent}, decision.Lineage(), &verification, nil,
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
	pendingLineage []artifact.Lineage,
	verification *Verification,
	prepared *artifact.Batch,
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
		decision, found, decisionErr := findDecision(
			ctx, store, definition.ID, recipe.DecisionAccepted, evidence, pending,
		)
		if decisionErr != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, decisionErr
		}
		if !found || !activationDecisionMatches(decision, definition.ID, *verification) {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: activation requires an exact accepted decision")
		}
	}
	pendingIDs := make(map[artifact.ID]struct{}, len(pending))
	for _, content := range pending {
		pendingIDs[content.Descriptor.ID] = struct{}{}
	}
	if prepared != nil {
		for _, content := range prepared.Contents {
			pendingIDs[content.Descriptor.ID] = struct{}{}
		}
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
	pendingLineage = append(pendingLineage, event.Lineage()...)
	aliases := []artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID, Previous: &previous.ID}}
	var descriptors []artifact.Descriptor
	if prepared != nil {
		contents = append(prepared.Contents, contents...)
		pendingLineage = append(prepared.Lineage, pendingLineage...)
		aliases = append(prepared.Aliases, aliases...)
		descriptors = append(descriptors, prepared.Artifacts...)
	}
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
		// A rollback supersedes an alias holder that is already retired;
		// its superseded event stands, so only the alias moves.
		if active {
			if oldEvent, loadErr := currentEvent(ctx, store, activeID); loadErr != nil {
				return artifact.CommitID{}, recipe.LifecycleEvent{}, loadErr
			} else if oldEvent.To == recipe.StatusSuperseded {
				active = false
			}
		}
		if active {
			oldDefinition, loadErr := recipe.RequireDefinition(ctx, store, activeID)
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
			pendingLineage = append(pendingLineage, superseded.Lineage()...)
			aliases = append(aliases, artifact.AliasBinding{
				Name: statusAlias(activeID), Target: superseded.ID, Previous: &oldEvent.ID,
			})
		}
	}
	batch, err := artifact.NewDocumentBatch(key, contents, pendingLineage, aliases)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	batch.Artifacts = append(batch.Artifacts, descriptors...)
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
	definition, err := recipe.RequireDefinition(ctx, store, id)
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
	verified, err := runrecord.VerifyEvidence(ctx, store, definition.ID, event.Evidence)
	if err != nil {
		return Activation{}, false, fmt.Errorf("model recipe: active recipe lacks verified evidence: %w", err)
	}
	tier := recipe.EvidenceVerified
	// Evidence produced at a remote backend verifies nothing the store can
	// reproduce, so such an activation reads as experimental whatever its
	// decisions claim.
	if environment, err := runrecord.RequireEnvironment(ctx, store, verified.Gate.Environment); err != nil {
		return Activation{}, false, fmt.Errorf("model recipe: active recipe evidence environment: %w", err)
	} else if !environment.Reproducible() {
		tier = recipe.EvidenceExperimental
	}
	accepted := false
	for _, decision := range decisions {
		verification := Verification{Gate: verified.Gate.ID, Run: verified.Run.ID}
		if activationDecisionMatches(decision, definition.ID, verification) {
			accepted = true
			if decision.Tier.StrongerThan(tier) {
				tier = decision.Tier
			}
		}
	}
	if !accepted {
		return Activation{}, false, errors.New("model recipe: active recipe lacks exact accepted decision")
	}
	return Activation{
		Definition: definition, Event: event, Tier: tier, Decisions: slices.Clone(decisions),
	}, true, nil
}

func activationDecisionMatches(decision recipe.Decision, subject artifact.ID, verification Verification) bool {
	return decision.Subject == subject && decision.Outcome == recipe.DecisionAccepted && decision.Reason != "" &&
		decision.Decider.Derivation == verification.Gate &&
		slices.Contains(decision.Evidence, verification.Gate) && slices.Contains(decision.Evidence, verification.Run)
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
	// Inference executes through the compiled runner, never through a
	// capability program, but its ordered program still resolves here: the
	// catalog carries the inference module contracts, and selection needs
	// the program only for its definition, resources, and policy bindings.
	if activation.Definition.Task == recipe.TaskInference {
		program, err := recipe.CompileProgram(activation.Definition, catalog)
		return activation, program, err
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
		content, ok, err := artifact.ReadContent(ctx, store, id)
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
	collect := func(content artifact.Content, id artifact.ID) error {
		if content.Descriptor.MediaType != recipe.DecisionMediaType ||
			content.Descriptor.Schema != recipe.DecisionSchema {
			return nil
		}
		decision, err := recipe.ParseDecision(content.Data)
		if err != nil {
			return err
		}
		if decision.ID != id {
			return errors.New("model recipe: decision identity mismatch")
		}
		decisions = append(decisions, decision)
		return nil
	}
	// Evidence lists mix decisions with runs and identity-only facts, so
	// absence is tolerated; with a presence-capable store the whole list
	// resolves in one acquisition plus one batched read.
	if presence, batched := store.(artifact.ContentPresence); batched {
		present, err := presence.PresentContents(ctx, ids)
		if err != nil {
			return nil, err
		}
		if err := artifact.ReadContents(ctx, store, present, func(content artifact.Content) error {
			return collect(content, content.Descriptor.ID)
		}); err != nil {
			return nil, err
		}
		return decisions, nil
	}
	for _, id := range ids {
		content, ok, err := artifact.ReadContent(ctx, store, id)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		if err := collect(content, id); err != nil {
			return nil, err
		}
	}
	return decisions, nil
}

func statusAlias(recipeID artifact.ID) string {
	return "recipe.status." + recipeID.String()
}

// activeAliasPrefix opens every activation alias; ParseActiveAlias inverts it.
const activeAliasPrefix = "recipe.active."

func activeAlias(modelID artifact.ID, task recipe.Task) string {
	return activeAliasPrefix + string(task) + "." + modelID.String()
}
