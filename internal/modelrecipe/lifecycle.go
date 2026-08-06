package modelrecipe

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
)

func CompileActive(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	spec model.Spec,
	weights model.Weights,
) (Plan, bool, error) {
	definition, ok, err := Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok {
		return Plan{}, ok, err
	}
	document, bound, err := activeBoundProfile(ctx, store, definition)
	if err != nil {
		return Plan{}, false, err
	}
	if bound {
		plan, compileErr := CompileWithProfile(definition, document, spec, weights)
		return plan, compileErr == nil, compileErr
	}
	plan, err := Compile(definition, spec, weights)
	return plan, err == nil, err
}

func PublishCandidate(ctx context.Context, store artifact.Repository, key string, definition recipe.Definition) (artifact.CommitID, recipe.LifecycleEvent, error) {
	return publishCandidate(ctx, store, key, definition, nil)
}

func PublishProfileCandidate(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	document ProfileDocument,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	if err := document.ValidateIdentity(); err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	return publishCandidate(ctx, store, key, definition, &document)
}

func publishCandidate(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	document *ProfileDocument,
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
	batch := artifact.Batch{
		Key: key, Contents: []artifact.Content{definitionContent, eventContent},
		Aliases: []artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID}},
	}
	if document != nil {
		profileContent, contentErr := ProfileContent(*document)
		if contentErr != nil {
			return artifact.CommitID{}, recipe.LifecycleEvent{}, contentErr
		}
		batch.Contents = append(batch.Contents, profileContent)
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
			Name: recipeProfileAlias(definition.ID), Target: document.ID,
		})
	}
	commit, err := store.Commit(ctx, batch)
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
	return transition(ctx, store, key, definition, to, evidence, supersedes, nil)
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
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	previous, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	if previous.Recipe != definition.ID || previous.Model != definition.Model || previous.Task != definition.Task {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle subject mismatch")
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
		activeID, active, lookupErr := store.ResolveAlias(ctx, activeAlias(definition.Model, definition.Task))
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
			Previous: cloneArtifactID(supersedes),
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
	commit, err := store.Commit(ctx, artifact.Batch{Key: key, Contents: contents, Aliases: aliases})
	return commit, event, err
}

func Active(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (recipe.Definition, bool, error) {
	id, ok, err := store.ResolveAlias(ctx, activeAlias(modelID, task))
	if err != nil || !ok {
		return recipe.Definition{}, ok, err
	}
	definition, err := loadDefinition(ctx, store, id)
	return definition, err == nil, err
}

// ActiveProfile: parity-gated policy for runtime ingestion.
func ActiveProfile(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	task recipe.Task,
) (ProfileDocument, bool, error) {
	definition, ok, err := Active(ctx, store, modelID, task)
	if err != nil || !ok {
		return ProfileDocument{}, ok, err
	}
	return activeBoundProfile(ctx, store, definition)
}

func activeBoundProfile(
	ctx context.Context,
	store artifact.Reader,
	definition recipe.Definition,
) (ProfileDocument, bool, error) {
	profileID, bound, err := store.ResolveAlias(ctx, recipeProfileAlias(definition.ID))
	if err != nil || !bound {
		return ProfileDocument{}, bound, err
	}
	document, err := loadProfile(ctx, store, profileID)
	if err != nil {
		return ProfileDocument{}, false, err
	}
	event, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return ProfileDocument{}, false, err
	}
	if _, err := matchingProfileParity(
		ctx, store, event.Evidence, definition, document,
	); err != nil {
		return ProfileDocument{}, false, err
	}
	return document, true, nil
}

func currentEvent(ctx context.Context, store artifact.Reader, recipeID artifact.ID) (recipe.LifecycleEvent, error) {
	eventID, ok, err := store.ResolveAlias(ctx, statusAlias(recipeID))
	if err != nil {
		return recipe.LifecycleEvent{}, err
	}
	if !ok {
		return recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle status is absent")
	}
	content, ok, err := store.Content(ctx, eventID)
	if err != nil {
		return recipe.LifecycleEvent{}, err
	}
	if !ok || content.Descriptor.Schema != recipe.LifecycleSchema {
		return recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle content is absent or incompatible")
	}
	return recipe.ParseLifecycleEvent(content.Data)
}

func loadDefinition(ctx context.Context, store artifact.Reader, id artifact.ID) (recipe.Definition, error) {
	content, ok, err := store.Content(ctx, id)
	if err != nil {
		return recipe.Definition{}, err
	}
	if !ok || content.Descriptor.Schema != recipe.Schema {
		return recipe.Definition{}, errors.New("model recipe: definition content is absent or incompatible")
	}
	return recipe.ParseDefinition(content.Data)
}

func statusAlias(recipeID artifact.ID) string {
	return "recipe.status." + recipeID.String()
}

func activeAlias(modelID artifact.ID, task recipe.Task) string {
	return "recipe.active." + string(task) + "." + modelID.String()
}

func cloneArtifactID(id *artifact.ID) *artifact.ID {
	if id == nil {
		return nil
	}
	copy := *id
	return &copy
}
