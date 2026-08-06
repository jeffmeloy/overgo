package modelrecipe

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
	"llamacpp2go/internal/repodb"
)

func CompileActive(
	ctx context.Context,
	store *repodb.Store,
	modelID artifact.ID,
	spec model.Spec,
	weights model.Weights,
) (Plan, bool, error) {
	definition, ok, err := Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok {
		return Plan{}, ok, err
	}
	plan, err := Compile(definition, spec, weights)
	return plan, err == nil, err
}

func PublishCandidate(ctx context.Context, store *repodb.Store, key string, definition recipe.Definition) (artifact.CommitID, recipe.LifecycleEvent, error) {
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
	commit, err := store.Commit(ctx, artifact.Batch{
		Key: key, Contents: []artifact.Content{definitionContent, eventContent},
		Aliases: []artifact.AliasBinding{{Name: statusAlias(definition.ID), Target: event.ID}},
	})
	return commit, event, err
}

func Transition(
	ctx context.Context,
	store *repodb.Store,
	key string,
	definition recipe.Definition,
	to recipe.Status,
	evidence []artifact.ID,
	supersedes *artifact.ID,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	previous, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	if previous.Recipe != definition.ID || previous.Model != definition.Model || previous.Task != definition.Task {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, errors.New("model recipe: lifecycle subject mismatch")
	}
	for _, evidenceID := range evidence {
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
	contents := []artifact.Content{eventContent}
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

func Active(ctx context.Context, store *repodb.Store, modelID artifact.ID, task recipe.Task) (recipe.Definition, bool, error) {
	id, ok, err := store.ResolveAlias(ctx, activeAlias(modelID, task))
	if err != nil || !ok {
		return recipe.Definition{}, ok, err
	}
	definition, err := loadDefinition(ctx, store, id)
	return definition, err == nil, err
}

func currentEvent(ctx context.Context, store *repodb.Store, recipeID artifact.ID) (recipe.LifecycleEvent, error) {
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

func loadDefinition(ctx context.Context, store *repodb.Store, id artifact.ID) (recipe.Definition, error) {
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
