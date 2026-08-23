package modelrecipe

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// ParseActiveAlias inverts activeAlias: it recognizes one activation alias and
// returns the model and task it names. The alias format is owned here, next to
// its writer, so enumerating consumers never restate it.
func ParseActiveAlias(name string) (artifact.ID, recipe.Task, bool) {
	if !strings.HasPrefix(name, activeAliasPrefix) {
		return artifact.ID{}, "", false
	}
	task, modelText, found := strings.Cut(strings.TrimPrefix(name, activeAliasPrefix), ".")
	if !found {
		return artifact.ID{}, "", false
	}
	modelID, err := artifact.ParseID(modelText)
	if err != nil || modelID.Kind() != artifact.KindModel {
		return artifact.ID{}, "", false
	}
	return modelID, recipe.Task(task), true
}

// VerifyPublishedChain proves one active recipe still loads through the
// production readers: alias, lifecycle event, decisions, verified evidence,
// model definition, profile with provenance, and tensor inventory, all under
// the current document schemas. It exists because source and published data
// can drift apart silently: every schema is tested against documents built by
// current code, so an unversioned schema change ships green while every
// already-published document becomes unreadable. This walk is the read path
// the server takes, so it fails exactly when serving would.
func VerifyPublishedChain(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	task recipe.Task,
) error {
	activation, ok, err := ActiveRecord(ctx, store, modelID, task)
	if err != nil {
		return err
	}
	if !ok {
		return errors.New("model recipe: active alias resolves to no record")
	}
	definitionID, bound := activation.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !bound {
		// Capability recipes without a model definition stop at lifecycle
		// proof; there is no deeper document chain to load.
		return nil
	}
	_, err = ResolveModelDefinition(ctx, store, definitionID)
	return err
}
