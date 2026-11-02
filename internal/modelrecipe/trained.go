package modelrecipe

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// A trained model holds two marks, each an alias naming its inference recipe. trained says a training run produced the
// model, and measured that its one measurement finished, after which a run of every model leaves it out. Neither is an
// activation: what a model may serve reads as every model's does (ReadModelState).
const (
	trainedAliasPrefix  = "recipe.trained."
	measuredAliasPrefix = "recipe.measured."
)

// MarkTrained records that a training run produced definition's model.
func MarkTrained(ctx context.Context, store artifact.Repository, definition recipe.Definition) error {
	return mark(ctx, store, trainedAliasPrefix, definition.Model, definition.Task, definition.ID)
}

// MarkMeasured records that a trained model's one measurement finished under recipeID.
func MarkMeasured(ctx context.Context, store artifact.Repository, model artifact.ID, task recipe.Task, recipeID artifact.ID) error {
	return mark(ctx, store, measuredAliasPrefix, model, task, recipeID)
}

// IsTrained says a training run produced model for task.
func IsTrained(ctx context.Context, store artifact.Reader, model artifact.ID, task recipe.Task) (bool, error) {
	_, marked, err := artifact.ResolveAlias(ctx, store, trainedAliasPrefix+string(task)+"."+model.String())
	return marked, err
}

// mark binds one mark of model to recipeID. A mark already bound to it is no change, which a retried run reaches.
func mark(ctx context.Context, store artifact.Repository, prefix string, model artifact.ID, task recipe.Task, recipeID artifact.ID) error {
	name := prefix + string(task) + "." + model.String()
	current, marked, err := artifact.ResolveAlias(ctx, store, name)
	if err != nil || marked && current == recipeID {
		return err
	}
	var previous *artifact.ID
	if marked {
		previous = &current
	}
	_, err = artifact.Publish(ctx, store, artifact.Batch{
		Key: "recipe/mark/" + name + "/" + recipeID.String(), Aliases: []artifact.AliasBinding{artifact.AliasMoveFrom(name, recipeID, previous)},
	})
	return err
}

// ParseTrainedAlias names the model a trained mark is of.
func ParseTrainedAlias(name string) (artifact.ID, bool) {
	model, _, marked := parseModelAlias(trainedAliasPrefix, name)
	return model, marked
}

// ParseMeasuredAlias names the model a measured mark is of.
func ParseMeasuredAlias(name string) (artifact.ID, bool) {
	model, _, marked := parseModelAlias(measuredAliasPrefix, name)
	return model, marked
}
