package trainingprogram

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

type resolvedObjective struct {
	spec ObjectiveSpec
}

func resolveObjective(ctx context.Context, reader artifact.Reader, id artifact.ID) (resolvedObjective, error) {
	if ctx == nil || reader == nil || id.Kind() != artifact.KindProfile {
		return resolvedObjective{}, errors.New("training objective: invalid repository authority")
	}
	document, err := LoadObjective(ctx, reader, id)
	if err != nil {
		return resolvedObjective{}, err
	}
	if err := validateObjectiveReferences(ctx, reader, document); err != nil {
		return resolvedObjective{}, err
	}
	return resolvedObjective{spec: document.ObjectiveSpec}, nil
}
