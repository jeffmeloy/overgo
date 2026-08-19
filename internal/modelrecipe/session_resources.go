package modelrecipe

import (
	"context"
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// SessionResourcePlan: recipe lifetime plus catalog extent.
type SessionResourcePlan struct {
	Identity      artifact.ID
	Model         artifact.ID
	Recipe        artifact.ID
	Session       recipe.SessionPolicy
	ArtifactBytes uint64
}

type sessionResourceIdentity struct {
	Model         artifact.ID          `json:"model"`
	Recipe        artifact.ID          `json:"recipe"`
	Session       recipe.SessionPolicy `json:"session"`
	ArtifactBytes uint64               `json:"artifact_bytes"`
}

// CompileSessionResourcePlan: catalog-derived session admission facts.
func CompileSessionResourcePlan(ctx context.Context, reader artifact.Reader, program recipe.Program) (SessionResourcePlan, error) {
	if ctx == nil || reader == nil {
		return SessionResourcePlan{}, errors.New("model recipe: session resource authority is absent")
	}
	definition := program.Definition()
	if definition.ID.Kind() != artifact.KindRecipe || definition.Model.Kind() != artifact.KindModel {
		return SessionResourcePlan{}, errors.New("model recipe: invalid session resource identity")
	}
	var session recipe.SessionPolicy
	for _, node := range definition.Nodes {
		if node.Session == "" {
			continue
		}
		if session != "" || !node.Session.Valid() {
			return SessionResourcePlan{}, errors.New("model recipe: session resource policy is ambiguous")
		}
		session = node.Session
	}
	if session == "" {
		return SessionResourcePlan{}, errors.New("model recipe: session resource policy is absent")
	}
	bytes, err := manifestArtifactBytes(ctx, reader, definition.Model, map[artifact.ID]struct{}{})
	if err != nil || bytes == 0 {
		return SessionResourcePlan{}, errors.Join(errors.New("model recipe: session artifact extent is unavailable"), err)
	}
	body := sessionResourceIdentity{Model: definition.Model, Recipe: definition.ID, Session: session, ArtifactBytes: bytes}
	id, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return SessionResourcePlan{}, err
	}
	return SessionResourcePlan{Identity: id, Model: body.Model, Recipe: body.Recipe, Session: body.Session, ArtifactBytes: body.ArtifactBytes}, nil
}

func manifestArtifactBytes(ctx context.Context, reader artifact.Reader, id artifact.ID, seen map[artifact.ID]struct{}) (uint64, error) {
	if _, duplicate := seen[id]; duplicate {
		return 0, nil
	}
	seen[id] = struct{}{}
	manifest, found, err := reader.Manifest(ctx, id)
	if err != nil {
		return 0, err
	}
	if !found {
		descriptor, found, err := reader.Artifact(ctx, id)
		if err != nil || !found || descriptor.Size == 0 {
			return 0, errors.Join(errors.New("model recipe: session component is absent"), err)
		}
		return descriptor.Size, nil
	}
	var total uint64
	for _, component := range manifest.Components {
		bytes, err := manifestArtifactBytes(ctx, reader, component.Artifact, seen)
		if err != nil || total > math.MaxUint64-bytes {
			return 0, errors.Join(errors.New("model recipe: session artifact extent overflow"), err)
		}
		total += bytes
	}
	return total, nil
}
