package modelrecipe

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/workflowrecipe"
)

const (
	// CandidateClosureMediaType identifies candidate-validation evidence.
	CandidateClosureMediaType = "application/vnd.overgo.candidate-recipe-closure+json"
	// CandidateClosureSchema identifies the closure evidence contract.
	CandidateClosureSchema = "overgo/candidate-recipe-closure/v1"
)

// CandidateRecipeClosure is the typed verdict of validating one candidate
// recipe against exact compiled capabilities. A candidate is not runnable
// because it parses: every module must resolve in the registered catalog,
// every node's placement, residency, and session policy must cohere, and
// every dependency must carry an exact artifact identity. Failures are
// recorded as exact refusals — never a fallback topology.
type CandidateRecipeClosure struct {
	Definition artifact.ID `json:"definition"`
	Task       recipe.Task `json:"task"`
	Runnable   bool        `json:"runnable"`
	Refusals   []string    `json:"refusals,omitempty"`
}

// ValidateCandidateRecipe compiles one candidate against the registered
// module catalogs and policy validators and returns the typed closure.
func ValidateCandidateRecipe(definition recipe.Definition) CandidateRecipeClosure {
	closure := CandidateRecipeClosure{Definition: definition.ID, Task: definition.Task}
	catalogForTask := catalog
	if definition.Task == recipe.TaskProjection || definition.Task == recipe.TaskTraining {
		catalogForTask = workflowrecipe.Catalog()
	}
	if err := definition.Validate(catalogForTask); err != nil {
		closure.Refusals = append(closure.Refusals, "module graph: "+err.Error())
	}
	for _, node := range definition.Nodes {
		if node.Residency == "" {
			// Host-default nodes carry no residency declaration; the graph
			// validation above already proved their module contract.
			continue
		}
		if err := validateResidencyPlacement(node.Residency, node.Placement); err != nil {
			closure.Refusals = append(closure.Refusals,
				fmt.Sprintf("node %s policy: %v", node.ID, err))
		}
	}
	for _, dependency := range definition.Dependencies {
		if !dependency.Artifact.Valid() {
			closure.Refusals = append(closure.Refusals,
				fmt.Sprintf("dependency %s lacks an exact artifact identity", dependency.Role))
		}
	}
	if definition.Task != recipe.TaskInference {
		if _, err := CompileCapability(definition); err != nil {
			closure.Refusals = append(closure.Refusals, "capability compile: "+err.Error())
		}
	}
	closure.Runnable = len(closure.Refusals) == 0
	return closure
}

// ValidateStoredCandidateRecipe adds local artifact closure to the compiled
// graph verdict. Every direct dependency and transitive lineage parent must
// exist in the supplied reader; a content-shaped ID alone cannot stand in for
// a registered transform, controller policy, checkpoint, or other capability.
func ValidateStoredCandidateRecipe(
	ctx context.Context,
	reader artifact.Reader,
	definition recipe.Definition,
) CandidateRecipeClosure {
	closure := ValidateCandidateRecipe(definition)
	if ctx == nil || reader == nil {
		closure.Refusals = append(closure.Refusals, "local capability authority is absent")
		closure.Runnable = false
		return closure
	}
	if err := ctx.Err(); err != nil {
		closure.Refusals = append(closure.Refusals, "local capability authority: "+err.Error())
		closure.Runnable = false
		return closure
	}

	roles := make(map[artifact.ID]string, len(definition.Dependencies))
	frontier := make([]artifact.ID, 0, len(definition.Dependencies))
	for _, dependency := range definition.Dependencies {
		if _, found := roles[dependency.Artifact]; found {
			continue
		}
		roles[dependency.Artifact] = fmt.Sprintf("%s[%d]", dependency.Role, dependency.Slot)
		frontier = append(frontier, dependency.Artifact)
	}
	seen := make(map[artifact.ID]bool, len(frontier))
	for len(frontier) != 0 {
		slices.SortFunc(frontier, artifact.CompareID)
		id := frontier[0]
		frontier = frontier[1:]
		if seen[id] {
			continue
		}
		seen[id] = true
		descriptor, found, err := reader.Artifact(ctx, id)
		if err != nil {
			closure.Refusals = append(closure.Refusals,
				fmt.Sprintf("capability %s artifact %s: %v", roles[id], id, err))
			continue
		}
		if !found || descriptor.ID != id {
			closure.Refusals = append(closure.Refusals,
				fmt.Sprintf("capability %s artifact %s is absent", roles[id], id))
			continue
		}
		parents, err := reader.Parents(ctx, id)
		if err != nil {
			closure.Refusals = append(closure.Refusals,
				fmt.Sprintf("capability %s lineage for %s: %v", roles[id], id, err))
			continue
		}
		for _, edge := range parents {
			if seen[edge.Parent] {
				continue
			}
			if _, found := roles[edge.Parent]; !found {
				roles[edge.Parent] = "lineage"
			}
			frontier = append(frontier, edge.Parent)
		}
	}
	closure.Runnable = len(closure.Refusals) == 0
	return closure
}

// PublishCandidateClosure commits one validation verdict as durable
// evidence citing the exact candidate definition, so a later activation can
// bind the verification — and a refusal stays on the record instead of
// disappearing into a retry.
func PublishCandidateClosure(
	ctx context.Context,
	store artifact.Repository,
	closure CandidateRecipeClosure,
) (artifact.ID, error) {
	if !closure.Definition.Valid() {
		return artifact.ID{}, errors.New("model recipe: candidate closure lacks its definition identity")
	}
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: CandidateClosureMediaType, Schema: CandidateClosureSchema,
	}
	content, err := artifact.JSONContent(contract, closure)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"recipe/candidate-closure/"+content.Descriptor.ID.String(),
		[]artifact.Content{content},
		artifact.DependencyLineage(content.Descriptor.ID, closure.Definition),
		nil,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return content.Descriptor.ID, nil
}
