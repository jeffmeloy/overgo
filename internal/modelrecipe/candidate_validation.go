package modelrecipe

import (
	"context"
	"errors"
	"fmt"

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
