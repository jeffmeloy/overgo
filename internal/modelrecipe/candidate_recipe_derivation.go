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
	candidateRecipePrimarySlot uint32 = iota
	candidateRecipeSecondarySlot
)

// CandidateRecipePolicy is the local execution policy encoded into every
// derived inference graph. Additional tasks retain the placement and lifetime
// policy compiled into their Go module catalog.
type CandidateRecipePolicy struct {
	Placement recipe.Placement
	Session   recipe.SessionPolicy
	Residency recipe.ResidencyPolicy
	Tasks     []recipe.Task
}

// CandidateTaskRecipe binds one supported task to its immutable execution graph.
type CandidateTaskRecipe struct {
	Task       recipe.Task
	Definition recipe.Definition
}

// CandidateArmRecipes is the exact recipe set for one primary or declared
// ablation output. Evaluation is deliberately only a graph candidate; this
// owner grants no validation, execution, promotion, or activation authority.
type CandidateArmRecipes struct {
	Component       artifact.ID
	Ablation        artifact.ID
	Omitted         artifact.ID
	Realization     artifact.ID
	Model           artifact.ID
	Profile         artifact.ID
	ModelDefinition artifact.ID
	TensorInventory artifact.ID
	Output          artifact.ID
	Run             artifact.ID
	Observation     artifact.ID
	Execution       []CandidateTaskRecipe
	Evaluation      recipe.Definition
}

// CandidateRecipeSet is the immutable-by-identity projection of the shared
// Candidate -> CandidateCompilation -> CandidateMaterialization spine.
// Definitions carry the exact authorities as typed dependencies, so copying
// this value cannot detach a graph from the realization that produced it.
type CandidateRecipeSet struct {
	Candidate       artifact.ID
	Admission       artifact.ID
	Trial           artifact.ID
	EvaluationPlan  artifact.ID
	Materialization artifact.ID
	Code            artifact.ID
	Environment     artifact.ID
	Arms            []CandidateArmRecipes
}

// DeriveCandidateRecipes reopens the one stored candidate spine, resolves each
// materialized model definition, and derives closed execution and evaluation
// graphs solely from the compiled modelrecipe and workflowrecipe catalogs.
// It performs no publication or discovery and refuses an unsupported or
// duplicate task instead of choosing a fallback topology.
func DeriveCandidateRecipes(
	ctx context.Context,
	reader artifact.Reader,
	materializationID artifact.ID,
	policy CandidateRecipePolicy,
) (CandidateRecipeSet, error) {
	if ctx == nil || reader == nil || materializationID.Kind() != artifact.KindEvidence {
		return CandidateRecipeSet{}, errors.New("model recipe: candidate recipe authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return CandidateRecipeSet{}, err
	}
	materialization, trial, evaluation, components, _, err := requireCandidateRecipeSpine(
		ctx, reader, materializationID,
	)
	if err != nil {
		return CandidateRecipeSet{}, err
	}
	componentByID := make(map[artifact.ID]CandidateComponentPlan, len(components))
	for _, component := range components {
		componentByID[component.ID] = component
	}
	arms := make([]CandidateArmRecipes, len(materialization.Arms))
	for index, arm := range materialization.Arms {
		component, found := componentByID[arm.Component]
		if !found {
			return CandidateRecipeSet{}, errors.New("model recipe: materialized arm has no compiled component")
		}
		resolved, resolveErr := ResolveModelDefinition(ctx, reader, arm.ModelDefinition)
		if resolveErr != nil {
			return CandidateRecipeSet{}, fmt.Errorf("model recipe: resolve materialized arm: %w", resolveErr)
		}
		if resolved.Document.Model != arm.Model || resolved.Tensors.ID != arm.TensorInventory {
			return CandidateRecipeSet{}, errors.New("model recipe: materialized arm model definition differs")
		}
		for _, id := range []artifact.ID{arm.Output, arm.Run, arm.Observation} {
			if _, requireErr := artifact.RequireTypedContent(ctx, reader, id); requireErr != nil {
				return CandidateRecipeSet{}, fmt.Errorf("model recipe: materialized arm fact %s: %w", id, requireErr)
			}
		}
		dependencies := candidateArmRecipeDependencies(
			materialization, trial, evaluation, component, arm,
		)
		execution, deriveErr := deriveResolvedExecutionRecipes(resolved, policy, dependencies)
		if deriveErr != nil {
			return CandidateRecipeSet{}, deriveErr
		}
		evaluationDefinition, deriveErr := deriveCandidateEvaluationRecipe(
			execution[0].Definition.Dependencies, policy.Placement, execution,
		)
		if deriveErr != nil {
			return CandidateRecipeSet{}, deriveErr
		}
		arms[index] = CandidateArmRecipes{
			Component: arm.Component, Ablation: arm.Ablation, Omitted: arm.Omitted,
			Realization: arm.Realization, Model: arm.Model, Profile: resolved.Profile.ID,
			ModelDefinition: arm.ModelDefinition, TensorInventory: arm.TensorInventory,
			Output: arm.Output, Run: arm.Run, Observation: arm.Observation,
			Execution: execution, Evaluation: evaluationDefinition,
		}
	}
	return CandidateRecipeSet{
		Candidate: materialization.Candidate, Admission: materialization.Admission,
		Trial: materialization.Trial, EvaluationPlan: materialization.EvaluationPlan,
		Materialization: materialization.ID, Code: materialization.Code,
		Environment: materialization.Environment, Arms: arms,
	}, nil
}

func requireCandidateRecipeSpine(
	ctx context.Context,
	reader artifact.Reader,
	materializationID artifact.ID,
) (CandidateMaterialization, CandidateTrial, CandidateEvaluationPlan, []CandidateComponentPlan, []CandidateAblation, error) {
	materialization, err := RequireCandidateMaterialization(ctx, reader, materializationID)
	if err != nil {
		return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
	}
	trial, err := candidateTrialCodec.RequireExactLineage(ctx, reader, materialization.Trial, CandidateTrial.Lineage)
	if err != nil {
		return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
	}
	if err := validateStoredCandidateTrial(ctx, reader, trial); err != nil {
		return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
	}
	evaluation, err := candidateEvaluationPlanCodec.RequireExactLineage(
		ctx, reader, trial.EvaluationPlan, CandidateEvaluationPlan.Lineage,
	)
	if err != nil {
		return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
	}
	if materialization.Candidate != trial.Candidate || materialization.Admission != trial.Admission ||
		materialization.EvaluationPlan != trial.EvaluationPlan || materialization.Code != trial.Code ||
		materialization.Environment != trial.Environment {
		return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil,
			errors.New("model recipe: candidate materialization leaves the compiled trial spine")
	}
	components := make([]CandidateComponentPlan, len(trial.Components))
	for index, id := range trial.Components {
		components[index], err = candidateComponentPlanCodec.RequireExactLineage(
			ctx, reader, id, CandidateComponentPlan.Lineage,
		)
		if err != nil {
			return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
		}
	}
	loadedAblations := make([]CandidateAblation, len(trial.Ablations))
	for index, id := range trial.Ablations {
		loadedAblations[index], err = candidateAblationCodec.RequireExactLineage(
			ctx, reader, id, CandidateAblation.Lineage,
		)
		if err != nil {
			return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
		}
	}
	compiled := CandidateCompilation{
		Trial: trial, EvaluationPlan: evaluation, Components: components, Ablations: loadedAblations,
	}
	if err := validateCandidateMaterializedArms(compiled, materialization.Arms); err != nil {
		return CandidateMaterialization{}, CandidateTrial{}, CandidateEvaluationPlan{}, nil, nil, err
	}
	return materialization, trial, evaluation, components, loadedAblations, nil
}

func deriveResolvedExecutionRecipes(
	resolved ResolvedModelDefinition,
	policy CandidateRecipePolicy,
	additional []recipe.Dependency,
) ([]CandidateTaskRecipe, error) {
	checked, err := resolved.Document.Resolve(resolved.Profile, resolved.Tensors)
	if err != nil || !checked.Document.ID.Valid() || !checked.Document.Model.Valid() {
		return nil, errors.Join(errors.New("model recipe: recipe derivation requires a resolved model definition"), err)
	}
	tasks, err := canonicalCandidateRecipeTasks(policy.Tasks)
	if err != nil {
		return nil, err
	}
	baseDependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: checked.Document.Model},
		{Role: recipe.DependencyProfile, Artifact: checked.Profile.ID},
		{Role: recipe.DependencyDefinition, Artifact: checked.Document.ID},
		{Role: recipe.DependencyTensorInventory, Artifact: checked.Tensors.ID},
	}
	baseDependencies, err = mergeCandidateRecipeDependencies(baseDependencies, additional)
	if err != nil {
		return nil, err
	}
	result := make([]CandidateTaskRecipe, len(tasks))
	for index, task := range tasks {
		var definition recipe.Definition
		if task == recipe.TaskInference {
			definition, err = InferenceWithModelDefinition(
				checked.Document.Model, checked.Profile.ID, checked.Document.ID,
				policy.Placement, policy.Session, policy.Residency,
			)
		} else {
			definition, err = CapabilityDefinition(task, checked.Document.Model)
		}
		if err != nil {
			return nil, err
		}
		definition, err = bindCandidateRecipeDependencies(definition, baseDependencies)
		if err != nil {
			return nil, err
		}
		if task == recipe.TaskInference {
			_, err = recipe.CompileProgram(definition, catalog)
		} else {
			_, err = CompileCapability(definition)
		}
		if err != nil {
			return nil, fmt.Errorf("model recipe: compile derived %s graph: %w", task, err)
		}
		result[index] = CandidateTaskRecipe{Task: task, Definition: definition}
	}
	return result, nil
}

func deriveCandidateEvaluationRecipe(
	dependencies []recipe.Dependency,
	placement recipe.Placement,
	execution []CandidateTaskRecipe,
) (recipe.Definition, error) {
	for index, candidate := range execution {
		dependencies = append(dependencies, recipe.Dependency{
			Role: recipe.DependencyExecutionRecipe, Slot: uint32(index), Artifact: candidate.Definition.ID,
		})
	}
	evaluate := recipe.Node{
		ID: "evaluate", Module: workflowrecipe.ModuleEvaluateModel, Placement: placement,
	}
	record := recipe.Node{
		ID: "record", Module: workflowrecipe.ModuleRecordModel, Placement: placement,
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies,
		[]recipe.Node{evaluate, record},
		[]recipe.Edge{{
			From: recipe.Endpoint{Node: evaluate.ID, Port: workflowrecipe.ModelBuildStatePort},
			To:   recipe.Endpoint{Node: record.ID, Port: workflowrecipe.ModelBuildStatePort},
		}},
		[]recipe.Input{{
			Name: workflowrecipe.ModelBuildStatePort, Data: recipe.DataArtifact,
			Target: recipe.Endpoint{Node: evaluate.ID, Port: workflowrecipe.ModelBuildStatePort},
		}},
		[]recipe.Output{{
			Name: workflowrecipe.ModelBuildStatePort, Data: recipe.DataArtifact,
			Source: recipe.Endpoint{Node: record.ID, Port: workflowrecipe.ModelBuildStatePort},
		}},
	)
	if err != nil {
		return recipe.Definition{}, err
	}
	if _, err := recipe.CompileProgram(definition, workflowrecipe.Catalog()); err != nil {
		return recipe.Definition{}, fmt.Errorf("model recipe: compile derived evaluation graph: %w", err)
	}
	return definition, nil
}

func candidateArmRecipeDependencies(
	materialization CandidateMaterialization,
	trial CandidateTrial,
	evaluation CandidateEvaluationPlan,
	component CandidateComponentPlan,
	arm CandidateMaterializedArm,
) []recipe.Dependency {
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyCandidateTrial, Artifact: trial.ID},
		{Role: recipe.DependencyCandidateEvaluation, Artifact: evaluation.ID},
		{Role: recipe.DependencyCandidateComponent, Artifact: component.ID},
		{Role: recipe.DependencyCandidateMaterialization, Artifact: materialization.ID},
		{Role: recipe.DependencyCodeAuthority, Artifact: trial.Code},
		{Role: recipe.DependencyEnvironmentAuthority, Artifact: trial.Environment},
		{Role: recipe.DependencyFalsifier, Artifact: evaluation.Falsifier},
		{Role: recipe.DependencyBudget, Slot: candidateRecipePrimarySlot, Artifact: evaluation.DevelopmentBudget},
		{Role: recipe.DependencyBudget, Slot: candidateRecipeSecondarySlot, Artifact: evaluation.PromotionBudget},
		{Role: recipe.DependencyDatasetShard, Slot: candidateRecipePrimarySlot, Artifact: evaluation.DevelopmentSplit},
		{Role: recipe.DependencyDatasetShard, Slot: candidateRecipeSecondarySlot, Artifact: evaluation.PromotionSplit},
		{Role: recipe.DependencyDerivationProfile, Artifact: arm.Realization},
		{Role: recipe.DependencyPlacement, Artifact: component.ResourcePolicy},
		{Role: recipe.DependencyOutput, Artifact: arm.Output},
		{Role: recipe.DependencyRun, Artifact: arm.Run},
		{Role: recipe.DependencyObservation, Artifact: arm.Observation},
	}
	for index, ablation := range trial.Ablations {
		dependencies = append(dependencies, recipe.Dependency{
			Role: recipe.DependencyCandidateAblation, Slot: uint32(index), Artifact: ablation,
		})
	}
	if arm.Omitted.Valid() {
		dependencies = append(dependencies, recipe.Dependency{
			Role: recipe.DependencyDefinition, Slot: candidateRecipeSecondarySlot, Artifact: arm.Omitted,
		})
	}
	return dependencies
}

func canonicalCandidateRecipeTasks(requested []recipe.Task) ([]recipe.Task, error) {
	seen := make(map[recipe.Task]bool, len(requested)+1)
	tasks := []recipe.Task{recipe.TaskInference}
	for _, task := range requested {
		if !task.Valid() {
			return nil, fmt.Errorf("model recipe: unsupported capability task %q", task)
		}
		if seen[task] {
			return nil, fmt.Errorf("model recipe: duplicate candidate task %q", task)
		}
		seen[task] = true
		if task != recipe.TaskInference {
			tasks = append(tasks, task)
		}
	}
	slices.Sort(tasks)
	return tasks, nil
}

func bindCandidateRecipeDependencies(
	definition recipe.Definition,
	dependencies []recipe.Dependency,
) (recipe.Definition, error) {
	merged, err := mergeCandidateRecipeDependencies(definition.Dependencies, dependencies)
	if err != nil {
		return recipe.Definition{}, err
	}
	return recipe.NewDefinitionWithDependencies(
		definition.Task, merged, definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
	)
}

func mergeCandidateRecipeDependencies(
	left, right []recipe.Dependency,
) ([]recipe.Dependency, error) {
	result := slices.Clone(left)
	byRole := make(map[struct {
		role recipe.DependencyRole
		slot uint32
	}]artifact.ID, len(left)+len(right))
	for _, dependency := range result {
		byRole[struct {
			role recipe.DependencyRole
			slot uint32
		}{dependency.Role, dependency.Slot}] = dependency.Artifact
	}
	for _, dependency := range right {
		key := struct {
			role recipe.DependencyRole
			slot uint32
		}{dependency.Role, dependency.Slot}
		if existing, found := byRole[key]; found {
			if existing != dependency.Artifact {
				return nil, fmt.Errorf(
					"model recipe: candidate dependency %s[%d] differs", dependency.Role, dependency.Slot,
				)
			}
			continue
		}
		byRole[key] = dependency.Artifact
		result = append(result, dependency)
	}
	return result, nil
}
