package modelrecipe

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

type candidateRecipeDerivationFixture struct {
	store           *overgodb.Store
	materialization CandidateMaterialization
	evaluation      CandidateEvaluationPlan
	ablation        CandidateAblation
	models          map[artifact.ID]ResolvedModelDefinition
}

// TestCandidateRecipeDerivationUsesSharedSpine proves recipe derivation is a
// read-only projection of the one persisted candidate spine. Every execution
// and evaluation graph binds the exact materialized output and all declared
// comparison, code, environment, split, budget, and observation authorities.
func TestCandidateRecipeDerivationUsesSharedSpine(t *testing.T) {
	fixture := newCandidateRecipeDerivationFixture(t)
	t.Cleanup(func() { _ = fixture.store.Close() })
	before, _ := fixture.store.Head()
	policy := CandidateRecipePolicy{
		Placement: recipe.PlacementHost, Session: recipe.SessionCapacity,
		Residency: recipe.ResidencyHostReference, Tasks: []recipe.Task{recipe.TaskForecast},
	}
	first, err := DeriveCandidateRecipes(
		t.Context(), fixture.store, fixture.materialization.ID, policy,
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveCandidateRecipes(
		t.Context(), fixture.store, fixture.materialization.ID, policy,
	)
	if err != nil {
		t.Fatal(err)
	}
	after, _ := fixture.store.Head()
	if before != after {
		t.Fatalf("read-only derivation changed HEAD: %s -> %s", before, after)
	}
	if first.Candidate != fixture.materialization.Candidate ||
		first.Trial != fixture.materialization.Trial ||
		first.EvaluationPlan != fixture.evaluation.ID ||
		first.Materialization != fixture.materialization.ID || len(first.Arms) != 2 {
		t.Fatalf("derived candidate spine = %+v", first)
	}
	for index, arm := range first.Arms {
		if len(arm.Execution) != 2 || !arm.Evaluation.ID.Valid() {
			t.Fatalf("arm %d graphs = %+v", index, arm)
		}
		resolved := fixture.models[arm.ModelDefinition]
		if arm.Model != resolved.Document.Model || arm.Profile != resolved.Profile.ID ||
			arm.TensorInventory != resolved.Tensors.ID {
			t.Fatalf("arm %d lost resolved model facts: %+v", index, arm)
		}
		want := []artifact.ID{
			fixture.materialization.ID, fixture.evaluation.ID, fixture.ablation.ID,
			fixture.materialization.Code, fixture.materialization.Environment,
			arm.Model, arm.Profile, arm.ModelDefinition, arm.TensorInventory,
			arm.Output, arm.Run, arm.Observation,
		}
		for _, candidate := range arm.Execution {
			assertCandidateRecipeDependencies(t, candidate.Definition, want)
			if candidate.Task == recipe.TaskInference {
				for _, node := range candidate.Definition.Nodes {
					if node.Placement != policy.Placement {
						t.Fatalf("inference placement = %q", node.Placement)
					}
					if node.Module == ModuleCompileDecodePlan && node.Session != policy.Session {
						t.Fatalf("inference session = %q", node.Session)
					}
					if node.Module == ModuleCompileModelPlan && node.Residency != policy.Residency {
						t.Fatalf("inference residency = %q", node.Residency)
					}
				}
			}
		}
		assertCandidateRecipeDependencies(t, arm.Evaluation, want)
		boundExecution := 0
		for _, dependency := range arm.Evaluation.Dependencies {
			if dependency.Role == recipe.DependencyExecutionRecipe {
				boundExecution++
			}
		}
		if boundExecution != len(arm.Execution) || len(arm.Evaluation.Nodes) != 2 ||
			arm.Evaluation.Nodes[0].Module != workflowrecipe.ModuleEvaluateModel ||
			arm.Evaluation.Nodes[1].Module != workflowrecipe.ModuleRecordModel {
			t.Fatalf("evaluation graph = %+v", arm.Evaluation)
		}
		if arm.Evaluation.ID != second.Arms[index].Evaluation.ID {
			t.Fatalf("evaluation identity changed: %s != %s", arm.Evaluation.ID, second.Arms[index].Evaluation.ID)
		}
		for taskIndex := range arm.Execution {
			if arm.Execution[taskIndex].Definition.ID != second.Arms[index].Execution[taskIndex].Definition.ID {
				t.Fatalf("execution identity changed at arm %d task %d", index, taskIndex)
			}
		}
	}
}

func assertCandidateRecipeDependencies(
	t *testing.T,
	definition recipe.Definition,
	want []artifact.ID,
) {
	t.Helper()
	bound := make(map[artifact.ID]bool, len(definition.Dependencies))
	for _, dependency := range definition.Dependencies {
		bound[dependency.Artifact] = true
	}
	for _, id := range want {
		if !bound[id] {
			t.Fatalf("definition %s lost authority %s: %+v", definition.ID, id, definition.Dependencies)
		}
	}
}

func newCandidateRecipeDerivationFixture(t *testing.T) candidateRecipeDerivationFixture {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	primary := derivationResolvedDefinition(t, "candidate-recipe-primary")
	ablated := derivationResolvedDefinition(t, "candidate-recipe-ablation")
	publishDerivationResolvedDefinition(t, store, primary, "primary")
	publishDerivationResolvedDefinition(t, store, ablated, "ablation")

	fact := func(kind artifact.Kind, name string) artifact.Content {
		t.Helper()
		contract := artifact.DocumentContract{
			Kind: kind, MediaType: "application/vnd.overgo.test-candidate-recipe-" + kind.String() + "+json",
			Schema: "overgo/test-candidate-recipe-" + kind.String() + "/v1",
		}
		content, contentErr := contract.ContentBytes([]byte(fmt.Sprintf(`{"name":%q}`, name)))
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		return content
	}
	componentSpec := fact(artifact.KindRecipe, "composition specification")
	code := fact(artifact.KindEvidence, "compiled code revision")
	environment := fact(artifact.KindEvidence, "declared environment")
	developmentSplit := fact(artifact.KindDatasetShard, "development split")
	promotionSplit := fact(artifact.KindDatasetShard, "promotion split")
	developmentBudget := fact(artifact.KindEvidence, "development budget")
	promotionBudget := fact(artifact.KindEvidence, "promotion budget")
	baseline := fact(artifact.KindEvidence, "baseline decision")
	gap := fact(artifact.KindEvidence, "gap decision")
	provenance := fact(artifact.KindEvidence, "component provenance decision")
	resourcePolicy := fact(artifact.KindProfile, "host resource policy")
	primaryRealization := fact(artifact.KindProfile, "primary realization")
	ablatedRealization := fact(artifact.KindProfile, "drop-delta realization")
	authority := fact(artifact.KindEvidence, "admission authority")
	charge := fact(artifact.KindEvidence, "materialization charge")

	prediction := recipe.SteeringPrediction{
		Metric: "held-out quality", Benefit: 0.2, Cost: 2, Unit: "ratio", Uncertainty: 0.1,
	}
	intent, err := NewCandidateEvaluationIntent(
		componentSpec.Descriptor.ID, prediction.Metric, "queries",
		CandidateStopOnBudgetOrNonPositiveIsolatedImprovement,
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	candidate, err := NewCandidate(CandidateSpec{
		Subject: componentSpec.Descriptor.ID, Parent: primary.Document.ID,
		Components: []CandidateComponent{{
			Domain: CandidateComposition, Specification: componentSpec.Descriptor.ID,
		}},
		Prediction: prediction, CostUnit: "queries", Falsifier: intent.ID,
		References: []CandidateReference{
			{Role: CandidateReferenceBaseline, Subject: primary.Document.ID, Evidence: baseline.Descriptor.ID},
			{Role: CandidateReferenceGap, Subject: componentSpec.Descriptor.ID, Evidence: gap.Descriptor.ID},
			{Role: CandidateReferenceProvenance, Subject: componentSpec.Descriptor.ID, Evidence: provenance.Descriptor.ID},
		},
		DevelopmentSplit: developmentSplit.Descriptor.ID, PromotionSplit: promotionSplit.Descriptor.ID,
		DevelopmentBudget: developmentBudget.Descriptor.ID, PromotionBudget: promotionBudget.Descriptor.ID,
		Code: code.Descriptor.ID, Environment: environment.Descriptor.ID,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	admission := runrecord.CandidateAdmission{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Authority: authority.Descriptor.ID,
	}
	admissionContent, err := artifact.JSONContent(artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.CandidateAdmissionMediaType,
		Schema: runrecord.CandidateAdmissionSchema,
	}, admission)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	admission.ID = admissionContent.Descriptor.ID

	ablation, err := candidateAblationCodec.New(CandidateAblation{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Admission: admission.ID,
		Domain: CandidateComposition, Specification: componentSpec.Descriptor.ID,
		Omitted: ablated.Document.ID, Realization: ablatedRealization.Descriptor.ID,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	component, err := candidateComponentPlanCodec.New(CandidateComponentPlan{
		Version: artifact.InitialDocumentVersion, Domain: CandidateComposition,
		Specification: componentSpec.Descriptor.ID, Subject: componentSpec.Descriptor.ID,
		Realization: primaryRealization.Descriptor.ID, ResourcePolicy: resourcePolicy.Descriptor.ID,
		Inputs: []artifact.ID{primary.Document.ID, ablated.Document.ID},
		Documents: []artifact.ID{
			primaryRealization.Descriptor.ID, ablatedRealization.Descriptor.ID, resourcePolicy.Descriptor.ID,
		},
		Ablations: []artifact.ID{ablation.ID}, PeakResidentBytes: 8, ArtifactBytes: 8,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	evaluation, err := candidateEvaluationPlanCodec.New(CandidateEvaluationPlan{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Admission: admission.ID,
		Subject: componentSpec.Descriptor.ID, Parent: primary.Document.ID, Prediction: prediction,
		CostUnit: "queries", Falsifier: intent.ID,
		DevelopmentSplit: developmentSplit.Descriptor.ID, PromotionSplit: promotionSplit.Descriptor.ID,
		DevelopmentBudget: developmentBudget.Descriptor.ID, PromotionBudget: promotionBudget.Descriptor.ID,
		ComponentPlans: []artifact.ID{component.ID}, Ablations: []artifact.ID{ablation.ID},
		CausalReferences: candidate.Spec().References, Code: code.Descriptor.ID,
		Environment: environment.Descriptor.ID,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	trial, err := candidateTrialCodec.New(CandidateTrial{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Admission: admission.ID,
		Subject: componentSpec.Descriptor.ID, Parent: primary.Document.ID,
		EvaluationPlan: evaluation.ID, Components: []artifact.ID{component.ID},
		Ablations: []artifact.ID{ablation.ID}, Code: code.Descriptor.ID, Environment: environment.Descriptor.ID,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	primaryOutput := fact(artifact.KindOutput, "primary output")
	primaryRun := fact(artifact.KindRun, "primary run")
	primaryObservation := fact(artifact.KindEvidence, "primary observation")
	ablatedOutput := fact(artifact.KindOutput, "ablation output")
	ablatedRun := fact(artifact.KindRun, "ablation run")
	ablatedObservation := fact(artifact.KindEvidence, "ablation observation")
	materialization, err := candidateMaterializationCodec.New(CandidateMaterialization{
		Version: artifact.InitialDocumentVersion, Decision: gap.Descriptor.ID,
		Candidate: candidate.ID(), Admission: admission.ID, Trial: trial.ID,
		EvaluationPlan: evaluation.ID, Code: code.Descriptor.ID, Environment: environment.Descriptor.ID,
		ResourceCharge: charge.Descriptor.ID,
		Arms: []CandidateMaterializedArm{
			{
				Component: component.ID, Realization: component.Realization, Model: primary.Document.Model,
				TensorInventory: primary.Tensors.ID, ModelDefinition: primary.Document.ID,
				Output: primaryOutput.Descriptor.ID, Run: primaryRun.Descriptor.ID,
				Observation: primaryObservation.Descriptor.ID,
				TensorBytes: 4, StoredBytes: 4, PeakResidentBytes: 4,
			},
			{
				Component: component.ID, Ablation: ablation.ID, Omitted: ablation.Omitted,
				Realization: ablation.Realization, Model: ablated.Document.Model,
				TensorInventory: ablated.Tensors.ID, ModelDefinition: ablated.Document.ID,
				Output: ablatedOutput.Descriptor.ID, Run: ablatedRun.Descriptor.ID,
				Observation: ablatedObservation.Descriptor.ID,
				TensorBytes: 4, StoredBytes: 4, PeakResidentBytes: 4,
			},
		},
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	contents := []artifact.Content{
		componentSpec, code, environment, developmentSplit, promotionSplit,
		developmentBudget, promotionBudget, baseline, gap, provenance,
		resourcePolicy, primaryRealization, ablatedRealization, authority, charge,
		primaryOutput, primaryRun, primaryObservation, ablatedOutput, ablatedRun, ablatedObservation,
		admissionContent,
		mustCandidateRecipeContent(t, intent), mustCandidateRecipeContent(t, candidate),
		mustCandidateRecipeContent(t, ablation), mustCandidateRecipeContent(t, component),
		mustCandidateRecipeContent(t, evaluation), mustCandidateRecipeContent(t, trial),
		mustCandidateRecipeContent(t, materialization),
	}
	lineage := slices.Clone(intent.Lineage())
	lineage = append(lineage, candidate.Lineage()...)
	lineage = append(lineage, admission.Lineage()...)
	lineage = append(lineage, ablation.Lineage()...)
	lineage = append(lineage, component.Lineage()...)
	lineage = append(lineage, evaluation.Lineage()...)
	lineage = append(lineage, trial.Lineage()...)
	lineage = append(lineage, materialization.Lineage()...)
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "fixture/candidate-recipe-derivation", Contents: contents, Lineage: lineage,
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	return candidateRecipeDerivationFixture{
		store: store, materialization: materialization, evaluation: evaluation, ablation: ablation,
		models: map[artifact.ID]ResolvedModelDefinition{
			primary.Document.ID: primary, ablated.Document.ID: ablated,
		},
	}
}

func publishDerivationResolvedDefinition(
	t *testing.T,
	store artifact.Repository,
	resolved ResolvedModelDefinition,
	name string,
) {
	t.Helper()
	profileContents, profileLineage, err := profilePublicationFacts(resolved.Profile)
	if err != nil {
		t.Fatal(err)
	}
	tensorContent, err := resolved.Tensors.Content()
	if err != nil {
		t.Fatal(err)
	}
	definitionContent, err := resolved.Document.Content()
	if err != nil {
		t.Fatal(err)
	}
	lineage := append(slices.Clone(profileLineage), resolved.Tensors.Lineage()...)
	lineage = append(lineage, resolved.Document.Lineage()...)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:       "fixture/candidate-recipe/model/" + name,
		Artifacts: []artifact.Descriptor{{ID: resolved.Document.Model}},
		Contents:  append(profileContents, tensorContent, definitionContent),
		Lineage:   lineage,
	}); err != nil {
		t.Fatal(err)
	}
}

func mustCandidateRecipeContent(t *testing.T, value any) artifact.Content {
	t.Helper()
	var (
		content artifact.Content
		err     error
	)
	switch document := value.(type) {
	case CandidateEvaluationIntent:
		content, err = document.Content()
	case Candidate:
		content, err = document.Content()
	case CandidateAblation:
		content, err = candidateAblationCodec.Content(document)
	case CandidateComponentPlan:
		content, err = candidateComponentPlanCodec.Content(document)
	case CandidateEvaluationPlan:
		content, err = candidateEvaluationPlanCodec.Content(document)
	case CandidateTrial:
		content, err = candidateTrialCodec.Content(document)
	case CandidateMaterialization:
		content, err = document.Content()
	default:
		t.Fatalf("unsupported candidate recipe fixture document %T", value)
	}
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func TestCandidateRecipeDerivationRejectsInvalidTasks(t *testing.T) {
	fixture := newCandidateRecipeDerivationFixture(t)
	t.Cleanup(func() { _ = fixture.store.Close() })
	base := CandidateRecipePolicy{
		Placement: recipe.PlacementHost, Session: recipe.SessionCapacity,
		Residency: recipe.ResidencyHostReference,
	}
	for name, tasks := range map[string][]recipe.Task{
		"unsupported": {recipe.Task("shell-training")},
		"duplicate":   {recipe.TaskForecast, recipe.TaskForecast},
	} {
		t.Run(name, func(t *testing.T) {
			policy := base
			policy.Tasks = tasks
			_, err := DeriveCandidateRecipes(t.Context(), fixture.store, fixture.materialization.ID, policy)
			if err == nil || name == "unsupported" && !strings.Contains(err.Error(), "unsupported capability task") ||
				name == "duplicate" && !strings.Contains(err.Error(), "duplicate candidate task") {
				t.Fatalf("%s task request = %v", name, err)
			}
		})
	}
}
