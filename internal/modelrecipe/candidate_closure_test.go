package modelrecipe

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// candidateOrderSeeds is how many label sets the declared orders are tried over: an order that compares sorted lists with
// declared ones passes on the sets whose hashes happen to fall in declared order.
const candidateOrderSeeds = 16

// TestACandidateValidatesInEveryDeclaredOrder: a trial and its plan hold their lists sorted by hash and a compilation holds
// them as declared, so two components, and two ablations of one, must validate however they were declared.
func TestACandidateValidatesInEveryDeclaredOrder(t *testing.T) {
	t.Parallel()
	for seed := range candidateOrderSeeds {
		id := func(kind artifact.Kind, name string) artifact.ID {
			return testutil.ArtifactID(t, kind, fmt.Sprintf("%s-%d", name, seed))
		}
		facts := CandidateCompileFacts{
			Candidate: id(artifact.KindRecipe, "candidate"), Admission: id(artifact.KindEvidence, "admission"),
			Subject: id(artifact.KindModel, "subject"), Parent: id(artifact.KindModel, "parent"),
			Prediction: recipePredictionFixture(), CostUnit: "steps", Falsifier: id(artifact.KindRecipe, "falsifier"),
			DevelopmentSplit: id(artifact.KindDatasetShard, "development"), PromotionSplit: id(artifact.KindDatasetShard, "promotion"),
			DevelopmentBudget: id(artifact.KindEvidence, "development budget"), PromotionBudget: id(artifact.KindEvidence, "promotion budget"),
			Code: id(artifact.KindEvidence, "code"), Environment: id(artifact.KindEvidence, "environment"),
			References: []CandidateReference{{Role: CandidateReferenceBaseline, Subject: id(artifact.KindModel, "baseline"), Evidence: id(artifact.KindEvidence, "reading")}},
		}
		component := func(name string, deltas int) (CandidateComponentPlan, []CandidateAblation) {
			specification := id(artifact.KindRecipe, name)
			var ablations []CandidateAblation
			var ablationIDs []artifact.ID
			for delta := range deltas {
				ablation, err := candidateAblationCodec.New(CandidateAblation{
					Version: artifact.InitialDocumentVersion, Candidate: facts.Candidate, Admission: facts.Admission,
					Domain: CandidateMechanism, Specification: specification,
					Omitted:     id(artifact.KindRecipe, fmt.Sprintf("%s omitted %d", name, delta)),
					Realization: id(artifact.KindProfile, fmt.Sprintf("%s without %d", name, delta)),
				})
				testcheck.NoError(t, err)
				ablations, ablationIDs = append(ablations, ablation), append(ablationIDs, ablation.ID)
			}
			realization, policy := id(artifact.KindProfile, name+" realization"), id(artifact.KindProfile, name+" policy")
			plan, err := candidateComponentPlanCodec.New(CandidateComponentPlan{
				Version: artifact.InitialDocumentVersion, Domain: CandidateMechanism, Specification: specification,
				Subject: facts.Subject, Realization: realization, ResourcePolicy: policy,
				Inputs: []artifact.ID{id(artifact.KindModel, name+" input")}, Documents: []artifact.ID{realization, policy},
				Ablations: ablationIDs, PeakResidentBytes: 1, ArtifactBytes: 1,
			})
			testcheck.NoError(t, err)
			return plan, ablations
		}
		one, oneAblations := component("one", 1)
		two, twoAblations := component("two", 2)
		for _, components := range [][]CandidateComponentPlan{{one, two}, {two, one}} {
			for _, ablations := range [][]CandidateAblation{
				{oneAblations[0], twoAblations[0], twoAblations[1]}, {twoAblations[1], twoAblations[0], oneAblations[0]},
				{twoAblations[0], oneAblations[0], twoAblations[1]},
			} {
				plan, err := newCandidateEvaluationPlan(facts, components, ablations)
				testcheck.NoError(t, err)
				trial, err := newCandidateTrial(facts, plan.ID, components, ablations)
				testcheck.NoError(t, err)
				compilation := CandidateCompilation{Trial: trial, EvaluationPlan: plan, Components: components, Ablations: ablations}
				if err := compilation.ValidateIdentity(); err != nil {
					t.Fatalf("seed %d: %v", seed, err)
				}
			}
		}
	}
}

// TestAnUnloadableRecipeIsNotRunnable: a candidate's closure answers for what would run. An inference recipe with no forward
// stage passes the module graph and cannot load, so it is not runnable, and a representation bridge, whose stages only this
// package's catalog holds, is.
func TestAnUnloadableRecipeIsNotRunnable(t *testing.T) {
	t.Parallel()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "unloadable")
	whole, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	testcheck.NoError(t, err)
	if closure := ValidateCandidateRecipe(whole); !closure.Runnable {
		t.Fatalf("a whole recipe: %+v", closure)
	}
	compile, decode := whole.Nodes[0], whole.Nodes[1]
	partial, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference, whole.Dependencies, []recipe.Node{compile, decode},
		[]recipe.Edge{{From: recipe.Endpoint{Node: compile.ID, Port: "plan"}, To: recipe.Endpoint{Node: decode.ID, Port: "plan"}}},
		nil, []recipe.Output{{Name: "session", Data: recipe.DataSessionPlan, Source: recipe.Endpoint{Node: decode.ID, Port: "session"}}},
	)
	testcheck.NoError(t, err)
	if closure := ValidateCandidateRecipe(partial); closure.Runnable {
		t.Fatalf("a recipe with no forward stage reads as runnable: %+v", closure)
	}
	bridge, err := (RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementDevice, TargetPlacement: recipe.PlacementDevice,
		SourceResidency: recipe.ResidencyDeviceF32, TargetResidency: recipe.ResidencyDeviceF32,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(modelID, testutil.ArtifactID(t, artifact.KindModel, "target"), testutil.ArtifactID(t, artifact.KindProfile, "source contract"),
		testutil.ArtifactID(t, artifact.KindProfile, "target contract"), testutil.ArtifactID(t, artifact.KindAdapter, "bridge"))
	testcheck.NoError(t, err)
	if closure := ValidateCandidateRecipe(bridge); !closure.Runnable {
		t.Fatalf("a representation bridge: %+v", closure)
	}
}
