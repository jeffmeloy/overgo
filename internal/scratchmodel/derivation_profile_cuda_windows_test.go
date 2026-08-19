//go:build windows

package scratchmodel

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

func TestDerivationProfilePromotedByDescendantQuality(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	candidate := AdaptiveDerivationProfile()
	incumbent := candidate
	incumbent.MuonMomentum = candidate.MuonMomentum / 2
	var err error
	incumbent, err = NewDerivationProfile(incumbent)
	if err != nil {
		t.Fatal(err)
	}
	content, err := candidate.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := derivationProfileCodec.Parse(content.Data)
	if err != nil || parsed.ID != candidate.ID {
		t.Fatalf("profile round trip = (%s, %v)", parsed.ID, err)
	}

	facts := CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: 100}
	incumbentLoss, incumbentModel, _, err := trainProfile(facts, incumbent)
	if err != nil {
		t.Fatal(err)
	}
	candidateLoss, candidateModel, construction, err := trainProfile(facts, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if candidateLoss >= incumbentLoss {
		t.Fatalf("candidate loss %.9f did not beat incumbent %.9f", candidateLoss, incumbentLoss)
	}

	promotionSplit := profileArtifact(t, artifact.KindDatasetShard, "profile-promotion")
	code := profileArtifact(t, artifact.KindEvidence, "profile-code")
	proposer := profileArtifact(t, artifact.KindEvidence, "profile-proposer")
	proposal, err := trainingprogram.CompileImprovementProposal(trainingprogram.ImprovementSpec{
		Kind: trainingprogram.ImprovementDerivationProfile, ParentModel: incumbentModel,
		Incumbent: incumbent.ID, Candidate: candidate.ID, Dataset: construction.Dataset(), DevelopmentSplit: construction.SplitID(),
		Recipe: construction.Program().ID(), Code: code, Proposer: proposer,
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluator := profileArtifact(t, artifact.KindEvidence, "profile-evaluator")
	authority := profileArtifact(t, artifact.KindEvidence, "profile-authority")
	admission, err := runrecord.AdmitImprovement(proposal, authority, evaluator, promotionSplit)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewRun(construction.Program().ID(), runrecord.OutcomeSucceeded,
		[]artifact.ID{candidateModel, construction.Dataset(), promotionSplit, code, admission.ID}, []artifact.ID{candidateModel}, "")
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := runrecord.NewEvaluation(construction.Program().ID(), run.ID, construction.Dataset(), []runrecord.Metric{{
		Name: "causal-loss", Value: candidateLoss, Direction: runrecord.DirectionMinimize,
	}})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := runrecord.DecideImprovement(admission, candidateModel, run, evaluation,
		profileArtifact(t, artifact.KindEvidence, "profile-decider"), runrecord.ImprovementPromote)
	if err != nil || decision.State != runrecord.ImprovementPromote || decision.Rollback != incumbentModel {
		t.Fatalf("profile promotion = (%s, %s, %v)", decision.State, decision.Rollback, err)
	}
}

func trainProfile(facts CorpusFacts, profile DerivationProfile) (float64, artifact.ID, Construction, error) {
	construction, err := Compile(facts, profile)
	if err != nil {
		return 0, artifact.ID{}, Construction{}, err
	}
	trainer, err := NewResidentTrainer(construction, facts.Steps)
	if err != nil {
		return 0, artifact.ID{}, Construction{}, err
	}
	defer trainer.Close()
	train := construction.Split().Train
	for step := 1; step <= facts.Steps; step++ {
		tokens, tokenErr := construction.Tokens(train[(step-1)%len(train)])
		if tokenErr != nil {
			return 0, artifact.ID{}, Construction{}, tokenErr
		}
		if _, stepErr := trainer.Step(tokens, step); stepErr != nil {
			return 0, artifact.ID{}, Construction{}, stepErr
		}
	}
	validation := construction.Split().Validation
	loss := 0.0
	for _, document := range validation {
		tokens, tokenErr := construction.Tokens(document)
		if tokenErr != nil {
			return 0, artifact.ID{}, Construction{}, tokenErr
		}
		value, evaluateErr := trainer.Evaluate(tokens)
		if evaluateErr != nil {
			return 0, artifact.ID{}, Construction{}, evaluateErr
		}
		loss += value
	}
	weights, _, _, err := trainer.Snapshot()
	if err != nil {
		return 0, artifact.ID{}, Construction{}, err
	}
	model, err := construction.IdentifyTrainedModel(weights)
	return loss / float64(len(validation)), model, construction, err
}

func profileArtifact(t *testing.T, kind artifact.Kind, label string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(fmt.Sprintf("%s/%s", t.Name(), label)))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
