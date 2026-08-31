package modelrecipe

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestCrossDomainCandidateIdentityAndValidation(t *testing.T) {
	spec := candidateFixture(t)
	first, err := NewCandidate(spec)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID().Kind() != artifact.KindRecipe || first.ValidateIdentity() != nil {
		t.Fatal("candidate identity is not a valid recipe document")
	}

	permuted := cloneCandidateSpec(spec)
	slices.Reverse(permuted.Components)
	slices.Reverse(permuted.References)
	second, err := NewCandidate(permuted)
	if err != nil {
		t.Fatal(err)
	}
	if second.ID() != first.ID() {
		t.Fatal("candidate identity depends on component or reference input order")
	}

	isolated := first.Spec()
	isolated.Components[0].Specification = candidateID(t, artifact.KindRecipe, "mutated component")
	isolated.References[0].Evidence = candidateID(t, artifact.KindEvidence, "mutated reference")
	if first.Spec().Components[0] == isolated.Components[0] || first.Spec().References[0] == isolated.References[0] {
		t.Fatal("candidate getters expose mutable document storage")
	}

	content, err := first.Content()
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.MediaType != CandidateMediaType || content.Descriptor.Schema != CandidateSchema {
		t.Fatalf("unexpected candidate contract: %+v", content.Descriptor)
	}
	parsedDocument, err := candidateCodec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	parsed := Candidate{document: parsedDocument}
	if parsed.ID() != first.ID() || parsed.ValidateIdentity() != nil {
		t.Fatal("candidate document did not round-trip with exact identity")
	}
	assertCandidateLineage(t, first)

	changes := map[string]func(*CandidateSpec){
		"subject": func(value *CandidateSpec) {
			previous := value.Subject
			value.Subject = candidateID(t, artifact.KindModelDefinition, "other subject")
			for index := range value.References {
				if value.References[index].Subject == previous {
					value.References[index].Subject = value.Subject
				}
			}
		},
		"parent": func(value *CandidateSpec) {
			previous := value.Parent
			value.Parent = candidateID(t, artifact.KindModel, "other parent")
			for index := range value.References {
				if value.References[index].Subject == previous {
					value.References[index].Subject = value.Parent
				}
			}
		},
		"component domain": func(value *CandidateSpec) {
			value.Components[1].Domain = CandidateMechanism
		},
		"component specification": func(value *CandidateSpec) {
			previous := value.Components[1].Specification
			value.Components[1].Specification = candidateID(t, artifact.KindRecipe, "other component")
			for index := range value.References {
				if value.References[index].Subject == previous {
					value.References[index].Subject = value.Components[1].Specification
				}
			}
		},
		"prediction metric":      func(value *CandidateSpec) { value.Prediction.Metric = "held-out exactness" },
		"prediction benefit":     func(value *CandidateSpec) { value.Prediction.Benefit = 0.3 },
		"prediction cost":        func(value *CandidateSpec) { value.Prediction.Cost++ },
		"prediction unit":        func(value *CandidateSpec) { value.Prediction.Unit = "fraction" },
		"cost unit":              func(value *CandidateSpec) { value.CostUnit = "gpu-seconds" },
		"prediction uncertainty": func(value *CandidateSpec) { value.Prediction.Uncertainty = 0.2 },
		"falsifier": func(value *CandidateSpec) {
			value.Falsifier = candidateID(t, artifact.KindRecipe, "other falsifier")
		},
		"reference role":    func(value *CandidateSpec) { value.References[0].Role = CandidateReferenceMeasurement },
		"reference subject": func(value *CandidateSpec) { value.References[0].Subject = value.Subject },
		"reference evidence": func(value *CandidateSpec) {
			value.References[0].Evidence = candidateID(t, artifact.KindEvidence, "other reference")
		},
		"development split": func(value *CandidateSpec) {
			value.DevelopmentSplit = candidateID(t, artifact.KindDatasetShard, "other development split")
		},
		"promotion split": func(value *CandidateSpec) {
			value.PromotionSplit = candidateID(t, artifact.KindDatasetShard, "other promotion split")
		},
		"development budget": func(value *CandidateSpec) {
			value.DevelopmentBudget = candidateID(t, artifact.KindEvidence, "other development budget")
		},
		"promotion budget": func(value *CandidateSpec) {
			value.PromotionBudget = candidateID(t, artifact.KindEvidence, "other promotion budget")
		},
		"code": func(value *CandidateSpec) { value.Code = candidateID(t, artifact.KindEvidence, "other code") },
		"environment": func(value *CandidateSpec) {
			value.Environment = candidateID(t, artifact.KindEvidence, "other environment")
		},
	}
	for name, change := range changes {
		t.Run("identity/"+name, func(t *testing.T) {
			changed := cloneCandidateSpec(spec)
			change(&changed)
			candidate, err := NewCandidate(changed)
			if err != nil {
				t.Fatal(err)
			}
			if candidate.ID() == first.ID() {
				t.Fatal("field change did not change candidate identity")
			}
		})
	}

	rejections := map[string]func(*CandidateSpec){
		"missing subject":          func(value *CandidateSpec) { value.Subject = artifact.ID{} },
		"same parent":              func(value *CandidateSpec) { value.Parent = value.Subject },
		"missing components":       func(value *CandidateSpec) { value.Components = nil },
		"unknown component domain": func(value *CandidateSpec) { value.Components[0].Domain = "unknown" },
		"wrong component kind": func(value *CandidateSpec) {
			value.Components[1].Specification = candidateID(t, artifact.KindEvidence, "wrong component kind")
		},
		"duplicate component": func(value *CandidateSpec) {
			value.Components = append(value.Components, value.Components[0])
		},
		"missing prediction metric": func(value *CandidateSpec) { value.Prediction.Metric = "" },
		"non-finite benefit":        func(value *CandidateSpec) { value.Prediction.Benefit = math.NaN() },
		"zero cost":                 func(value *CandidateSpec) { value.Prediction.Cost = 0 },
		"missing cost unit":         func(value *CandidateSpec) { value.CostUnit = "" },
		"unbounded uncertainty":     func(value *CandidateSpec) { value.Prediction.Uncertainty = 1.1 },
		"wrong falsifier kind": func(value *CandidateSpec) {
			value.Falsifier = candidateID(t, artifact.KindEvidence, "wrong falsifier")
		},
		"missing references": func(value *CandidateSpec) { value.References = nil },
		"irrelevant reference": func(value *CandidateSpec) {
			value.References[0].Subject = candidateID(t, artifact.KindRecipe, "outside candidate")
		},
		"untyped reference": func(value *CandidateSpec) {
			value.References[0].Evidence = candidateID(t, artifact.KindRecipe, "not evidence")
		},
		"duplicate reference": func(value *CandidateSpec) {
			value.References = append(value.References, value.References[0])
		},
		"same split": func(value *CandidateSpec) { value.PromotionSplit = value.DevelopmentSplit },
		"wrong split kind": func(value *CandidateSpec) {
			value.DevelopmentSplit = candidateID(t, artifact.KindDataset, "wrong split")
		},
		"collapsed evidence roles": func(value *CandidateSpec) { value.Code = value.DevelopmentBudget },
		"wrong budget kind": func(value *CandidateSpec) {
			value.DevelopmentBudget = candidateID(t, artifact.KindRecipe, "wrong budget")
		},
		"wrong code kind": func(value *CandidateSpec) {
			value.Code = candidateID(t, artifact.KindRecipe, "wrong code")
		},
		"wrong environment kind": func(value *CandidateSpec) {
			value.Environment = candidateID(t, artifact.KindProfile, "wrong environment")
		},
	}
	for name, change := range rejections {
		t.Run("reject/"+name, func(t *testing.T) {
			invalid := cloneCandidateSpec(spec)
			change(&invalid)
			if _, err := NewCandidate(invalid); err == nil {
				t.Fatal("invalid candidate accepted")
			}
		})
	}

	forged := first
	forged.document.ID = candidateID(t, artifact.KindRecipe, "forged candidate identity")
	if forged.ValidateIdentity() == nil {
		t.Fatal("forged candidate identity accepted")
	}
}

func candidateFixture(t testing.TB) CandidateSpec {
	t.Helper()
	subject := candidateID(t, artifact.KindModelDefinition, "candidate subject")
	parent := candidateID(t, artifact.KindModel, "candidate parent")
	prototype := candidateID(t, artifact.KindRecipe, "prototype component")
	transform := candidateID(t, artifact.KindEvidence, "transform component")
	return CandidateSpec{
		Subject: subject,
		Parent:  parent,
		Components: []CandidateComponent{
			{Domain: CandidateDatasetTransform, Specification: transform},
			{Domain: CandidateModelPrototype, Specification: prototype},
		},
		Prediction: recipePredictionFixture(),
		CostUnit:   "queries",
		Falsifier:  candidateID(t, artifact.KindRecipe, "candidate falsifier"),
		References: []CandidateReference{
			{Role: CandidateReferenceProvenance, Subject: prototype, Evidence: candidateID(t, artifact.KindEvidence, "provenance")},
			{Role: CandidateReferenceGap, Subject: subject, Evidence: candidateID(t, artifact.KindEvidence, "gap")},
			{Role: CandidateReferenceBaseline, Subject: parent, Evidence: candidateID(t, artifact.KindEvidence, "baseline")},
		},
		DevelopmentSplit:  candidateID(t, artifact.KindDatasetShard, "development split"),
		PromotionSplit:    candidateID(t, artifact.KindDatasetShard, "promotion split"),
		DevelopmentBudget: candidateID(t, artifact.KindEvidence, "development budget"),
		PromotionBudget:   candidateID(t, artifact.KindEvidence, "promotion budget"),
		Code:              candidateID(t, artifact.KindEvidence, "code revision"),
		Environment:       candidateID(t, artifact.KindEvidence, "environment"),
	}
}

func recipePredictionFixture() recipe.SteeringPrediction {
	return recipe.SteeringPrediction{
		Metric: "held-out quality", Benefit: 0.2, Cost: 12, Unit: "ratio", Uncertainty: 0.1,
	}
}

func candidateID(t testing.TB, kind artifact.Kind, label string) artifact.ID {
	t.Helper()
	return testutil.ArtifactID(t, kind, label)
}

func assertCandidateLineage(t testing.TB, candidate Candidate) {
	t.Helper()
	spec := candidate.Spec()
	want := map[artifact.ID]bool{
		spec.Subject: true, spec.Parent: true, spec.Falsifier: true,
		spec.DevelopmentSplit: true, spec.PromotionSplit: true,
		spec.DevelopmentBudget: true, spec.PromotionBudget: true,
		spec.Code: true, spec.Environment: true,
	}
	for _, component := range spec.Components {
		want[component.Specification] = true
	}
	for _, reference := range spec.References {
		want[reference.Subject], want[reference.Evidence] = true, true
	}
	for _, edge := range candidate.Lineage() {
		if edge.Child != candidate.ID() || edge.Relation != artifact.RelationDependsOn || !want[edge.Parent] {
			t.Fatalf("unexpected candidate lineage edge: %+v", edge)
		}
		delete(want, edge.Parent)
	}
	if len(want) != 0 {
		t.Fatalf("candidate lineage omitted %d authorities", len(want))
	}
}
