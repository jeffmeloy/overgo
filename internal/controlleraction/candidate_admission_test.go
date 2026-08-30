package controlleraction

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type candidateAdmissionFixture struct {
	store     *overgodb.Store
	action    Action
	spec      modelrecipe.CandidateSpec
	binding   runrecord.AdmissionBinding
	prototype modelrecipe.ModelPrototype
}

type candidateDecisionFixture struct {
	decision recipe.Decision
	source   artifact.Content
}

func TestCrossDomainCandidateAdmissionUsesOneOwner(t *testing.T) {
	fixture := newCandidateAdmissionFixture(t)
	defer fixture.store.Close()

	first, err := CompileTransaction(t.Context(), fixture.store, fixture.action)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileTransaction(t.Context(), fixture.store, fixture.action)
	if err != nil || len(first.Contents) != 1 || len(second.Contents) != 1 || len(first.Aliases) != 0 {
		t.Fatalf("candidate admission batches = (%+v, %+v, %v)", first, second, err)
	}
	for index := range first.Contents {
		if first.Contents[index].Descriptor.ID != second.Contents[index].Descriptor.ID {
			t.Fatal("candidate admission is not deterministic")
		}
	}
	candidateContent, present, err := artifact.ReadContent(t.Context(), fixture.store, fixture.action.Admission.Candidate)
	if err != nil || !present || candidateContent.Descriptor.MediaType != modelrecipe.CandidateMediaType {
		t.Fatalf("persisted candidate = (%+v, %v, %t)", candidateContent.Descriptor, err, present)
	}
	admissionContent := contentWithMediaType(t, first.Contents, runrecord.CandidateAdmissionMediaType)
	var admitted runrecord.CandidateAdmission
	if err := json.Unmarshal(admissionContent.Data, &admitted); err != nil {
		t.Fatal(err)
	}
	if admitted.Candidate != candidateContent.Descriptor.ID || admitted.Authority != fixture.binding.ID {
		t.Fatalf("admission = %+v", admitted)
	}
	for _, forbidden := range []string{"active", "execute", "promote", "runtime", "capability"} {
		if strings.Contains(string(admissionContent.Data), forbidden) {
			t.Fatalf("eligibility document carries %q authority: %s", forbidden, admissionContent.Data)
		}
	}

	unsupportedSpec := cloneCandidateSpec(fixture.spec)
	unsupportedSpec.Components[1].Domain = modelrecipe.CandidateDatasetTransform
	unsupported := publishCandidateAdmissionAction(t, fixture.store, unsupportedSpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, unsupported); err == nil {
		t.Fatal("candidate with no compiled domain adapter was admitted")
	}

	descriptorOnlySpec := cloneCandidateSpec(fixture.spec)
	descriptorOnlyReference := testutil.ArtifactID(t, artifact.KindEvidence, "descriptor-only candidate reference")
	testutil.PublishArtifact(t, fixture.store, descriptorOnlyReference)
	descriptorOnlySpec.References[0].Evidence = descriptorOnlyReference
	descriptorOnly := publishCandidateAdmissionAction(t, fixture.store, descriptorOnlySpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, descriptorOnly); err == nil {
		t.Fatal("descriptor-only candidate reference was admitted")
	}

	descriptorParentSpec := cloneCandidateSpec(fixture.spec)
	descriptorParent := testutil.ArtifactID(t, artifact.KindModel, "descriptor-only candidate parent")
	testutil.PublishArtifact(t, fixture.store, descriptorParent)
	parentDecision := publishCandidateDecision(t, fixture.store, fixture.binding, descriptorParent, "descriptor-parent")
	descriptorParentSpec.Parent = descriptorParent
	for index, reference := range descriptorParentSpec.References {
		if reference.Role == modelrecipe.CandidateReferenceBaseline {
			descriptorParentSpec.References[index].Subject = descriptorParent
			descriptorParentSpec.References[index].Evidence = parentDecision.ID
		}
	}
	descriptorParentAction := publishCandidateAdmissionAction(t, fixture.store, descriptorParentSpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, descriptorParentAction); err == nil {
		t.Fatal("descriptor-only candidate parent was admitted")
	}

	foreignSpec := cloneCandidateSpec(fixture.spec)
	foreignSubject := testutil.ArtifactID(t, artifact.KindModel, "foreign reference subject")
	testutil.PublishArtifact(t, fixture.store, foreignSubject)
	foreignDecision := publishCandidateDecision(t, fixture.store, fixture.binding, foreignSubject, "foreign")
	foreignSpec.References[0].Evidence = foreignDecision.ID
	foreign := publishCandidateAdmissionAction(t, fixture.store, foreignSpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, foreign); err == nil {
		t.Fatal("typed evidence for a foreign subject was admitted")
	}

	wrongRoleSpec := cloneCandidateSpec(fixture.spec)
	replacement := publishCandidateDecision(t, fixture.store, fixture.binding, fixture.prototype.ID(), "replacement-provenance")
	wrongRoleSpec.References[2].Role = modelrecipe.CandidateReferenceBaseline
	wrongRoleSpec.References = append(wrongRoleSpec.References, modelrecipe.CandidateReference{
		Role: modelrecipe.CandidateReferenceProvenance, Subject: fixture.prototype.ID(), Evidence: replacement.ID,
	})
	wrongRole := publishCandidateAdmissionAction(t, fixture.store, wrongRoleSpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, wrongRole); err == nil {
		t.Fatal("baseline role bound to a component subject was admitted")
	}

	for name, budget := range map[string]runrecord.Budget{
		"wrong unit": candidateBudget(t, fixture.spec.DevelopmentSplit, "tokens",
			fixture.spec.Prediction.Cost, fixture.binding.Decider.Identity),
		"insufficient": candidateBudget(t, fixture.spec.DevelopmentSplit,
			fixture.spec.CostUnit, fixture.spec.Prediction.Cost-1, fixture.binding.Decider.Identity),
		"wrong authority": candidateBudget(t, fixture.spec.DevelopmentSplit,
			fixture.spec.CostUnit, fixture.spec.Prediction.Cost, fixture.binding.Proposer.Identity),
	} {
		t.Run(name, func(t *testing.T) {
			content, err := budget.Content()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
				Key: "fixture/candidate-admission/budget/" + budget.ID.String(), Contents: []artifact.Content{content},
				Lineage: budget.Lineage(),
			}); err != nil {
				t.Fatal(err)
			}
			candidateSpec := cloneCandidateSpec(fixture.spec)
			candidateSpec.DevelopmentBudget = budget.ID
			candidate := publishCandidateAdmissionAction(t, fixture.store, candidateSpec, fixture.binding.ID)
			if _, err := CompileTransaction(t.Context(), fixture.store, candidate); err == nil {
				t.Fatal("invalid split grant admitted candidate")
			}
		})
	}

	extraParent := testutil.ArtifactID(t, artifact.KindEvidence, "candidate extra lineage")
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
		Key: "fixture/candidate-admission/extra-lineage", Artifacts: []artifact.Descriptor{{ID: extraParent}},
		Lineage: []artifact.Lineage{{
			Child: fixture.action.Admission.Candidate, Parent: extraParent, Relation: artifact.RelationDependsOn,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := CompileTransaction(t.Context(), fixture.store, fixture.action); err == nil {
		t.Fatal("candidate with non-canonical added lineage was admitted")
	}
}

func TestModelPrototypeAdmissionContract(t *testing.T) {
	fixture := newCandidateAdmissionFixture(t)
	defer fixture.store.Close()
	batch, err := CompileTransaction(t.Context(), fixture.store, fixture.action)
	if err != nil || len(batch.Aliases) != 0 {
		t.Fatalf("exact compiled prototype admission = (%+v, %v)", batch, err)
	}

	wrongContractSpec := cloneCandidateSpec(fixture.spec)
	wrongContractContent := candidateTypedDocument(t, artifact.KindRecipe, "wrong prototype contract", "wrong-prototype")
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
		Key: "fixture/candidate-admission/wrong-prototype", Contents: []artifact.Content{wrongContractContent},
	}); err != nil {
		t.Fatal(err)
	}
	wrongContractSpec.Components[0].Specification = wrongContractContent.Descriptor.ID
	for index, reference := range wrongContractSpec.References {
		if reference.Subject == fixture.prototype.ID() {
			decision := publishCandidateDecision(
				t, fixture.store, fixture.binding, wrongContractContent.Descriptor.ID, "wrong-contract",
			)
			wrongContractSpec.References[index].Subject = wrongContractContent.Descriptor.ID
			wrongContractSpec.References[index].Evidence = decision.ID
		}
	}
	wrongContract := publishCandidateAdmissionAction(t, fixture.store, wrongContractSpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, wrongContract); err == nil {
		t.Fatal("typed recipe under the model-prototype domain was admitted")
	}

	descriptorOnlySpec := cloneCandidateSpec(fixture.spec)
	missing := testutil.ArtifactID(t, artifact.KindRecipe, "descriptor-only model prototype")
	testutil.PublishArtifact(t, fixture.store, missing)
	descriptorOnlySpec.Components[0].Specification = missing
	for index, reference := range descriptorOnlySpec.References {
		if reference.Subject == fixture.prototype.ID() {
			decision := publishCandidateDecision(t, fixture.store, fixture.binding, missing, "descriptor-prototype")
			descriptorOnlySpec.References[index].Subject = missing
			descriptorOnlySpec.References[index].Evidence = decision.ID
		}
	}
	descriptorOnly := publishCandidateAdmissionAction(t, fixture.store, descriptorOnlySpec, fixture.binding.ID)
	if _, err := CompileTransaction(t.Context(), fixture.store, descriptorOnly); err == nil {
		t.Fatal("descriptor-only model prototype was admitted")
	}
}

func newCandidateAdmissionFixture(t *testing.T) candidateAdmissionFixture {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		store.Close()
		t.Fatal(err)
	}
	names := model.SupportedArchitectures()
	if len(names) == 0 {
		store.Close()
		t.Fatal("compiled architecture catalog is empty")
	}
	profile, err := modelrecipe.ResolveRegisteredArchitectureProfile(ctx, store, names[0])
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	prototype, err := modelrecipe.NewModelPrototype(profile)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	subjectContent := candidateTypedDocument(t, artifact.KindModelDefinition, "candidate subject", "subject")
	parentContent := candidateTypedDocument(t, artifact.KindModel, "candidate parent", "parent")
	subject := subjectContent.Descriptor.ID
	parent := parentContent.Descriptor.ID
	developmentSplit := id(artifact.KindDatasetShard, "candidate development split")
	promotionSplit := id(artifact.KindDatasetShard, "candidate promotion split")
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation:   0,
		Proposer:     runrecord.AuthorityDomain{Name: "candidate-proposer", Identity: id(artifact.KindEvidence, "candidate proposer")},
		Evaluator:    runrecord.AuthorityDomain{Name: "candidate-evaluator", Identity: id(artifact.KindEvidence, "candidate evaluator")},
		Decider:      runrecord.AuthorityDomain{Name: "candidate-decider", Identity: id(artifact.KindEvidence, "candidate decider")},
		SealedInputs: promotionSplit,
		CleanWorker:  id(artifact.KindEvidence, "candidate clean worker"),
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	code, err := runrecord.NewCodeRevision(strings.Repeat("a", 40))
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "candidate-host", OS: "windows", Arch: "amd64", Device: "cpu",
		Backend: "go", Driver: "process", Runtime: "go-test",
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	prediction := recipe.SteeringPrediction{
		Metric: "held-out quality", Benefit: 0.2, Cost: 3, Unit: "ratio", Uncertainty: 0.1,
	}
	development := candidateBudget(t, developmentSplit, "queries", prediction.Cost, binding.Decider.Identity)
	promotion := candidateBudget(t, promotionSplit, "queries", prediction.Cost, binding.Decider.Identity)
	falsifier := candidateTypedDocument(t, artifact.KindRecipe, "candidate-falsifier", "falsifier")
	decisionFixtures := []candidateDecisionFixture{
		newCandidateDecision(t, binding, parent, "baseline"),
		newCandidateDecision(t, binding, subject, "gap"),
		newCandidateDecision(t, binding, prototype.ID(), "prototype-provenance"),
		newCandidateDecision(t, binding, code.ID, "code-provenance"),
	}
	decisions := make([]recipe.Decision, len(decisionFixtures))
	for index := range decisionFixtures {
		decisions[index] = decisionFixtures[index].decision
	}
	references := []modelrecipe.CandidateReference{
		{Role: modelrecipe.CandidateReferenceBaseline, Subject: parent, Evidence: decisions[0].ID},
		{Role: modelrecipe.CandidateReferenceGap, Subject: subject, Evidence: decisions[1].ID},
		{Role: modelrecipe.CandidateReferenceProvenance, Subject: prototype.ID(), Evidence: decisions[2].ID},
		{Role: modelrecipe.CandidateReferenceProvenance, Subject: code.ID, Evidence: decisions[3].ID},
	}
	spec := modelrecipe.CandidateSpec{
		Subject: subject, Parent: parent,
		Components: []modelrecipe.CandidateComponent{
			{Domain: modelrecipe.CandidateModelPrototype, Specification: prototype.ID()},
			{Domain: modelrecipe.CandidateCode, Specification: code.ID},
		},
		Prediction: prediction, CostUnit: "queries", Falsifier: falsifier.Descriptor.ID, References: references,
		DevelopmentSplit: developmentSplit, PromotionSplit: promotionSplit,
		DevelopmentBudget: development.ID, PromotionBudget: promotion.ID,
		Code: code.ID, Environment: environment.ID,
	}
	contents := []artifact.Content{subjectContent, parentContent, mustCandidateDocumentContent(t, prototype), mustCandidateDocumentContent(t, development), mustCandidateDocumentContent(t, promotion),
		mustCandidateDocumentContent(t, binding), mustCodeRevisionContent(t, code), mustCandidateDocumentContent(t, environment), falsifier}
	lineage := append(prototype.Lineage(), development.Lineage()...)
	lineage = append(lineage, promotion.Lineage()...)
	lineage = append(lineage, artifact.Lineage{Child: falsifier.Descriptor.ID, Parent: subject, Relation: artifact.RelationDependsOn})
	for index, decision := range decisions {
		content, contentErr := decision.Content()
		if contentErr != nil {
			store.Close()
			t.Fatal(contentErr)
		}
		contents = append(contents, decisionFixtures[index].source, content)
		lineage = append(lineage, decision.Lineage()...)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "fixture/candidate-admission",
		Artifacts: []artifact.Descriptor{
			{ID: developmentSplit}, {ID: promotionSplit},
			{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity}, {ID: binding.Decider.Identity}, {ID: binding.CleanWorker},
		},
		Contents: contents, Lineage: lineage,
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	action := publishCandidateAdmissionAction(t, store, spec, binding.ID)
	return candidateAdmissionFixture{store: store, action: action, spec: spec, binding: binding, prototype: prototype}
}

func cloneCandidateSpec(source modelrecipe.CandidateSpec) modelrecipe.CandidateSpec {
	source.Components = append([]modelrecipe.CandidateComponent(nil), source.Components...)
	source.References = append([]modelrecipe.CandidateReference(nil), source.References...)
	return source
}

func publishCandidateAdmissionAction(
	t *testing.T, store *overgodb.Store, spec modelrecipe.CandidateSpec, authority artifact.ID,
) Action {
	t.Helper()
	proposal, err := CompileTransaction(t.Context(), store, Action{
		Version: ActionVersion, Kind: KindCandidateProposal, Candidate: &CandidateProposalAction{Spec: spec},
	})
	if err != nil || len(proposal.Contents) != 1 || proposal.Contents[0].Descriptor.MediaType != modelrecipe.CandidateMediaType {
		t.Fatalf("candidate proposal = (%+v, %v)", proposal, err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:      "fixture/candidate-proposal/" + proposal.Contents[0].Descriptor.ID.String(),
		Contents: proposal.Contents, Lineage: proposal.Lineage,
	}); err != nil {
		t.Fatal(err)
	}
	return Action{Version: ActionVersion, Kind: KindCandidateAdmission, Admission: &CandidateAdmissionAction{
		Candidate: proposal.Contents[0].Descriptor.ID, Authority: authority,
	}}
}

func newCandidateDecision(t *testing.T, binding runrecord.AdmissionBinding, subject artifact.ID, label string) candidateDecisionFixture {
	t.Helper()
	source := candidateTypedDocument(t, artifact.KindEvidence, label, "measurement")
	decision, err := recipe.NewDecision(
		subject, recipe.DecisionObserved, recipe.EvidenceVerified, "",
		recipe.Decider{CodeCommit: strings.Repeat("b", 40), Derivation: binding.Evaluator.Identity},
		[]artifact.ID{source.Descriptor.ID},
	)
	if err != nil {
		t.Fatal(err)
	}
	return candidateDecisionFixture{decision: decision, source: source}
}

func publishCandidateDecision(
	t *testing.T, store *overgodb.Store, binding runrecord.AdmissionBinding, subject artifact.ID, label string,
) recipe.Decision {
	t.Helper()
	source := candidateTypedDocument(t, artifact.KindEvidence, label, "measurement")
	decision, err := recipe.NewDecision(
		subject, recipe.DecisionObserved, recipe.EvidenceVerified, "",
		recipe.Decider{CodeCommit: strings.Repeat("c", 40), Derivation: binding.Evaluator.Identity},
		[]artifact.ID{source.Descriptor.ID},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:      "fixture/candidate-admission/decision/" + decision.ID.String(),
		Contents: []artifact.Content{source, content}, Lineage: decision.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	return decision
}

func candidateTypedDocument(t *testing.T, kind artifact.Kind, label, schema string) artifact.Content {
	t.Helper()
	body := struct {
		Label string `json:"label"`
	}{Label: label}
	id, err := artifact.JSONID(kind, body)
	if err != nil {
		t.Fatal(err)
	}
	contract := artifact.DocumentContract{
		Kind: kind, MediaType: "application/vnd.overgo.test-candidate-" + schema + "+json", Schema: "overgo/test-candidate-" + schema + "/v1",
	}
	content, err := contract.ContentJSON(id, body)
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func candidateBudget(t *testing.T, split artifact.ID, unit string, issued uint64, authority artifact.ID) runrecord.Budget {
	t.Helper()
	value, err := runrecord.NewBudget(unit, split, issued, authority)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func contentWithMediaType(t *testing.T, contents []artifact.Content, mediaType string) artifact.Content {
	t.Helper()
	for _, content := range contents {
		if content.Descriptor.MediaType == mediaType {
			return content
		}
	}
	t.Fatalf("content %q is absent", mediaType)
	return artifact.Content{}
}

type candidateContentDocument interface {
	Content() (artifact.Content, error)
}

func mustCandidateDocumentContent(t *testing.T, value candidateContentDocument) artifact.Content {
	t.Helper()
	content, err := value.Content()
	if err != nil {
		t.Fatalf("candidate fixture content: %v", err)
	}
	if content.Descriptor.ID.Kind() == artifact.KindInvalid {
		t.Fatal("candidate fixture content has no identity")
	}
	return content
}

func mustCodeRevisionContent(t *testing.T, value runrecord.CodeRevision) artifact.Content {
	t.Helper()
	batch, err := value.Publication()
	if err != nil || len(batch.Contents) != 1 {
		t.Fatalf("code revision content = (%+v, %v)", batch, err)
	}
	return batch.Contents[0]
}
