package evaluation

import (
	"context"
	"errors"
	"runtime"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type productionProbeFixture struct {
	store      *overgodb.Store
	request    CapabilityProbeEvidence
	testCase   ActivationCase
	profile    ActivationProfile
	receipt    runrecord.StageReceipt
	alias      string
	model      artifact.ID
	definition recipe.Definition
	executed   string
}

func TestProductionCapabilityProbeBindsExecutionPath(t *testing.T) {
	fixture := newProductionProbeFixture(t, "binds-production-path")
	result, commit, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	if !commit.Valid() || result.Entry != ProductionEntryLocalPlacement ||
		result.Case != fixture.testCase.ID || result.Profile != fixture.profile.ID ||
		result.Model != fixture.model || result.Recipe != fixture.definition.ID ||
		result.Selection != fixture.request.Placement.Selection.Identity ||
		result.Placement != fixture.request.Placement.ID ||
		result.StageReceipt != fixture.receipt.ID || result.Outcome != runrecord.OutcomeSucceeded ||
		fixture.executed == "" || fixture.receipt.Attempt != 1 {
		t.Fatalf("probe result = %+v", result)
	}
	loaded, err := RequireCapabilityProbeResult(t.Context(), fixture.store, result.ID)
	if err != nil || loaded.ID != result.ID || loaded.Resources.Scope.Workload != fixture.definition.ID ||
		loaded.Observation != fixture.request.Observation || loaded.CheckDecision != fixture.request.CheckDecision {
		t.Fatalf("loaded probe = (%+v, %v)", loaded, err)
	}
	parents, err := fixture.store.Parents(t.Context(), result.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []artifact.ID{
		fixture.testCase.ID, fixture.profile.ID, fixture.model,
		fixture.definition.ID, fixture.request.Placement.Capability.ID,
		fixture.request.Environment, fixture.request.Trace, fixture.receipt.ID,
		fixture.request.CheckDecision, fixture.request.Observation,
	} {
		if !slices.ContainsFunc(parents, func(edge artifact.Lineage) bool { return edge.Parent == required }) {
			t.Fatalf("probe lineage omits %s: %+v", required, parents)
		}
	}
}

func TestProductionCapabilityProbeRejectsStaleOrSyntheticEvidence(t *testing.T) {
	t.Run("stale terminal receipt", func(t *testing.T) {
		fixture := newProductionProbeFixture(t, "stale-stage")
		if _, err := runrecord.PublishStageReceipt(t.Context(), fixture.store, runrecord.StageReceipt{
			Recipe: fixture.receipt.Recipe, Node: fixture.receipt.Node,
			Operation: fixture.receipt.Operation, Attempt: fixture.receipt.Attempt + 1,
			State: runrecord.StageAdmitted, Inputs: fixture.receipt.Inputs,
		}, nil, nil); err != nil {
			t.Fatal(err)
		}
		if _, _, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, fixture.request); err == nil {
			t.Fatal("stale terminal stage receipt was admitted")
		}
	})

	t.Run("stale model selection", func(t *testing.T) {
		fixture := newProductionProbeFixture(t, "stale-selection")
		previous := fixture.model
		if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
			Key: "probe/retire-selection",
			Aliases: []artifact.AliasBinding{{
				Name: fixture.alias, Target: fixture.model, Previous: &previous, Remove: true,
			}},
		}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, fixture.request); err == nil {
			t.Fatal("stale model selection was admitted")
		}
	})

	t.Run("descriptor-only trace", func(t *testing.T) {
		fixture := newProductionProbeFixture(t, "synthetic-trace")
		fixture.request.Trace = testutil.ArtifactID(t, artifact.KindEvidence, "descriptor-only-probe-trace")
		testutil.PublishArtifact(t, fixture.store, fixture.request.Trace)
		if _, _, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, fixture.request); err == nil {
			t.Fatal("descriptor-only synthetic trace was admitted")
		}
	})

	t.Run("caller cannot synthesize a resource observation", func(t *testing.T) {
		fixture := newProductionProbeFixture(t, "synthetic-resource")
		fixture.request.Observation = testutil.ArtifactID(t, artifact.KindEvidence, "descriptor-only-serving-observation")
		testutil.PublishArtifact(t, fixture.store, fixture.request.Observation)
		if _, _, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, fixture.request); err == nil {
			t.Fatal("descriptor-only resource observation was admitted")
		}
	})

	t.Run("cross case evidence", func(t *testing.T) {
		fixture := newProductionProbeFixture(t, "cross-case")
		otherInput := testutil.ArtifactID(t, artifact.KindDataset, "other activation input")
		otherCheck := testutil.ArtifactID(t, artifact.KindProfile, "other activation check")
		if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
			Key:       "probe/other-case-parents",
			Artifacts: []artifact.Descriptor{{ID: otherInput}, {ID: otherCheck}},
		}); err != nil {
			t.Fatal(err)
		}
		other, err := NewActivationCase(ActivationCase{
			Name: "other.production.case", Task: recipe.TaskGeneration,
			Contract: fixture.definition.ID, Input: otherInput, EvidenceCheck: otherCheck,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := PublishActivationCaseRegistry(t.Context(), fixture.store, []ActivationCase{fixture.testCase, other}); err != nil {
			t.Fatal(err)
		}
		fixture.request.Case = other.ID
		if _, _, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, fixture.request); err == nil {
			t.Fatal("evidence bound to another activation case was admitted")
		}
	})
}

func newProductionProbeFixture(t *testing.T, name string) productionProbeFixture {
	t.Helper()
	return newProductionProbeFixtureInStore(t, newCapabilityEvaluationStore(t), name)
}

func newCapabilityEvaluationStore(t *testing.T) *overgodb.Store {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close capability evaluation store: %v", err)
		}
	})
	return store
}

func newProductionProbeFixtureInStore(t *testing.T, store *overgodb.Store, name string) productionProbeFixture {
	t.Helper()
	ctx := t.Context()

	weights := []byte(name + "-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "probe/model/" + name,
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskGeneration, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "probe/candidate/"+name, definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "probe/verification/"+name, definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceParity, "production probe fixture"); err != nil {
		t.Fatal(err)
	}
	alias := "probe/capability/" + name
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "probe/alias/" + name,
		Aliases: []artifact.AliasBinding{{Name: alias, Target: manifest.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	selection, err := modelrecipe.ResolveCapabilityEvidenceSelector(ctx, store, modelrecipe.CapabilityEvidenceSelector{
		Alias: alias, Task: recipe.TaskGeneration, Session: modelrecipe.SessionWarm,
	})
	if err != nil {
		t.Fatal(err)
	}
	schema := testutil.ArtifactID(t, artifact.KindProfile, name+"-capability-schema")
	testutil.PublishArtifact(t, store, schema)
	capability, err := (runrecord.CapabilityIdentity{
		Implementation: manifest.ID, Release: "1.0.0",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportBuiltin, Protocol: "probe-model/1"},
		Schema:    schema,
		Platform:  runrecord.CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 4096, MaxConcurrent: 1,
			CPUThreads: 1, HostBytes: selection.Resources.ArtifactBytes,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	capabilityContent, err := capability.Content()
	if err != nil {
		t.Fatal(err)
	}
	capabilityBatch, err := artifact.NewDocumentBatch(
		"probe/model-capability/"+name,
		[]artifact.Content{capabilityContent}, capability.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, capabilityBatch); err != nil {
		t.Fatal(err)
	}
	placement, err := capabilityruntime.ResolveExactCapabilityPlacement(ctx, store, capability.ID, selection)
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "probe-host-" + name, OS: runtime.GOOS, Arch: runtime.GOARCH,
		Device: "host", Backend: "go", Driver: "production-probe", Runtime: runtime.Version(),
	})
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := environment.Batch("probe/environment/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, environmentBatch); err != nil {
		t.Fatal(err)
	}

	input := testutil.ArtifactID(t, artifact.KindDataset, name+"-activation-input")
	check := testutil.ArtifactID(t, artifact.KindProfile, name+"-activation-check")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "probe/case-parents/" + name,
		Artifacts: []artifact.Descriptor{{ID: input}, {ID: check}},
	}); err != nil {
		t.Fatal(err)
	}
	testCase, err := NewActivationCase(ActivationCase{
		Name: name + ".production", Task: recipe.TaskGeneration,
		Contract: definition.ID, Input: input, EvidenceCheck: check,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishActivationCaseRegistry(ctx, store, []ActivationCase{testCase}); err != nil {
		t.Fatal(err)
	}
	profile, err := NewActivationProfile(ActivationProfile{
		Name: name + ".local", Kind: ActivationProfileLocalModel,
		Capability: capability.ID, Tasks: []recipe.Task{recipe.TaskGeneration},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishActivationMatrix(ctx, store, []ActivationProfile{profile}, nil); err != nil {
		t.Fatal(err)
	}

	stages := program.Stages()
	if len(stages) == 0 {
		t.Fatal("compiled capability has no stages")
	}
	const executorName = "production-probe"
	catalog := capabilityruntime.ExecutorCatalog{
		modelrecipe.ModuleThoughtBankGenerate: capabilityruntime.JSONScalar[string, struct{}, string](
			executorName,
			func(value string) error {
				if value == "" {
					return errors.New("probe input is empty")
				}
				return nil
			},
			func(context.Context, artifact.Repository, string, recipe.Program, string) (struct{}, error) {
				return struct{}{}, nil
			},
			func(runtime *workflowruntime.Runtime, bound artifact.ID, _ struct{}) error {
				return workflowruntime.RegisterJSONStage[string, string](
					runtime, modelrecipe.ModuleThoughtBankGenerate, bound,
					artifact.JSONContract(artifact.KindOutput, "overgo.production-probe-output/v1"),
					func(value string) (string, error) { return name + ":" + value, nil },
				)
			},
		),
	}
	const raw = `"probe request"`
	started := time.Now()
	executed, err := catalog.ExecutePlaced(ctx, store, "model", placement, raw)
	executionWall := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	measured, ok := executed.(capabilityruntime.Measured)
	if !ok {
		t.Fatalf("placed execution = %T, want measured production result", executed)
	}
	output, ok := measured.Output.(string)
	if !ok || output == "" {
		t.Fatalf("placed output = %#v", measured.Output)
	}
	inputContent, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindFile, "overgo."+executorName+"-input.v1"), "probe request",
	)
	if err != nil {
		t.Fatal(err)
	}
	operation, err := workflowruntime.ExecutionID(
		definition.ID,
		"recipe/run/"+definition.ID.String()+"/"+inputContent.Descriptor.ID.String(),
	)
	if err != nil {
		t.Fatal(err)
	}
	receipt, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, stages[0].Node.ID)
	if err != nil || !found || receipt.State != runrecord.StageCompleted {
		t.Fatalf("placed stage receipt = %+v, found=%t err=%v", receipt, found, err)
	}
	interaction, err := runrecord.PublishInteraction(ctx, store, runrecord.Interaction{
		Response: name + "-response", Recipe: definition.ID, Model: manifest.ID,
		Node: stages[0].Node.ID, Operation: operation, Run: receipt.ID,
	}, []runrecord.InteractionMessage{
		{Role: "user", Content: "probe request"},
		{Role: "assistant", Content: output},
	})
	if err != nil {
		t.Fatal(err)
	}
	var measuredNS uint64
	for _, phase := range measured.Phases {
		measuredNS += phase.WallNS
	}
	if observed := uint64(executionWall.Nanoseconds()); observed > measuredNS {
		measuredNS = observed
	}
	observation, err := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
		Model: manifest.ID, Recipe: definition.ID, Environment: environment.ID,
		Operation: operation, Task: definition.Task, Outcome: runrecord.OutcomeSucceeded,
		StartedUnixNS: started.UnixNano(), MeasuredNS: measuredNS,
		Usage:     runrecord.ServingUsage{InputBytes: uint64(len(raw)), OutputBytes: uint64(len(output))},
		Resources: runrecord.ServingResources{PeakDeviceBytes: measured.PeakDeviceBytes},
	})
	if err != nil {
		t.Fatal(err)
	}
	checkDecision, err := recipe.NewDecision(
		testCase.ID, recipe.DecisionAccepted, recipe.EvidenceProduction, "",
		recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: check},
		[]artifact.ID{testCase.Input, interaction.Trace, receipt.ID, observation.ID},
	)
	if err != nil {
		t.Fatal(err)
	}
	decisionBatch, err := checkDecision.Batch("probe/check-decision/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, decisionBatch); err != nil {
		t.Fatal(err)
	}
	return productionProbeFixture{
		store: store, testCase: testCase, profile: profile, receipt: receipt,
		alias: alias, model: manifest.ID, definition: definition, executed: output,
		request: CapabilityProbeEvidence{
			Case: testCase.ID, Profile: profile.ID,
			Placement: placement, Environment: environment.ID, Trace: interaction.Trace,
			StageReceipt: receipt.ID, CheckDecision: checkDecision.ID, Observation: observation.ID,
		},
	}
}
