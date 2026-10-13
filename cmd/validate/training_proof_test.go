package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// fixtureCommit is the commit a fixture verification ran at.
const fixtureCommit = "0123456789abcdef0123456789abcdef01234567"

// TestEveryScorecardCellHasABoundCommandOrIsRetired holds every proof the
// scorecard waits on to a command that re-acquires it into the selected store,
// or to a declared reason the measurement run does not run it. Each training
// recipe declares an objective over the data its route grounds on, and the
// record reader names the route from that objective alone: the dit route
// trains a video stack from its directory and a raw checkpoint against that
// stack, and the dense workflow trains the recipe it is named. A recipe whose
// objective no committed data grounds but an alias registers is re-verified by
// a proof run of train -objective under that alias, bounded as the dense
// workflow's run, or, when it is a table objective, by the tabular route's
// proof run on the model directory above the head the catalog names, or, when
// it is a picture objective, by the codec route's proof run on the directory
// of the bridged pair, with a batch size and no token cache, since a proof
// keeps no tokens. A reward objective is proved by a reward session the
// operator starts, as the instruction pilot's sessions and a workbench's
// reward run are, and the measurement run has no bounded run to start for it:
// its cell is filed as the operator's with that reason, which the run prints,
// and is not run; its recipe and activation stay as they are. One over
// data nothing grounds or registers, or of a kind no route is bound for and no
// reason is declared for, still runs and binds nothing, so the run fails
// rather than pass over it. Each projection recipe passed a verification
// the reader names from its gate alone: an exact verification whose plan ran
// the committed suite re-runs it through recipe verify -reverify against the
// projector its activation binds, one the E4B serving producer published
// against the committed oracle runs that producer into the selected store,
// and an exact verification of another suite grounds nothing.
func TestEveryScorecardCellHasABoundCommandOrIsRetired(t *testing.T) {
	t.Parallel()
	ctx, root, models := t.Context(), testutil.RepoRoot(t), t.TempDir()
	const selected = "C:/stores/selected"
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publish := func(content artifact.Content, err error) artifact.ID {
		t.Helper()
		if err == nil {
			_, err = artifact.Publish(ctx, store, artifact.Batch{Key: content.Descriptor.ID.String(), Contents: []artifact.Content{content}})
		}
		if err != nil {
			t.Fatal(err)
		}
		return content.Descriptor.ID
	}
	trained := func(name, weights, grounding, alias string, kind trainingprogram.ObjectiveKind, metric trainingprogram.EvaluationMetric, modality recipecontract.Modality) ModelValidation {
		data := []byte("data no route grounds")
		if grounding != "" {
			if data, err = os.ReadFile(filepath.Join(root, filepath.FromSlash(grounding))); err != nil {
				t.Fatal(err)
			}
		}
		profile := testutil.ArtifactID(t, artifact.KindProfile, name)
		objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
			Name: name, Kind: kind,
			Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{modality}, Outputs: []recipecontract.Modality{modality}},
			Dataset:   testutil.ArtifactBytesID(t, artifact.KindDataset, data), Split: testutil.ArtifactID(t, artifact.KindDatasetShard, name),
			Processors: []artifact.ID{profile}, Loss: profile, Evaluation: profile, Metric: metric,
			Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, name)}, Authority: trainingprogram.ObjectiveApproved,
		})
		if err != nil {
			t.Fatal(err)
		}
		model, objectiveID := testutil.ArtifactID(t, artifact.KindModel, name), publish(objective.Content())
		if alias != "" {
			if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/objective/" + name, Aliases: []artifact.AliasBinding{{Name: alias, Target: objectiveID}}}); err != nil {
				t.Fatal(err)
			}
		}
		definition, err := recipe.NewDefinitionWithDependencies(recipe.TaskTraining,
			[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: model}, {Role: recipe.DependencyObjective, Artifact: objectiveID}},
			[]recipe.Node{{ID: "optimize", Module: "training.optimize", Placement: recipe.PlacementHost}}, nil, nil,
			[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}})
		if err != nil {
			t.Fatal(err)
		}
		return ModelValidation{
			Model: model, ModelName: name, Location: filepath.Join(models, name, weights),
			Kind: recipe.TaskTraining, Validation: trainingProof, Recipe: publish(definition.ArtifactContent()),
		}
	}
	training := func(name, weights, grounding, alias string) ModelValidation {
		return trained(name, weights, grounding, alias, trainingprogram.ObjectiveTokenPrediction, trainingprogram.MetricTokenAccuracy, recipecontract.ModalityText)
	}
	stack, edit := training("Wan", ditWeights, ditGrounding, ""), training("LiveEdit", "edit.pt", ditGrounding, "")
	mot := training("RxBrain", "model-00001-of-00003.safetensors", "docs/verification/rxbrain-mot-training-stimulus.txt", "")
	hybrid := training("Qwen35", "qwen35.gguf", "fixtures/qwen35_4b_serving_golden.json", "")
	dense, ungrounded := training("Qwen25", "model.safetensors", denseGrounding, ""), training("Unrouted", "model.safetensors", "", "")
	const registered = "objective.registered.text-text"
	objective := training("Qwen05", "model.safetensors", "", registered)
	// A table objective an alias registers trains through the tabular route. A
	// forecast objective is registered too, and no route is bound for its kind.
	const tabled, forecasted = "objective.registered.table-table", "objective.registered.time-series-time-series"
	table := trained("TabFM", filepath.Join("classification", "model.safetensors"), "", tabled,
		trainingprogram.ObjectiveTablePrediction, trainingprogram.MetricTableAccuracy, recipecontract.ModalityTable)
	forecast := trained("TimesFM", "model.safetensors", "", forecasted,
		trainingprogram.ObjectiveForecast, trainingprogram.MetricForecastMAE, recipecontract.ModalityTimeSeries)
	// A picture objective an alias registers trains through the codec route, on
	// the directory of the pair its bridge sits in. The reward cell is shaped
	// as the instruction pilot's is, a reward objective no alias registers, on
	// a GGUF: the measurement run binds no proof run for its kind.
	const pictured = "objective.registered.image-image"
	bridge := trained("Bridge", "encoder_bridge.safetensors", "", pictured,
		trainingprogram.ObjectiveImageLatent, trainingprogram.MetricImagePSNRUnit, recipecontract.ModalityImage)
	reward := trained("Pilot", "pilot.gguf", "", "",
		trainingprogram.ObjectiveGRPO, trainingprogram.MetricTokenAccuracy, recipecontract.ModalityText)

	projectorPaths := map[string]string{}
	projection := func(name, evidence string) ModelValidation {
		key := "fixture/projection/" + name
		modelWeights, projectorWeights := []byte(name+" model weights"), []byte(name+" projector weights")
		modelComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, modelWeights)
		projectorComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, projectorWeights)
		directory := filepath.Join(models, name)
		modelPath, projectorPath := filepath.Join(directory, "model.gguf"), filepath.Join(directory, "mmproj.gguf")
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		for path, payload := range map[string][]byte{modelPath: modelWeights, projectorPath: projectorWeights} {
			if err := os.WriteFile(path, payload, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: modelComponent}})
		if err != nil {
			t.Fatal(err)
		}
		projector, err := artifact.NewManifest(artifact.KindProjector, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: projectorComponent}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: key + "/facts",
			Artifacts: []artifact.Descriptor{
				{ID: modelComponent, Size: uint64(len(modelWeights))}, {ID: projectorComponent, Size: uint64(len(projectorWeights))},
			},
			Manifests: []artifact.Manifest{model, projector},
			Locations: []artifact.LocationEvent{
				{Location: artifact.Location{Artifact: modelComponent, Kind: artifact.LocationFile, Value: modelPath}, Action: artifact.LocationAdd},
				{Location: artifact.Location{Artifact: projectorComponent, Kind: artifact.LocationFile, Value: projectorPath}, Action: artifact.LocationAdd},
			},
		}); err != nil {
			t.Fatal(err)
		}
		definition, err := modelrecipe.ProjectionDefinition(model.ID, projector.ID, artifact.ID{}, recipe.DataImage)
		if err != nil {
			t.Fatal(err)
		}
		if err := modelrecipetest.PublishActivation(ctx, store, key, definition); err != nil {
			t.Fatal(err)
		}
		environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: key, OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test"})
		if err != nil {
			t.Fatal(err)
		}
		environmentBatch, err := environment.Batch(key + "/environment")
		if err == nil {
			_, err = artifact.CommitBatch(ctx, store, environmentBatch)
		}
		if err != nil {
			t.Fatal(err)
		}
		gate, err := runrecord.NewGateRecord(definition.ID, environment.ID, fixtureCommit, runrecord.OutcomeSucceeded, "", 1, []runrecord.GateStep{{
			Name: "candidate-execution", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: evidence,
		}})
		if err != nil {
			t.Fatal(err)
		}
		gateBatch, err := gate.Batch(key + "/gate")
		if err == nil {
			_, err = artifact.CommitBatch(ctx, store, gateBatch)
		}
		if err != nil {
			t.Fatal(err)
		}
		projectorPaths[name] = projectorPath
		return ModelValidation{
			Model: model.ID, ModelName: name, Location: modelPath,
			Kind: recipe.TaskProjection, Validation: projectionProof, Recipe: definition.ID,
		}
	}
	exactRan := func(suite evaluation.ExactSuite) string {
		compiled, err := evaluation.CompileExact(suite)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := evaluation.BindExact(compiled, evaluation.ExactAuthorities{
			ModelDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "exact"), RuntimeRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "exact"),
			CodeCommit: fixtureCommit, Environment: testutil.ArtifactID(t, artifact.KindEvidence, "exact"),
			Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
		})
		if err != nil {
			t.Fatal(err)
		}
		return exactPlan + publish(plan.Content()).String() + ";report=" + testutil.ArtifactID(t, artifact.KindEvaluation, "exact").String() + ";cases=" + strconv.Itoa(len(suite.Cases))
	}
	var suite evaluation.ExactSuite
	if err := jsonfile.DecodeStrict(filepath.Join(root, filepath.FromSlash(projectionSuites[0])), &suite); err != nil {
		t.Fatal(err)
	}
	exact := projection("Qwen27", exactRan(suite))
	oracle, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(protocolOracle)))
	if err != nil {
		t.Fatal(err)
	}
	protocol := projection("E4B", fmt.Sprintf("%s%x;model_definition=%s;cases=18", protocolNames, sha256.Sum256(oracle), testutil.ArtifactID(t, artifact.KindModelDefinition, "e4b")))
	suite.Cases = slices.Clone(suite.Cases)
	suite.Cases[0].Text += " but another answer"
	otherSuite := projection("OtherSuite", exactRan(suite))

	cells := []ModelValidation{stack, edit, mot, hybrid, dense, ungrounded, objective, table, forecast, bridge, reward, exact, protocol, otherSuite}
	for _, task := range []recipe.Task{
		recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskTranscription,
		recipe.TaskVQA, recipe.TaskSeq2Seq, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskImageEmbedding, recipe.TaskReconstruction,
	} {
		for _, validation := range kindValidation(task) {
			cells = append(cells, ModelValidation{
				Model: testutil.ArtifactID(t, artifact.KindModel, string(task)), ModelName: string(task),
				Location: "C:/models/" + string(task) + "/model.safetensors", Directory: "C:/models/" + string(task),
				Kind: task, Validation: validation, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, string(task)),
			})
		}
	}
	if err := groundTraining(ctx, root, store, cells); err != nil {
		t.Fatal(err)
	}
	if err := groundProjection(ctx, root, store, cells); err != nil {
		t.Fatal(err)
	}
	plan := SelectValidation(Inputs{Registered: cells})
	// Every proof runs but the reward recipe's, which is left to the operator
	// with its reason, and the run says so.
	if len(plan.Run) != len(cells)-1 || len(plan.OperatorOwned) != 1 || plan.OperatorOwned[0].ModelName != reward.ModelName ||
		!strings.Contains(plan.OperatorOwned[0].Reason, "reward recipe") {
		t.Errorf("the run takes %d of %d proofs and leaves the operator %+v, want every proof run but %s's, and that one left to the operator as a reward recipe",
			len(plan.Run), len(cells), plan.OperatorOwned, reward.ModelName)
	}
	var printed bytes.Buffer
	if err := executeRun(ctx, &printed, root, selected, nil, Plan{OperatorOwned: plan.OperatorOwned}, scopeCells); err != nil {
		t.Fatal(err)
	}
	for _, owned := range plan.OperatorOwned {
		if !strings.Contains(printed.String(), owned.ModelName) || !strings.Contains(printed.String(), owned.Reason) {
			t.Errorf("the run printed %q, want %s named as not run, with its reason %q", printed.String(), owned.ModelName, owned.Reason)
		}
	}

	// A kind with no route and no declared reason still runs and still refuses.
	refusals := map[string]string{
		ungrounded.ModelName: "no cmd/train route grounds", forecast.ModelName: "no cmd/train route grounds",
		otherSuite.ModelName: "no committed exact suite or oracle grounds",
	}
	refusing := 0
	commands, environments := map[string][]string{}, map[string][]string{}
	for _, sel := range plan.Run {
		_, environments[sel.ModelName], _ = trainingRun(sel.ModelValidation, selected)
		if sel.Validation == projectionProof {
			_, environments[sel.ModelName], _ = projectionRun(sel.ModelValidation, selected)
		}
		name, args, err := reacquireCommand(sel.ModelValidation, selected)
		if sel.ModelName == protocol.ModelName {
			if err != nil || name != "go" || !slices.Equal(args, protocolProducer) ||
				!slices.Contains(environments[sel.ModelName], protocolStore+selected) || !slices.Contains(environments[sel.ModelName], protocolPublish) ||
				!slices.Contains(environments[sel.ModelName], protocolDevice) {
				t.Fatalf("%s %s bound (%q, %v, %v) under %v, want the serving producer publishing into the selected store", sel.ModelName, sel.Validation, name, args, err, environments[sel.ModelName])
			}
			continue
		}
		if refusal, refused := refusals[sel.ModelName]; refused {
			if err == nil || !strings.Contains(err.Error(), refusal) {
				t.Fatalf("%s %s bound (%v, %v), want the refusal that it %s", sel.ModelName, sel.Validation, args, err, refusal)
			}
			refusing++
			continue
		}
		if err != nil || name != "go" || !slices.Contains(args, selected) {
			t.Errorf("%s %s bound (%q, %v, %v), want a command into the selected store", sel.ModelName, sel.Validation, name, args, err)
			continue
		}
		commands[sel.ModelName] = args
	}
	if refusing != len(refusals) {
		t.Errorf("%d of the %d proofs nothing grounds were run and refused, want each of them: a cell without a command or a reason must fail the run", refusing, len(refusals))
	}
	flag := func(model, name string) string {
		args := commands[model]
		if at := slices.Index(args, name); at >= 0 && at+1 < len(args) {
			return args[at+1]
		}
		return ""
	}
	for model, want := range map[string]map[string]string{
		stack.ModelName:  {"-route": "dit", "-model-dir": filepath.Dir(stack.Location), "-checkpoint": ""},
		edit.ModelName:   {"-route": "dit", "-model-dir": filepath.Dir(stack.Location), "-checkpoint": edit.Location},
		mot.ModelName:    {"-route": "mot", "-model": filepath.Dir(mot.Location)},
		hybrid.ModelName: {"-route": "hybrid-stack", "-model": hybrid.Location},
		dense.ModelName:  {"-route": "", "-recipe": dense.Recipe.String(), "-model": filepath.Dir(dense.Location), "-dataset": denseGrounding},
		objective.ModelName: {
			"-route": "", "-recipe": objective.Recipe.String(), "-dataset": "", "-objective": registered, "-model": filepath.Dir(objective.Location),
			"-steps": denseSteps, "-seq": denseSequence,
		},
		table.ModelName: {
			"-route": tabularRoute, "-recipe": table.Recipe.String(), "-objective": tabled, "-model": filepath.Dir(filepath.Dir(table.Location)),
			"-steps": routeSteps, "-seq": "", "-dataset": "",
		},
		bridge.ModelName: {
			"-route": "codec", "-recipe": bridge.Recipe.String(), "-objective": pictured, "-pair": filepath.Dir(bridge.Location),
			"-steps": routeSteps, "-model": "", "-cache": "", "-seq": "", "-dataset": "",
		},
	} {
		for name, value := range want {
			if got := flag(model, name); got != value {
				t.Errorf("%s runs %s %q, want %q: %v", model, name, got, value, commands[model])
			}
		}
		if out := flag(model, "-out"); out != "checkpoint" {
			t.Errorf("%s preview publishes its checkpoint at %q, want a caller-chosen output", model, out)
		}
	}
	// The objective's proof run re-verifies its recipe without publishing the
	// held-out verdict its quality reads, frozen as the dense workflow's run.
	if proof := commands[objective.ModelName]; !slices.Contains(proof, "-proof") || !slices.Contains(proof, "-freeze-lexical") {
		t.Errorf("%s runs %v, want a lexically frozen -objective proof run", objective.ModelName, proof)
	}
	// A table objective's is the tabular route's proof run, which has no text to freeze.
	if proof := commands[table.ModelName]; !slices.Contains(proof, "-proof") || slices.Contains(proof, "-freeze-lexical") {
		t.Errorf("%s runs %v, want the tabular route's proof run", table.ModelName, proof)
	}
	// A picture objective's is the codec route's proof run, a batch of pictures a step.
	if batch, err := strconv.Atoi(flag(bridge.ModelName, "-batch")); err != nil || batch <= 0 || !slices.Contains(commands[bridge.ModelName], "-proof") {
		t.Errorf("%s runs %v, want the codec route's proof run on a positive batch", bridge.ModelName, commands[bridge.ModelName])
	}
	verify := commands[exact.ModelName]
	for name, value := range map[string]string{
		"run": "./cmd/recipe", "-task": string(recipe.TaskProjection), "-projector": projectorPaths[exact.ModelName], "-input-file": projectionSuites[0],
	} {
		if got := flag(exact.ModelName, name); got != value {
			t.Errorf("%s runs %s %q, want %q: %v", exact.ModelName, name, got, value, verify)
		}
	}
	if !slices.Contains(verify, "-reverify") || verify[len(verify)-1] != exact.Location {
		t.Errorf("%s runs %v, want its exact suite to re-verify the active recipe on its model", exact.ModelName, verify)
	}
	if !slices.Contains(environments[hybrid.ModelName], "GOMEMLIMIT=68GiB") {
		t.Errorf("the hybrid stack runs without the memory limit its route was recorded under: %v", environments[hybrid.ModelName])
	}
	// A selected cell keeps its recipe even if another invocation replaces
	// the active alias before its proof is read.
	definition, err := recipe.RequireDefinition(ctx, store, objective.Recipe)
	if err != nil {
		t.Fatal(err)
	}
	testutil.PublishArtifact(t, store, objective.Model)
	for _, placement := range []recipe.Placement{recipe.PlacementHost, recipe.PlacementDevice} {
		nodes := slices.Clone(definition.Nodes)
		nodes[0].Placement = placement
		active, err := recipe.NewDefinitionWithDependencies(definition.Task, definition.Dependencies,
			nodes, definition.Edges, definition.Inputs, definition.Outputs)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/proof/"+string(placement)+"/candidate", active); err != nil {
			t.Fatal(err)
		}
		verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/proof/"+string(placement), active.ID)
		if err != nil {
			t.Fatal(err)
		}
		if err := modelrecipe.ActivateCapability(ctx, store, active, verification, recipe.EvidenceVerified, "proof selection fixture"); err != nil {
			t.Fatal(err)
		}
		current := objective
		current.Recipe = active.ID
		selections := []ModelValidation{objective}
		if current.Recipe != objective.Recipe {
			selections = append(selections, current)
		}
		for _, cell := range selections {
			t.Run(string(placement)+"/"+cell.Recipe.String(), func(t *testing.T) {
				cost, err := proofCost(ctx, store, cell)
				if err != nil {
					t.Fatal(err)
				}
				if cell.Recipe != active.ID {
					if cost != 0 {
						t.Fatalf("another recipe supplied selected proof: cost %d", cost)
					}
					return
				}
				run, err := runrecord.RequireRun(ctx, store, verification.Run)
				if err != nil || cost != int64(run.MeasuredNS) {
					t.Fatalf("selected proof cost = (%d, %v), want run %s's %d", cost, err, run.ID, run.MeasuredNS)
				}
			})
		}
	}
}
