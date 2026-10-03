package trainingworkflow

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// TestHeldoutApprovalReconstructsTheTrainedResult trains a GGUF base under a
// token-prediction objective through the held-out text run and approves the
// objective on its verdict, which binds the checkpoint the run published,
// holding the trained GGUF registered as a candidate, and the held-out
// inputs both models read. From the verdict alone the trained model reloads
// and, on the held-out inputs read again, reproduces the frozen view, plan
// and both reports. A second split of the same dataset coexists, and each
// objective resolves its own split's held-out membership; a checkpoint that
// trained on the other split's membership is refused as the trained result.
func TestHeldoutApprovalReconstructsTheTrainedResult(t *testing.T) {
	ctx, root := t.Context(), t.TempDir()
	base := writeTrainingGGUF(t)
	corpus := filepath.Join(root, "corpus")
	if err := os.MkdirAll(corpus, 0o755); err != nil {
		t.Fatal(err)
	}
	const documents = 12
	for index := range documents {
		// Distinct lengths keep every file its own record; the pattern fixes
		// each next token.
		document := strings.Repeat("abcd", 5)[:6+index]
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("text-%02d.txt", index)), []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, "texts", corpus)
	if err != nil {
		t.Fatal(err)
	}
	var records []dataset.Record
	for index := range uint64(documents) {
		id := dataset.RecordID(registered.Dataset, dataset.InventoryAssetName, index)
		records = append(records, dataset.Record{ID: id, Group: id})
	}
	// Two splits of one dataset under one seed: the same-seed sibling scan
	// found three candidate partners here.
	derived := derivedIdentity(t, "overgo/heldout-binding-test/")
	objectives := map[string]trainingprogram.ObjectiveDocument{}
	partners := map[string]artifact.ID{}
	for name, weights := range map[string][]dataset.SplitPartition{
		"halves": {{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}},
		"thirds": {{Name: "heldout", Weight: 1}, {Name: "train", Weight: 2}},
	} {
		split, err := dataset.BuildGroupSplit(registered.Dataset, records, 17, weights)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := split.PublicationBatch("test-split-"+name, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		var training artifact.ID
		for _, membership := range split.Memberships {
			if membership.Partition == "train" {
				training = membership.ID
			} else {
				partners[name] = membership.ID
			}
		}
		objectives[name], err = trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
			Name: "texts-" + name, Kind: trainingprogram.ObjectiveTokenPrediction, Signature: textSignature,
			Dataset: registered.Dataset, Split: training, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
			Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
			Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
			Authority: trainingprogram.ObjectiveDeclared,
		})
		if err != nil {
			t.Fatal(err)
		}
		content, err := objectives[name].Content()
		if err != nil {
			t.Fatal(err)
		}
		registration := artifact.Batch{
			Key: "test-objective-" + name, Contents: []artifact.Content{content},
			Aliases: []artifact.AliasBinding{{Name: "objective.registered.texts-" + name, Target: objectives[name].ID}},
		}
		referenced := []artifact.ID{derived(artifact.KindProfile, "processor"), derived(artifact.KindProfile, "loss"), derived(artifact.KindProfile, "evaluation"), derived(artifact.KindEvidence, "note")}
		for _, policy := range []string{"precision", "placement", "memory", "checkpoint", "promotion"} {
			id, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+policy))
			if err != nil {
				t.Fatal(err)
			}
			referenced = append(referenced, id)
		}
		for _, id := range referenced {
			registration.Artifacts = append(registration.Artifacts, artifact.Descriptor{ID: id})
		}
		if _, err := artifact.CommitBatch(ctx, store, registration); err != nil {
			t.Fatal(err)
		}
	}
	for name, objective := range objectives {
		if _, heldout, err := objectiveMemberships(ctx, store, objective); err != nil || heldout.ID != partners[name] {
			t.Fatalf("objective %s resolves held-out %s (%v), want its own split's %s", name, heldout.ID, err, partners[name])
		}
	}

	const alias = "objective.registered.texts-halves"
	result, err := RunTextHeldout(ctx, store, TextHeldoutRequest{Alias: alias, ModelDirectory: base, OutputDirectory: filepath.Join(root, "trained"), Steps: 48, Host: true})
	if err != nil || !result.Passed {
		t.Fatalf("text held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	if approved, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{result.Verdict}); err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}

	// Everything below starts from the committed verdict.
	var verdict struct {
		View, Plan, Trained, Inputs, Baseline, Candidate artifact.ID
	}
	content, found, err := artifact.ReadContent(ctx, store, result.Verdict)
	if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil || verdict.Trained != result.Trained {
		t.Fatalf("verdict %s = (%+v, found %t, %v), want it to bind the trained checkpoint %s", result.Verdict, verdict, found, err, result.Trained)
	}
	directory, err := artifact.AvailablePath(ctx, store, verdict.Trained, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(directory)
	if err != nil || checkpoint.ID() != verdict.Trained {
		t.Fatalf("the bound checkpoint reloads as (%s, %v), want %s", checkpoint.ID(), err, verdict.Trained)
	}
	file, err := gguf.Open(filepath.Join(directory, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	weights, manifestErr := artifact.NewManifest(artifact.KindTensorSet, inventory.Manifest.Components)
	if closeErr := file.Close(); err != nil || closeErr != nil || manifestErr != nil || weights.ID != checkpoint.Weights {
		t.Fatalf("published model = (%+v, %v, %v), want the checkpoint's weights %s", inventory.Components, err, closeErr, checkpoint.Weights)
	}
	published := inventory.Manifest.ID
	if parents, err := store.Parents(ctx, published); err != nil || !slices.Contains(parents, artifact.Lineage{Child: published, Parent: checkpoint.ID(), Relation: artifact.RelationProducedBy}) {
		t.Fatalf("published model %s lineage = (%v, %v), want it produced by the bound checkpoint", published, parents, err)
	}

	objective := objectives["halves"]
	tokenizer, err := loadTrainableTokenizer(base)
	if err != nil {
		t.Fatal(err)
	}
	baseModel, err := loadTrainableModel(base)
	if err != nil {
		t.Fatal(err)
	}
	data, _, err := textObjectiveData(ctx, store, objective, tokenizer, baseModel.Dims.ContextLength)
	if err != nil || data.Heldout != verdict.Inputs {
		t.Fatalf("held-out inputs read again = (%s, %v), want the bound %s", data.Heldout, err, verdict.Inputs)
	}
	training, heldout, err := objectiveMemberships(ctx, store, objective)
	if err != nil {
		t.Fatal(err)
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
	if err != nil || view.ID != verdict.View {
		t.Fatalf("view recompiled = (%s, %v), want %s", view.ID, err, verdict.View)
	}
	var cases []evaluation.HeldoutCase
	for _, record := range view.Records {
		cases = append(cases, evaluation.HeldoutCase{Record: record.ID, Values: data.Examples[record.ID].Target})
	}
	plan, err := evaluation.CompileHeldoutPlan(objective, view, cases)
	if err != nil || plan.ID() != verdict.Plan {
		t.Fatalf("plan recompiled = (%s, %v), want %s", plan.ID(), err, verdict.Plan)
	}
	trained, err := loadTrainableModel(directory)
	if err != nil {
		t.Fatal(err)
	}
	score := func(model *densecausal.Model) evaluation.HeldoutReport {
		t.Helper()
		var observations []evaluation.HeldoutObservation
		for _, target := range cases {
			example := data.Examples[target.Record]
			example.Record = target.Record
			prediction, err := nextTokens(model, example)
			if err != nil {
				t.Fatal(err)
			}
			observations = append(observations, evaluation.HeldoutObservation{Record: target.Record, Values: prediction.Values})
		}
		report, err := plan.Score(observations)
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	baseline, candidate := score(baseModel), score(trained)
	if baseline.ID() != verdict.Baseline || candidate.ID() != verdict.Candidate {
		t.Fatalf("reports reproduced = (%s, %s), want baseline %s and candidate %s", baseline.ID(), candidate.ID(), verdict.Baseline, verdict.Candidate)
	}

	// The other split's training membership did not train this objective's
	// result.
	otherwise := recordedCheckpoint(t, store, checkpoint.Model, objective.Dataset, objectives["thirds"].Split)
	if _, err := evaluation.JudgeHeldout(objective, view, plan, baseline, candidate, evaluation.HeldoutTraining{
		Observation: result.Observation, Model: checkpoint.Model, Trained: otherwise, Inputs: verdict.Inputs,
	}); err == nil {
		t.Fatal("a checkpoint that trained on another split's membership was judged as the trained result")
	}
}
