package trainingprogram_test

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestPreferenceLossVariantsShareTheDPOPath trains a tiny policy in each
// loss mode of the DPO objective through the one DPO workflow and recipe,
// then judges it with the one held-out preference judge: every pair,
// training or held out, prefers c to d after its prompt, and each mode must
// lower DPO's held-out loss against the frozen reference on the pairs
// training never read. Every mode trains the DPO objective, seals its mode
// into a program of its own, and is judged on the same held-out view; DPO's
// own mode keeps the policy encoding, and so the program identities, it had
// before modes existed.
func TestPreferenceLossVariantsShareTheDPOPath(t *testing.T) {
	fixture := newPreferenceVariantFixture(t)
	views, programs := map[artifact.ID]bool{}, map[artifact.ID]bool{}
	for _, variant := range []struct {
		mode          trainingprogram.PreferenceMode
		scale, margin float64
	}{
		{scale: 0.1},
		{mode: trainingprogram.PreferenceSimPO, scale: 2, margin: 1},
		{mode: trainingprogram.PreferenceORPO, scale: 0.1},
	} {
		name := cmp.Or(string(variant.mode), "dpo")
		t.Run(name, func(t *testing.T) {
			const steps = 12
			request := trainingworkflow.Request{
				Repository: fixture.store, Recipe: fixture.recipe, Observations: fixture.store, Host: true,
				ModelDirectory: fixture.policy, ReferenceDirectory: fixture.reference, DatasetPath: fixture.dataset,
				OutputDirectory: filepath.Join(fixture.root, name), Steps: steps,
				ObjectiveScale: variant.scale, PreferenceMode: variant.mode, PreferenceMargin: variant.margin,
			}
			result, err := trainingworkflow.Execute(t.Context(), request)
			if err != nil || result.Objective != trainingprogram.ObjectiveDPO || len(result.DPO) != steps {
				t.Fatalf("objective=%s observations=%d err=%v, want %d DPO steps", result.Objective, len(result.DPO), err, steps)
			}
			verdict, judged, err := trainingworkflow.JudgePreferenceHeldout(t.Context(), fixture.store, request, result)
			if err != nil || !judged || !verdict.Passed {
				t.Fatalf("held-out verdict=%+v judged=%t err=%v, want a passing preference verdict", verdict, judged, err)
			}
			views[verdict.View], programs[result.Checkpoint.Program] = true, true
			t.Logf("first step loss %.6f, last step loss %.6f, held-out verdict %s passed", result.DPO[0].Loss, result.DPO[steps-1].Loss, verdict.Verdict)
		})
	}
	if len(views) != 1 || len(programs) != 3 {
		t.Fatalf("held-out views=%d programs=%d, want one view judging three programs", len(views), len(programs))
	}
	reference := testutil.ArtifactID(t, artifact.KindModel, "reference")
	own, err := json.Marshal(trainingprogram.PreferencePolicy{Reference: reference, Scale: 0.1})
	if err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(struct {
		Reference artifact.ID
		Scale     float64
	}{reference, 0.1})
	if err != nil || !bytes.Equal(own, before) {
		t.Fatalf("DPO's own policy encodes %s, before modes %s (err %v)", own, before, err)
	}
}

// preferenceVariantFixture is a store holding an active DPO recipe whose
// objective trains on one membership of a group split and holds out the
// other, with the policy, reference and dataset it binds on disk.
type preferenceVariantFixture struct {
	store                            *overgodb.Store
	recipe                           artifact.ID
	root, policy, reference, dataset string
	training, heldout                dataset.Membership
}

// preferencePrompts are the fixtures' prompts over a and b; every
// completion prefers c to d after one.
var preferencePrompts = []string{"a", "b", "aa", "ab", "ba", "bb", "aaa", "aab", "aba", "abb", "baa", "bab"}

func newPreferenceVariantFixture(t *testing.T) preferenceVariantFixture {
	t.Helper()
	documents := make([]map[string]any, len(preferencePrompts))
	for index, prompt := range preferencePrompts {
		documents[index] = map[string]any{
			"id": fmt.Sprintf("pair-%02d", index), "group": fmt.Sprintf("group-%02d", index), "prompt": prompt, "chosen": "c", "rejected": "d",
		}
	}
	return newPreferenceFixture(t, documents, "overgo/training/preference-token-pair/v1", trainingprogram.MetricPreferenceLoss)
}

// newPreferenceFixture commits documents, each with an id and a group, as
// a dataset split by group, and an active DPO recipe whose objective reads
// them through the processor profile names, trains on the training
// membership and declares metric on the held-out one.
func newPreferenceFixture(t *testing.T, documents []map[string]any, processorProfile string, metric trainingprogram.EvaluationMetric) preferenceVariantFixture {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	fixture := preferenceVariantFixture{
		store: store, root: root, policy: filepath.Join(root, "policy"), reference: filepath.Join(root, "reference"),
		dataset: filepath.Join(root, "preference.jsonl"),
	}
	policy := writePreferenceModel(t, fixture.policy, 9)
	reference := writePreferenceModel(t, fixture.reference, 10)
	records := make([]dataset.Record, len(documents))
	var lines bytes.Buffer
	for index, document := range documents {
		records[index] = dataset.Record{ID: document["id"].(string), Group: document["group"].(string)}
		line, err := json.Marshal(document)
		if err != nil {
			t.Fatal(err)
		}
		lines.Write(append(line, '\n'))
	}
	if err := os.WriteFile(fixture.dataset, lines.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	source, err := artifact.IdentifyBytes(artifact.KindDataset, lines.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	const splitSeed = 17
	split, err := dataset.BuildGroupSplit(source, records, splitSeed, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := split.PublicationBatch("fixture/preference-variants/split", nil)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: source, Size: uint64(lines.Len())})
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	for _, membership := range split.Memberships {
		if membership.Partition == "train" {
			fixture.training = membership
		} else {
			fixture.heldout = membership
		}
	}
	profile := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindProfile, name) }
	processor, evaluation := profile(processorProfile), profile("dpo evaluation")
	optimizer := trainingprogram.BuiltinOptimizerPolicy()
	policies := trainingprogram.PolicySpec{
		Precision: profile("dpo precision"), Placement: profile("dpo placement"), Memory: profile("dpo memory"),
		Checkpoint: profile("dpo checkpoint"), Evaluation: evaluation, Promotion: profile("dpo promotion"), Optimizer: optimizer.ID,
	}
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "dpo objective evidence")
	text := []recipecontract.Modality{recipecontract.ModalityText}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "dpo", Kind: trainingprogram.ObjectiveDPO, Signature: recipecontract.ModalitySignature{Inputs: text, Outputs: text},
		Dataset: source, Split: fixture.training.ID, Processors: []artifact.ID{processor}, Loss: profile("dpo loss"), Evaluation: evaluation,
		Metric: metric, Evidence: []artifact.ID{evidence}, Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	policies.Objective = objective.ID
	objectiveContent, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	optimizerContent, err := optimizer.Content()
	if err != nil {
		t.Fatal(err)
	}
	var descriptors []artifact.Descriptor
	for _, id := range []artifact.ID{
		policy, reference, processor, objective.Loss, evidence, policies.Precision, policies.Placement,
		policies.Memory, policies.Checkpoint, policies.Evaluation, policies.Promotion,
	} {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/preference-variants/authority", Artifacts: descriptors,
		Contents: []artifact.Content{objectiveContent, optimizerContent},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipetest.DPODefinition(append([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: policy},
		{Role: recipe.DependencyModel, Slot: 1, Artifact: reference},
	}, modelrecipetest.PolicyDependencies(policies)...))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/preference-variants/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/preference-variants/verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "preference variants fixture"); err != nil {
		t.Fatal(err)
	}
	fixture.recipe = definition.ID
	return fixture
}

// writePreferenceModel writes a tiny dense causal model seeded by seed, with
// a four-letter tokenizer, to directory and returns its identity.
func writePreferenceModel(t *testing.T, directory string, seed int64) artifact.ID {
	t.Helper()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: seed,
	})
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, trainingprogram.CheckpointWeights)
	if err := safetensors.Save(path, weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"config.json":    `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"max_position_embeddings":16,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`,
		"tokenizer.json": `{"model":{"type":"BPE","vocab":{"a":1,"b":2,"c":3,"d":4},"merges":[]}}`,
	} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindModel, data)
	if err != nil {
		t.Fatal(err)
	}
	return id
}
