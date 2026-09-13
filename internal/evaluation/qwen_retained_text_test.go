package evaluation

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
)

type retainedNativeSequence struct {
	Text             string
	IDs              []int32
	LogProbabilities []float64 `json:"continuation_log_probabilities"`
}

func checkRetainedNativeSequence(native retainedNativeSequence, actual inference.PerplexityResult) error {
	if len(native.IDs) < 2 || len(native.LogProbabilities) != len(native.IDs)-1 || actual.TokenCount != len(native.IDs) || actual.EvaluatedTokens != len(native.LogProbabilities) || len(actual.Scores) != len(native.LogProbabilities) {
		return errors.New("native sequence denominator differs")
	}
	for i, score := range actual.Scores {
		if score.Position != i+1 || int32(score.TokenID) != native.IDs[i+1] || math.IsNaN(score.NegativeLogLik) || math.IsInf(score.NegativeLogLik, 0) {
			return errors.New("native sequence token or score differs")
		}
		for j := range i {
			nativeOrder := native.LogProbabilities[j] - native.LogProbabilities[i]
			actualOrder := score.NegativeLogLik - actual.Scores[j].NegativeLogLik
			if math.IsNaN(nativeOrder) || math.IsInf(nativeOrder, 0) || nativeOrder == 0 && actualOrder != 0 || nativeOrder != 0 && nativeOrder*actualOrder <= 0 {
				return errors.New("native within-sequence ordering differs")
			}
		}
	}
	return nil
}

// Accept only the pinned native contracts; no quality or throughput floor.
func TestQwenSmallRetainedTextAcceptance(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(value string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	var retained struct {
		Inputs       map[string]artifact.ID
		Plan, Report artifact.ID
	}
	readRetainedEvidence(t, store, parse("evidence:sha256:d19ef1fb3ca0303f93d0e5937c19f68dbe0cff1aacda4f385ec5e43b58080849"), &retained)
	var acquisition struct {
		Producer      string
		Identity      modelrecipe.ProgramIdentity
		Environment   artifact.ID
		Likelihoods   []inference.PerplexityResult
		Continuations []sequencescore.Score
		Reversed      []sequencescore.Score `json:"reversed_continuations"`
		ModelLoads    int                   `json:"model_loads"`
		Failure       string
	}
	readRetainedEvidence(t, store, retained.Inputs["qwen-current-bundle.json"], &acquisition)
	model := parse("model:sha256:2164ee6535ced88ef8f705243eecaad686de45521f7a33e84bff5ba7643587e8")
	active, found, err := modelrecipe.ActiveRecord(t.Context(), store, model, recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("active model absent: %v", err)
	}
	definition, found := active.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !found || acquisition.Identity.Model != model || acquisition.Identity.Definition != definition || acquisition.Identity.Recipe != active.Definition.ID || acquisition.Producer != "85c67e71c17dc0d72ae65571c6a2a9cf33372c83" || acquisition.ModelLoads != 1 || acquisition.Failure != "evaluation: resource evidence is not process isolated" {
		t.Fatal("original acquisition authority or disposition differs")
	}
	data, err := os.ReadFile(filepath.Join(root, "fixtures/qwen25_serving_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != "b6d7d8c8f35aa9dce6ada81aca2ddafbdeefba5ff25305312c57d1fac9f02a1e" {
		t.Fatal("native exact fixture differs")
	}
	var fixture ExactSuite
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	exact, err := CompileExact(fixture)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindExact(exact, ExactAuthorities{ModelDefinition: definition, RuntimeRecipe: active.Definition.ID, CodeCommit: acquisition.Producer, Environment: acquisition.Environment, Execution: ExecutionPolicy{Lifecycle: LifecycleResident}})
	if err != nil || plan.Identity() != retained.Plan {
		t.Fatalf("current exact protocol differs: %v", err)
	}
	shards, err := compileExactShards(exact, plan)
	if err != nil {
		t.Fatal(err)
	}
	reports, err := loadShardReports(t.Context(), store, plan, shards)
	if err != nil || len(reports) != len(fixture.Cases) || len(reports) != 4 {
		t.Fatalf("retained case denominator differs: %v", err)
	}
	ordered := make([]shardReport, len(shards))
	for i, testCase := range fixture.Cases {
		report, found := reports[i]
		if !found || len(report.Observations) != 1 || !report.Observations[0].Passed {
			t.Fatal("exact case missing or failed")
		}
		var output textOutput
		readRetainedEvidence(t, store, report.Observations[0].Output, &output)
		if output.Text != testCase.Text {
			t.Fatalf("exact case %s differs", testCase.Name)
		}
		ordered[i] = report
	}
	merged, err := mergeShardReports(ordered)
	if err != nil || merged.ID != retained.Report {
		t.Fatalf("retained report differs: %v", err)
	}
	var recovery struct {
		Plan, Report artifact.ID
		Loads        int `json:"new_model_loads"`
		Calls        int `json:"new_generation_calls"`
	}
	readRetainedEvidence(t, store, retained.Inputs["qwen-recovered-report.json"], &recovery)
	if recovery.Plan != retained.Plan || recovery.Report != retained.Report || recovery.Loads != 0 || recovery.Calls != 0 {
		t.Fatal("recovery repeated acquisition or replaced report")
	}
	nativeBytes, err := os.ReadFile(filepath.Join(root, "docs/verification/smoke-qwen05-reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(nativeBytes)) != "86a7eb02d634614aecc50de372aeac736329d74f56532972cd4165259a6d1573" {
		t.Fatal("native likelihood reference differs")
	}
	var native struct {
		Cases  []retainedNativeSequence
		Source map[string]string `json:"source_sha256"`
	}
	if err = json.Unmarshal(nativeBytes, &native); err != nil {
		t.Fatal(err)
	}
	if len(native.Cases) != 2 || len(acquisition.Likelihoods) != len(native.Cases) {
		t.Fatal("native sequence denominator differs")
	}
	for i, c := range native.Cases {
		if err := checkRetainedNativeSequence(c, acquisition.Likelihoods[i]); err != nil {
			t.Fatal(err)
		}
		for _, mutation := range []string{"omitted", "token", "position", "order", "nonfinite"} {
			t.Run(fmt.Sprintf("sequence %d rejects %s", i, mutation), func(t *testing.T) {
				altered := acquisition.Likelihoods[i]
				altered.Scores = slices.Clone(altered.Scores)
				switch mutation {
				case "omitted":
					altered.Scores = altered.Scores[1:]
				case "token":
					altered.Scores[0].TokenID++
				case "position":
					altered.Scores[0].Position++
				case "order":
					altered.Scores[0].NegativeLogLik = altered.Scores[1].NegativeLogLik
				case "nonfinite":
					altered.Scores[0].NegativeLogLik = math.NaN()
				}
				if checkRetainedNativeSequence(c, altered) == nil {
					t.Fatal("accepted corrupted native result")
				}
			})
		}
	}
	if len(acquisition.Continuations) != len(native.Cases) || !slices.Equal(acquisition.Continuations, acquisition.Reversed) {
		t.Fatal("candidate-order isolation differs")
	}
	selected, err := sequencescore.Select(acquisition.Continuations, sequencescore.NormalizationSum)
	if err != nil || selected.Index != 0 || selected.Tied {
		t.Fatal("native continuation ordering differs")
	}
	var binding struct {
		Files   map[string]string `json:"file_sha256"`
		Tensors []struct {
			Name, SHA256 string
			Bytes        uint64
		}
		Bytes uint64 `json:"tensor_bytes"`
		Loads int    `json:"model_loads"`
	}
	readRetainedEvidence(t, store, retained.Inputs["qwen-conversion-binding.json"], &binding)
	sourceHashes := map[string]bool{}
	for _, hash := range binding.Files {
		sourceHashes[hash] = true
	}
	if !sourceHashes[native.Source["model.safetensors"]] || !sourceHashes[native.Source["config.json"]] || !sourceHashes["764daf929f9d93ae9b2ffe886e6bb5871dd5cb85ebb5e0d94700c2158931e471"] || binding.Loads != 0 || len(binding.Tensors) != 290 {
		t.Fatal("native conversion binding differs")
	}
	seen := map[string]bool{}
	var total uint64
	for _, tensor := range binding.Tensors {
		if tensor.Name == "" || seen[tensor.Name] || len(tensor.SHA256) != sha256.Size*2 || tensor.Bytes == 0 {
			t.Fatal("tensor denominator differs")
		}
		seen[tensor.Name] = true
		total += tensor.Bytes
	}
	if total != binding.Bytes || total != 988120832 {
		t.Fatal("tensor byte denominator differs")
	}
	var metadata struct {
		Model     artifact.ID
		Inventory artifact.ID `json:"tensor_inventory"`
		Matched   []string    `json:"matched_metadata"`
		Different []struct {
			Key     string
			Present bool
		} `json:"different_metadata"`
		Components []struct {
			Role     string
			Artifact artifact.ID
		}
	}
	readRetainedEvidence(t, store, retained.Inputs["qwen-metadata-binding.json"], &metadata)
	resolved, err := modelrecipe.ResolveModelDefinition(t.Context(), store, definition)
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Model != model || metadata.Inventory != resolved.Document.TensorInventory || len(metadata.Components) != 1 || metadata.Components[0].Role != "weights" || metadata.Components[0].Artifact.String() != "tensor-set:sha256:764daf929f9d93ae9b2ffe886e6bb5871dd5cb85ebb5e0d94700c2158931e471" {
		t.Fatal("converted container does not bind the activated model")
	}
	expectedMetadata := []string{"general.architecture", "general.name", "qwen2.block_count", "qwen2.context_length", "qwen2.embedding_length", "qwen2.feed_forward_length", "qwen2.attention.head_count", "qwen2.attention.head_count_kv", "qwen2.attention.layer_norm_rms_epsilon", "qwen2.rope.freq_base", "qwen2.rope.dimension_count", "qwen2.vocab_size", "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"}
	if !slices.Equal(metadata.Matched, expectedMetadata) || len(metadata.Different) != 2 || metadata.Different[0].Key != "qwen2.attention.key_length" || metadata.Different[1].Key != "qwen2.attention.value_length" || metadata.Different[0].Present || metadata.Different[1].Present {
		t.Fatal("native metadata comparison differs")
	}
	if resolved.Spec.HeadCount == 0 || resolved.Spec.EmbeddingLength%resolved.Spec.HeadCount != 0 || resolved.Spec.KeyLength != resolved.Spec.EmbeddingLength/resolved.Spec.HeadCount || resolved.Spec.ValueLength != resolved.Spec.KeyLength {
		t.Fatal("omitted head dimensions no longer derive the native geometry")
	}
	t.Log("4 exact cases; 12 native continuation token/rank scores; reversed candidate order; 290 converted tensors; zero model loads or writes")
}
