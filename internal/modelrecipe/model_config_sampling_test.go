package modelrecipe

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestApplyDeclaredSamplingOverlaysDeclaredFieldsOnly(t *testing.T) {
	policy, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy = %v, %v", found, err)
	}
	catalogTopK := policy.Serving.Sampling.TopK
	catalogMinP := policy.Serving.Sampling.MinP
	policy = testcheck.Must(withDeclaredSampling(policy, &modelartifact.GenerationSampling{
		Origin: "generation_config.json", DoSample: true, Temperature: 0.7, TopP: 0.8, PresencePenalty: 1.5,
	}, nil))(t)
	for _, overlaid := range []sampling.Policy{policy.Interactive.Sampling, policy.Serving.Sampling} {
		if overlaid.Temperature != 0.7 || overlaid.TopP != 0.8 || overlaid.PresencePenalty != 1.5 {
			t.Fatalf("declared fields not applied: %+v", overlaid)
		}
		if overlaid.TopK != catalogTopK || overlaid.MinP != catalogMinP {
			t.Fatalf("undeclared fields changed: %+v", overlaid)
		}
	}
	if err := policy.Serving.Sampling.Validate(); err != nil {
		t.Fatalf("overlaid policy is invalid: %v", err)
	}
	policy = testcheck.Must(withDeclaredSampling(policy, &modelartifact.GenerationSampling{Origin: "generation_config.json", DoSample: false}, nil))(t)
	if policy.Serving.Sampling.Temperature != 0 || policy.Interactive.Sampling.Temperature != 0 {
		t.Fatalf("do_sample=false did not select greedy decoding: %+v", policy.Serving.Sampling)
	}
	unchanged := policy.Serving.Sampling.Config()
	policy = testcheck.Must(withDeclaredSampling(policy, nil, nil))(t)
	if !reflect.DeepEqual(policy.Serving.Sampling.Config(), unchanged) {
		t.Fatal("nil declaration changed the policy")
	}
}

// TestFilesSamplingRecommendationOverlaysTheCatalog: a GGUF file's general.sampling keys, as Flash-Next's file writes
// them, overlay the catalog's serving and interactive sampling, so a request naming none decodes at the file's values;
// a file naming none leaves the catalog's; a file's temperature of zero asks for greedy decoding.
func TestFilesSamplingRecommendationOverlaysTheCatalog(t *testing.T) {
	catalog, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy = %v, %v", found, err)
	}
	open := func(sampling ...gguf.Metadata) *gguf.File {
		t.Helper()
		path := filepath.Join(t.TempDir(), "model.gguf")
		file := testcheck.Must(os.Create(path))(t)
		metadata := append([]gguf.Metadata{testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen4exp")}, sampling...)
		testcheck.NoError(t, errors.Join(gguf.Write(file, metadata, nil, gguf.WriteOptions{}), file.Close()))
		opened := testcheck.Must(gguf.Open(path))(t)
		t.Cleanup(func() { _ = opened.Close() })
		return opened
	}
	flashNext := open(
		testutil.GGUFScalar("general.sampling.top_k", gguf.ValueTypeInt32, int32(20)),
		testutil.GGUFScalar("general.sampling.top_p", gguf.ValueTypeFloat32, float32(0.95)),
		testutil.GGUFScalar("general.sampling.temp", gguf.ValueTypeFloat32, float32(1)),
	)
	policy := testcheck.Must(withDeclaredSampling(catalog, fileDeclaredSampling(flashNext), nil))(t)
	for _, overlaid := range []sampling.Policy{policy.Serving.Sampling, policy.Interactive.Sampling} {
		if overlaid.Temperature != 1 || overlaid.TopP != 0.95 || overlaid.TopK != 20 || overlaid.MinP != catalog.Serving.Sampling.MinP {
			t.Fatalf("the file's sampling did not overlay the catalog's: %+v", overlaid)
		}
	}
	if declared := fileDeclaredSampling(open()); declared != nil {
		t.Fatalf("a file naming no sampling declared %+v", declared)
	}
	greedy, err := withDeclaredSampling(catalog, fileDeclaredSampling(open(testutil.GGUFScalar("general.sampling.temp", gguf.ValueTypeFloat32, float32(0)))), nil)
	if err != nil || greedy.Serving.Sampling.Temperature != 0 {
		t.Fatalf("a file's zero temperature = %+v, %v", greedy.Serving.Sampling, err)
	}
	for _, malformed := range []float32{float32(math.NaN()), float32(math.Inf(-1))} {
		declared := fileDeclaredSampling(open(testutil.GGUFScalar("general.sampling.temp", gguf.ValueTypeFloat32, malformed)))
		if policy, err := withDeclaredSampling(catalog, declared, nil); err == nil {
			t.Fatalf("a file's temperature %v was accepted as %+v", malformed, policy.Serving.Sampling)
		}
	}
}

func TestDeclaredSamplingPreservesPolicyIdentity(t *testing.T) {
	policy, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy: %v", err)
	}
	original := policy
	declaration := &modelartifact.GenerationSampling{
		Origin: "generation_config.json", DoSample: true, Temperature: 0.7, TopP: 0.8,
	}
	policy = testcheck.Must(withDeclaredSampling(policy, declaration, nil))(t)
	if err := policy.ValidateIdentity(); err != nil {
		t.Fatalf("declared sampling policy cannot enter the HTTP handler: %v", err)
	}
	if policy.ID == original.ID || original.ValidateIdentity() != nil {
		t.Fatal("overlay reused or mutated catalog identity")
	}
	repeated, err := withDeclaredSampling(original, declaration, nil)
	if err != nil || repeated.ID != policy.ID {
		t.Fatalf("identical declaration changed policy identity: %v", err)
	}
	for _, invalid := range []float64{math.NaN(), math.Inf(1), -1} {
		declaration.Temperature = invalid
		if _, err := withDeclaredSampling(original, declaration, nil); err == nil {
			t.Fatalf("accepted invalid declared temperature %v", invalid)
		}
		if original.ValidateIdentity() != nil {
			t.Fatal("failed overlay mutated catalog policy")
		}
	}
}

// TestDeclaredSamplingIsBoundToTheRun: a model's declared sampling is outside its recipe's identity and can change after
// activation, so two attempts of one recipe under two declarations are two records, each naming a policy the store holds.
func TestDeclaredSamplingIsBoundToTheRun(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := openProgramStore(t)
	defer store.Close()
	catalog, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy: %v", err)
	}
	attempt := func(temperature float64) runrecord.ServingObservation {
		t.Helper()
		policy, err := withDeclaredSampling(catalog, &modelartifact.GenerationSampling{Origin: "generation_config.json", DoSample: true, Temperature: temperature}, nil)
		testcheck.NoError(t, err)
		observation := runrecord.ServingObservation{
			Version: artifact.InitialDocumentVersion, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1,
			Model:       testutil.ArtifactID(t, artifact.KindModel, "declared"),
			Recipe:      testutil.ArtifactID(t, artifact.KindRecipe, "one recipe"),
			Environment: testutil.ArtifactID(t, artifact.KindEvidence, "one machine"),
		}
		testcheck.NoError(t, BindServingPolicy(ctx, store, &observation, policy))
		if observation, err = runrecord.NewServingObservation(observation); err != nil {
			t.Fatal(err)
		}
		if _, err := runtimePolicyCodec.Require(ctx, store, observation.Policy); err != nil {
			t.Fatalf("the attempt names a policy the store does not hold: %v", err)
		}
		return observation
	}
	if warm, cool := attempt(0.9), attempt(0.3); warm.ID == cool.ID || warm.Policy == cool.Policy {
		t.Fatalf("two declarations made one record: %v and %v", warm.ID, cool.ID)
	}
	if first, again := attempt(0.9), attempt(0.9); first.ID != again.ID {
		t.Fatalf("one declaration made two records: %v and %v", first.ID, again.ID)
	}
}

// TestADeclaredPenaltyReachesTheSampler: a window of 0 idles every penalty, so a penalty a model declares opens the window to
// the text so far, and a neutral one leaves it shut.
func TestADeclaredPenaltyReachesTheSampler(t *testing.T) {
	t.Parallel()
	catalog, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy: %v", err)
	}
	for _, declared := range []struct {
		sampling modelartifact.GenerationSampling
		idle     bool
	}{
		{modelartifact.GenerationSampling{DoSample: true, RepetitionPenalty: 1.1}, false},
		{modelartifact.GenerationSampling{DoSample: true, PresencePenalty: 0.5}, false},
		{modelartifact.GenerationSampling{DoSample: true, RepetitionPenalty: 1}, true},
		{modelartifact.GenerationSampling{DoSample: true}, true},
	} {
		declared.sampling.Origin = "generation_config.json"
		policy := testcheck.Must(withDeclaredSampling(catalog, &declared.sampling, nil))(t)
		for _, overlaid := range []sampling.Policy{policy.Serving.Sampling, policy.Interactive.Sampling} {
			if sampling.PenaltiesIdle(overlaid.Config()) != declared.idle {
				t.Errorf("%+v leaves the penalties idle %t, want %t", declared.sampling, !declared.idle, declared.idle)
			}
			if err := overlaid.Validate(); err != nil {
				t.Errorf("%+v: %v", declared.sampling, err)
			}
		}
	}
}
