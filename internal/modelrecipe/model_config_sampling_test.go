package modelrecipe

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
)

func TestApplyDeclaredSamplingOverlaysDeclaredFieldsOnly(t *testing.T) {
	policy, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy = %v, %v", found, err)
	}
	catalogTopK := policy.Serving.Sampling.TopK
	catalogMinP := policy.Serving.Sampling.MinP
	policy, err = withDeclaredSampling(policy, &modelartifact.GenerationSampling{
		Origin: "generation_config.json", DoSample: true, Temperature: 0.7, TopP: 0.8, PresencePenalty: 1.5,
	})
	if err != nil {
		t.Fatal(err)
	}
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
	policy, err = withDeclaredSampling(policy, &modelartifact.GenerationSampling{Origin: "generation_config.json", DoSample: false})
	if err != nil {
		t.Fatal(err)
	}
	if policy.Serving.Sampling.Temperature != 0 || policy.Interactive.Sampling.Temperature != 0 {
		t.Fatalf("do_sample=false did not select greedy decoding: %+v", policy.Serving.Sampling)
	}
	unchanged := policy.Serving.Sampling.Config()
	policy, err = withDeclaredSampling(policy, nil)
	if err != nil {
		t.Fatal(err)
	}
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
		file, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		metadata := append([]gguf.Metadata{testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen4exp")}, sampling...)
		if err := errors.Join(gguf.Write(file, metadata, nil, gguf.WriteOptions{}), file.Close()); err != nil {
			t.Fatal(err)
		}
		opened, err := gguf.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = opened.Close() })
		return opened
	}
	flashNext := open(
		testutil.GGUFScalar("general.sampling.top_k", gguf.ValueTypeInt32, int32(20)),
		testutil.GGUFScalar("general.sampling.top_p", gguf.ValueTypeFloat32, float32(0.95)),
		testutil.GGUFScalar("general.sampling.temp", gguf.ValueTypeFloat32, float32(1)),
	)
	policy, err := withDeclaredSampling(catalog, fileDeclaredSampling(flashNext))
	if err != nil {
		t.Fatal(err)
	}
	for _, overlaid := range []sampling.Policy{policy.Serving.Sampling, policy.Interactive.Sampling} {
		if overlaid.Temperature != 1 || overlaid.TopP != 0.95 || overlaid.TopK != 20 || overlaid.MinP != catalog.Serving.Sampling.MinP {
			t.Fatalf("the file's sampling did not overlay the catalog's: %+v", overlaid)
		}
	}
	if declared := fileDeclaredSampling(open()); declared != nil {
		t.Fatalf("a file naming no sampling declared %+v", declared)
	}
	greedy, err := withDeclaredSampling(catalog, fileDeclaredSampling(open(testutil.GGUFScalar("general.sampling.temp", gguf.ValueTypeFloat32, float32(0)))))
	if err != nil || greedy.Serving.Sampling.Temperature != 0 {
		t.Fatalf("a file's zero temperature = %+v, %v", greedy.Serving.Sampling, err)
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
	policy, err = withDeclaredSampling(policy, declaration)
	if err != nil {
		t.Fatal(err)
	}
	if err := policy.ValidateIdentity(); err != nil {
		t.Fatalf("declared sampling policy cannot enter the HTTP handler: %v", err)
	}
	if policy.ID == original.ID || original.ValidateIdentity() != nil {
		t.Fatal("overlay reused or mutated catalog identity")
	}
	repeated, err := withDeclaredSampling(original, declaration)
	if err != nil || repeated.ID != policy.ID {
		t.Fatalf("identical declaration changed policy identity: %v", err)
	}
	for _, invalid := range []float64{math.NaN(), math.Inf(1), -1} {
		declaration.Temperature = invalid
		if _, err := withDeclaredSampling(original, declaration); err == nil {
			t.Fatalf("accepted invalid declared temperature %v", invalid)
		}
		if original.ValidateIdentity() != nil {
			t.Fatal("failed overlay mutated catalog policy")
		}
	}
}
