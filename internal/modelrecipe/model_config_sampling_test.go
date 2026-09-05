package modelrecipe

import (
	"reflect"
	"testing"

	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
)

func TestApplyDeclaredSamplingOverlaysDeclaredFieldsOnly(t *testing.T) {
	policy, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy = %v, %v", found, err)
	}
	catalogTopK := policy.Serving.Sampling.TopK
	catalogMinP := policy.Serving.Sampling.MinP
	catalogID := policy.ID
	if err := ApplyDeclaredSampling(&policy, &modelartifact.GenerationSampling{
		Origin: "generation_config.json", DoSample: true, Temperature: 0.7, TopP: 0.8, PresencePenalty: 1.5,
	}); err != nil {
		t.Fatal(err)
	}
	// The overlaid policy is a new document: it re-identifies from its
	// content, so the identity the server validates names these bytes.
	if policy.ID == catalogID || policy.ValidateIdentity() != nil {
		t.Fatalf("overlaid policy identity = %s (catalog %s): %v", policy.ID, catalogID, policy.ValidateIdentity())
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
	if err := ApplyDeclaredSampling(&policy, &modelartifact.GenerationSampling{Origin: "generation_config.json", DoSample: false}); err != nil {
		t.Fatal(err)
	}
	if policy.Serving.Sampling.Temperature != 0 || policy.Interactive.Sampling.Temperature != 0 {
		t.Fatalf("do_sample=false did not select greedy decoding: %+v", policy.Serving.Sampling)
	}
	unchanged := policy.Serving.Sampling.Config()
	if err := ApplyDeclaredSampling(&policy, nil); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(policy.Serving.Sampling.Config(), unchanged) {
		t.Fatal("nil declaration changed the policy")
	}
}
