package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/sampling"
)

// ResolveModelConfig finds the latest committed model-config declaration
// for a model identity; absent means the model ships no declaration and
// the runtime policy stands unchanged.
func ResolveModelConfig(ctx context.Context, store overgodb.DocumentReader, model artifact.ID) (modelartifact.ModelConfigDocument, bool, error) {
	if ctx == nil || store == nil || model.Kind() != artifact.KindModel {
		return modelartifact.ModelConfigDocument{}, false, errors.New("model recipe: incomplete model-config lookup")
	}
	var found modelartifact.ModelConfigDocument
	located := false
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindProfile, MediaType: modelartifact.ModelConfigMediaType, Schema: modelartifact.ModelConfigSchema,
		}},
		Order: overgodb.DocumentNewestFirst,
	}, modelartifact.ParseModelConfigDocument, func(_ overgodb.DocumentView, document modelartifact.ModelConfigDocument) error {
		if located || document.Model != model {
			return nil
		}
		found, located = document, true
		return nil
	})
	if err != nil {
		return modelartifact.ModelConfigDocument{}, false, err
	}
	return found, located, nil
}

// ApplyDeclaredSampling overlays a model's declared sampling onto both
// request policies: a declared field replaces the catalog value, an
// undeclared (zero) field leaves it standing, and a checkpoint that
// declares do_sample=false asks for greedy decoding (temperature zero).
// The declaration is the model's own recommendation; the catalog policy
// is the fallback for models that ship none. The overlaid policy is a new
// document, so it is re-identified from its content: the identity the
// server validates and records names exactly the bytes in force.
func ApplyDeclaredSampling(policy *RuntimePolicy, declared *modelartifact.GenerationSampling) error {
	if policy == nil || declared == nil {
		return nil
	}
	for _, target := range []*sampling.Policy{&policy.Interactive.Sampling, &policy.Serving.Sampling} {
		if !declared.DoSample {
			target.Temperature = 0
		} else if declared.Temperature != 0 {
			target.Temperature = float32(declared.Temperature)
		}
		if declared.TopP != 0 {
			target.TopP = float32(declared.TopP)
		}
		if declared.TopK != 0 {
			target.TopK = declared.TopK
		}
		if declared.MinP != 0 {
			target.MinP = float32(declared.MinP)
		}
		if declared.PresencePenalty != 0 {
			target.PresencePenalty = float32(declared.PresencePenalty)
		}
		if declared.RepetitionPenalty != 0 {
			target.RepeatPenalty = float32(declared.RepetitionPenalty)
		}
	}
	derived, err := runtimePolicyCodec.New(*policy)
	if err != nil {
		return err
	}
	*policy = derived
	return nil
}
