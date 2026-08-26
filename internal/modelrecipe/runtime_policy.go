package modelrecipe

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
)

const (
	runtimePolicyVersion uint16 = 2
)

//go:embed runtime_policies.json
var runtimePolicyCatalogJSON []byte

type videoInputPolicy struct {
	FPS       float64 `json:"fps"`
	MaxFrames int     `json:"max_frames"`
}

type interactiveRuntimePolicy struct {
	OutputTokens int              `json:"output_tokens"`
	Sampling     sampling.Policy  `json:"sampling"`
	Video        videoInputPolicy `json:"video"`
	Diffusion    diffusionPolicy  `json:"diffusion"`
}

type diffusionPolicy struct {
	Length      int     `json:"length"`
	Steps       int     `json:"steps"`
	Algorithm   int     `json:"algorithm"`
	Temperature float32 `json:"temperature"`
	TopK        int     `json:"top_k"`
	TopP        float32 `json:"top_p"`
}

type requestLimitPolicy struct {
	StopSequences           int `json:"stop_sequences"`
	ResponseFields          int `json:"response_fields"`
	ResponseFieldBytes      int `json:"response_field_bytes"`
	ResponseFieldComponents int `json:"response_field_components"`
}

type servingRuntimePolicy struct {
	ModelID            string             `json:"model_id"`
	OutputTokens       int                `json:"output_tokens"`
	MaxTokens          int                `json:"max_tokens"`
	MaxConcurrent      int                `json:"max_concurrent"`
	PromptCacheEntries int                `json:"prompt_cache_entries"`
	MaxEmbeddingInputs int                `json:"max_embedding_inputs"`
	StoredResponses    int                `json:"stored_responses"`
	ResponseStoreBytes int                `json:"response_store_bytes"`
	Sampling           sampling.Policy    `json:"sampling"`
	Video              videoInputPolicy   `json:"video"`
	Limits             requestLimitPolicy `json:"limits"`
}

// RuntimePolicy binds request defaults to supported recipe tasks.
type RuntimePolicy struct {
	ID          artifact.ID              `json:"-"`
	Version     uint16                   `json:"version"`
	Tasks       []recipe.Task            `json:"tasks"`
	Interactive interactiveRuntimePolicy `json:"interactive"`
	Serving     servingRuntimePolicy     `json:"serving"`
}

var runtimePolicyCodec = artifact.DocumentCodec[RuntimePolicy]{
	Name: "runtime policy",
	Contract: artifact.DocumentContract{
		Kind:      artifact.KindProfile,
		MediaType: "application/vnd.overgo.runtime-policy+json", Schema: "overgo/runtime-policy/v2",
	},
	Decode:       func(data []byte, value *RuntimePolicy) error { return strictjson.DecodeBytes(data, value) },
	Encode:       func(value RuntimePolicy) ([]byte, error) { return json.Marshal(value) },
	Canonicalize: canonicalizeRuntimePolicy,
	Identity:     func(value RuntimePolicy) artifact.ID { return value.ID },
	SetIdentity:  func(value *RuntimePolicy, id artifact.ID) { value.ID = id },
}

// CatalogRuntimePolicy returns the shared policy for a recipe task.
func CatalogRuntimePolicy(task recipe.Task) (RuntimePolicy, bool, error) {
	var catalog []RuntimePolicy
	if err := strictjson.DecodeBytes(runtimePolicyCatalogJSON, &catalog); err != nil {
		return RuntimePolicy{}, false, fmt.Errorf("model recipe: decode runtime policy catalog: %w", err)
	}
	for _, declared := range catalog {
		policy, err := runtimePolicyCodec.New(declared)
		if err != nil {
			return RuntimePolicy{}, false, err
		}
		if slices.Contains(policy.Tasks, task) {
			return policy, true, nil
		}
	}
	return RuntimePolicy{}, false, nil
}

func runtimePolicyAlias(definition artifact.ID) string {
	return "runtime-policy/v2/" + definition.String()
}

func runtimePolicyBinding(definition recipe.Definition) (artifact.Content, artifact.AliasBinding, bool, error) {
	policy, found, err := CatalogRuntimePolicy(definition.Task)
	if err != nil || !found {
		return artifact.Content{}, artifact.AliasBinding{}, false, err
	}
	content, err := runtimePolicyCodec.Content(policy)
	if err != nil {
		return artifact.Content{}, artifact.AliasBinding{}, false, err
	}
	return content, artifact.AliasBinding{Name: runtimePolicyAlias(definition.ID), Target: policy.ID}, true, nil
}

// EnsureRuntimePolicy publishes the policy selected by a recipe task.
func EnsureRuntimePolicy(ctx context.Context, store artifact.Repository, definition recipe.Definition) error {
	content, binding, found, err := runtimePolicyBinding(definition)
	if err != nil || !found {
		return err
	}
	current, bound, err := artifact.ResolveAlias(ctx, store, binding.Name)
	if err != nil {
		return err
	}
	if bound {
		if current != binding.Target {
			return errors.New("model recipe: runtime policy binding differs")
		}
		return nil
	}
	batch, err := artifact.NewDocumentBatch(
		"runtime-policy/ensure/"+definition.ID.String(), []artifact.Content{content}, nil, []artifact.AliasBinding{binding},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		return nil
	}
	return err
}

// ResolveRuntimePolicy loads the exact policy bound to a recipe.
func ResolveRuntimePolicy(ctx context.Context, store artifact.Reader, definition recipe.Definition) (RuntimePolicy, error) {
	id, found, err := artifact.ResolveAlias(ctx, store, runtimePolicyAlias(definition.ID))
	if err != nil || !found {
		return RuntimePolicy{}, errors.Join(errors.New("model recipe: active recipe has no runtime policy"), err)
	}
	policy, err := runtimePolicyCodec.Require(ctx, store, id)
	if err != nil {
		return RuntimePolicy{}, err
	}
	if !slices.Contains(policy.Tasks, definition.Task) {
		return RuntimePolicy{}, errors.New("model recipe: runtime policy does not admit recipe task")
	}
	return policy, nil
}

// ValidateIdentity checks policy content and identity.
func (policy RuntimePolicy) ValidateIdentity() error {
	return runtimePolicyCodec.ValidateIdentity(policy)
}

func canonicalizeRuntimePolicy(policy *RuntimePolicy) error {
	if policy == nil || policy.Version != runtimePolicyVersion || len(policy.Tasks) == 0 {
		return errors.New("model recipe: invalid runtime policy envelope")
	}
	slices.Sort(policy.Tasks)
	policy.Tasks = slices.Compact(policy.Tasks)
	for _, task := range policy.Tasks {
		if !task.Valid() {
			return errors.New("model recipe: runtime policy has invalid task")
		}
	}
	interactive, serving := policy.Interactive, policy.Serving
	if interactive.OutputTokens <= 0 || interactive.Video.FPS <= 0 || interactive.Video.MaxFrames <= 0 ||
		interactive.Diffusion.Length <= 0 || interactive.Diffusion.Steps <= 0 || interactive.Diffusion.Algorithm < 0 ||
		interactive.Diffusion.Temperature < 0 || interactive.Diffusion.TopK < 0 ||
		interactive.Diffusion.TopP <= 0 || interactive.Diffusion.TopP > 1 ||
		serving.ModelID == "" || serving.OutputTokens <= 0 || serving.MaxTokens < serving.OutputTokens || serving.MaxConcurrent <= 0 ||
		serving.PromptCacheEntries <= 0 || serving.MaxEmbeddingInputs <= 0 ||
		serving.StoredResponses <= 0 || serving.ResponseStoreBytes <= 0 ||
		serving.Video.FPS <= 0 || serving.Video.MaxFrames <= 0 ||
		serving.Limits.StopSequences <= 0 || serving.Limits.ResponseFields <= 0 ||
		serving.Limits.ResponseFieldBytes <= 0 || serving.Limits.ResponseFieldComponents <= 0 {
		return errors.New("model recipe: incomplete runtime policy")
	}
	if err := interactive.Sampling.Validate(); err != nil {
		return fmt.Errorf("model recipe: interactive sampling policy: %w", err)
	}
	if err := serving.Sampling.Validate(); err != nil {
		return fmt.Errorf("model recipe: serving sampling policy: %w", err)
	}
	return nil
}
