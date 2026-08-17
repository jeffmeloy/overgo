// Tier-0 chain viability: compose two whole validated models through typed
// workflowrecipe ports -- the drafter's generated text feeds the scorer's
// tokenizer over a text edge, no latent bridging -- and measure whether the
// chain's context beats the scorer's own single-model context on held-out
// continuation cross-entropy. The verdict contract mirrors RunViability: SHIP
// only if every window separates strictly below the single-model baseline;
// anything else is a measured refusal, never a silent pass. Both arms execute
// through the generic workflow runtime as content-addressed recipe artifacts,
// so every probe run is replayable from its committed run records.
package composition

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/hostmath"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/workflowrecipe"
	"overgo/internal/workflowruntime"
)

// ChainConfig: the pre-written Tier-0 protocol. Windows are disjoint slices of
// a recorded scorer-token stream; each window splits into a Prefix context, a
// Draft gap both arms must fill generatively, and a Target the scorer predicts.
type ChainConfig struct {
	ScorerDir  string
	DrafterDir string
	Prefix     int
	Draft      int
	Target     int
	Windows    [][]int
}

// ChainWindowOutcome: one window's measured arms. ChainText is the exact
// context text the two-model chain handed the scorer.
type ChainWindowOutcome struct {
	Window     int
	BaselineCE float64
	ChainCE    float64
	ChainText  string
}

// ChainResult: verdict plus the content-addressed recipe identities that make
// the probe replayable.
type ChainResult struct {
	ChainRecipe    artifact.ID
	BaselineRecipe artifact.ID
	Outcomes       []ChainWindowOutcome
	Ship           bool
	Reason         string
}

type chainModel struct {
	model     *densecausal.Model
	tokenizer *hfbpe.Tokenizer
}

// RunChainViability executes the full Tier-0 experiment. It returns an error
// only when the experiment could not run; a refusal is a successful experiment
// whose verdict is Ship=false with the measured reason.
func RunChainViability(store *repodb.Store, config ChainConfig) (ChainResult, error) {
	if store == nil {
		return ChainResult{}, fmt.Errorf("composition: chain viability requires a store")
	}
	if config.Prefix < 2 || config.Draft < 1 || config.Target < 2 || len(config.Windows) == 0 {
		return ChainResult{}, fmt.Errorf("composition: chain prefix, draft, target and windows are required")
	}
	span := config.Prefix + config.Draft + config.Target
	for index, window := range config.Windows {
		if len(window) < span {
			return ChainResult{}, fmt.Errorf("composition: window %d has %d tokens, need %d", index, len(window), span)
		}
	}
	scorerID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("tier0-chain/v1:model:"+config.ScorerDir))
	if err != nil {
		return ChainResult{}, err
	}
	drafterID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("tier0-chain/v1:model:"+config.DrafterDir))
	if err != nil {
		return ChainResult{}, err
	}
	models := map[artifact.ID]*chainModel{}
	for id, dir := range map[artifact.ID]string{scorerID: config.ScorerDir, drafterID: config.DrafterDir} {
		loaded, err := densecausal.Load(dir)
		if err != nil {
			return ChainResult{}, fmt.Errorf("composition: load chain model: %w", err)
		}
		tokenizer, err := hfbpe.Load(dir)
		if err != nil {
			return ChainResult{}, fmt.Errorf("composition: load chain tokenizer: %w", err)
		}
		models[id] = &chainModel{model: loaded, tokenizer: tokenizer}
	}
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "tier0-chain/models",
		Artifacts: []artifact.Descriptor{{ID: scorerID}, {ID: drafterID}},
	}); err != nil {
		return ChainResult{}, err
	}
	chainProgram, baseProgram, err := ChainPrograms(scorerID, drafterID)
	if err != nil {
		return ChainResult{}, err
	}
	runtime, err := workflowruntime.NewForProgram(store, chainProgram)
	if err != nil {
		return ChainResult{}, err
	}
	if err := registerChainAdapters(runtime, models, config.Draft); err != nil {
		return ChainResult{}, err
	}
	scorer := models[scorerID]
	result := ChainResult{
		ChainRecipe:    chainProgram.Definition().ID,
		BaselineRecipe: baseProgram.Definition().ID,
	}
	shipped := 0
	for index, window := range config.Windows {
		prefixText := scorer.tokenizer.Decode(window[:config.Prefix])
		target := window[config.Prefix+config.Draft : span]
		// Batch keys carry the full protocol identity (recipes fix the model
		// pair; prefix/draft/target and window fix the data), so distinct
		// configurations sharing one store never collide and identical
		// replays deduplicate.
		protocol := fmt.Sprintf("%s/p%d-d%d-t%d/w%d",
			result.ChainRecipe, config.Prefix, config.Draft, config.Target, index)
		chainCtx, chainText, err := executeChainArm(
			ctx, runtime, chainProgram, "tier0-chain/chain/"+protocol, prefixText)
		if err != nil {
			return ChainResult{}, fmt.Errorf("composition: window %d chain arm: %w", index, err)
		}
		baseCtx, _, err := executeChainArm(
			ctx, runtime, baseProgram, "tier0-chain/baseline/"+protocol, prefixText)
		if err != nil {
			return ChainResult{}, fmt.Errorf("composition: window %d baseline arm: %w", index, err)
		}
		baselineCE, err := suffixCE(scorer.model, baseCtx, target)
		if err != nil {
			return ChainResult{}, fmt.Errorf("composition: window %d baseline score: %w", index, err)
		}
		chainCE, err := suffixCE(scorer.model, chainCtx, target)
		if err != nil {
			return ChainResult{}, fmt.Errorf("composition: window %d chain score: %w", index, err)
		}
		result.Outcomes = append(result.Outcomes, ChainWindowOutcome{
			Window: index, BaselineCE: baselineCE, ChainCE: chainCE, ChainText: chainText,
		})
		if chainCE < baselineCE {
			shipped++
		}
	}
	if shipped == len(config.Windows) {
		result.Ship = true
		result.Reason = fmt.Sprintf(
			"envelope separation: chain held-out CE beats the single-model baseline on all %d windows",
			len(config.Windows))
	} else {
		result.Reason = fmt.Sprintf(
			"refusal: chain context beat the single-model baseline on %d of %d windows; failure mode: drafter context did not transfer predictive gain",
			shipped, len(config.Windows))
	}
	return result, nil
}

// ChainPrograms compiles the two content-addressed probe recipes: the
// two-model chain (drafter text into scorer tokenizer over a typed text edge)
// and the single-model baseline (the scorer drafting its own gap).
func ChainPrograms(scorer, drafter artifact.ID) (recipe.Program, recipe.Program, error) {
	node := func(id recipe.NodeID, module recipe.ModuleID, slot uint32) recipe.Node {
		return recipe.Node{ID: id, Module: module, Placement: recipe.PlacementHost, ModelSlot: slot}
	}
	edge := func(from recipe.NodeID, fromPort recipe.PortName, to recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{
			From: recipe.Endpoint{Node: from, Port: fromPort},
			To:   recipe.Endpoint{Node: to, Port: toPort},
		}
	}
	chainDefinition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Slot: 0, Artifact: scorer},
			{Role: recipe.DependencyModel, Slot: 1, Artifact: drafter},
		},
		[]recipe.Node{
			node("draft-tokenize", workflowrecipe.ModuleTokenize, 1),
			node("draft-generate", workflowrecipe.ModuleGenerate, 1),
			node("draft-detokenize", workflowrecipe.ModuleDetokenize, 1),
			node("score-tokenize", workflowrecipe.ModuleTokenize, 0),
		},
		[]recipe.Edge{
			edge("draft-tokenize", "tokens", "draft-generate", "tokens"),
			edge("draft-generate", "tokens", "draft-detokenize", "tokens"),
			edge("draft-detokenize", "text", "score-tokenize", "text"),
		},
		[]recipe.Input{{
			Name: "prompt", Data: recipe.DataText,
			Target: recipe.Endpoint{Node: "draft-tokenize", Port: "text"},
		}},
		[]recipe.Output{
			{Name: "context", Data: recipe.DataTokens, Source: recipe.Endpoint{Node: "score-tokenize", Port: "tokens"}},
			{Name: "context-text", Data: recipe.DataText, Source: recipe.Endpoint{Node: "draft-detokenize", Port: "text"}},
		},
	)
	if err != nil {
		return recipe.Program{}, recipe.Program{}, err
	}
	baseDefinition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Slot: 0, Artifact: scorer}},
		[]recipe.Node{
			node("self-tokenize", workflowrecipe.ModuleTokenize, 0),
			node("self-generate", workflowrecipe.ModuleGenerate, 0),
		},
		[]recipe.Edge{edge("self-tokenize", "tokens", "self-generate", "tokens")},
		[]recipe.Input{{
			Name: "prompt", Data: recipe.DataText,
			Target: recipe.Endpoint{Node: "self-tokenize", Port: "text"},
		}},
		[]recipe.Output{{
			Name: "context", Data: recipe.DataTokens, Source: recipe.Endpoint{Node: "self-generate", Port: "tokens"},
		}},
	)
	if err != nil {
		return recipe.Program{}, recipe.Program{}, err
	}
	chainProgram, err := recipe.CompileProgram(chainDefinition, workflowrecipe.Catalog())
	if err != nil {
		return recipe.Program{}, recipe.Program{}, err
	}
	baseProgram, err := recipe.CompileProgram(baseDefinition, workflowrecipe.Catalog())
	if err != nil {
		return recipe.Program{}, recipe.Program{}, err
	}
	return chainProgram, baseProgram, nil
}

func registerChainAdapters(
	runtime *workflowruntime.Runtime,
	models map[artifact.ID]*chainModel,
	draft int,
) error {
	resolve := func(request workflowruntime.StepRequest) (*chainModel, error) {
		bound, ok := models[request.Model]
		if !ok {
			return nil, fmt.Errorf("composition: step model %s is not loaded", request.Model)
		}
		return bound, nil
	}
	register := func(module recipe.ModuleID, port recipe.PortName, execute func(*chainModel, workflowruntime.StepRequest) (workflowruntime.Value, error)) error {
		return runtime.Register(module, workflowruntime.AdapterFunc(func(
			_ context.Context, request workflowruntime.StepRequest,
		) (map[recipe.PortName]workflowruntime.Value, error) {
			bound, err := resolve(request)
			if err != nil {
				return nil, err
			}
			value, err := execute(bound, request)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]workflowruntime.Value{port: value}, nil
		}))
	}
	if err := register(workflowrecipe.ModuleTokenize, "tokens",
		func(bound *chainModel, request workflowruntime.StepRequest) (workflowruntime.Value, error) {
			text, err := scalarString(request, "text")
			if err != nil {
				return workflowruntime.Value{}, err
			}
			tokens, err := bound.tokenizer.Encode(text)
			if err != nil {
				return workflowruntime.Value{}, err
			}
			return tokensValue(tokens)
		}); err != nil {
		return err
	}
	if err := register(workflowrecipe.ModuleGenerate, "tokens",
		func(bound *chainModel, request workflowruntime.StepRequest) (workflowruntime.Value, error) {
			tokens, err := scalarTokens(request, "tokens")
			if err != nil {
				return workflowruntime.Value{}, err
			}
			generated, err := greedyExtend(bound.model, tokens, draft)
			if err != nil {
				return workflowruntime.Value{}, err
			}
			return tokensValue(generated)
		}); err != nil {
		return err
	}
	return register(workflowrecipe.ModuleDetokenize, "text",
		func(bound *chainModel, request workflowruntime.StepRequest) (workflowruntime.Value, error) {
			tokens, err := scalarTokens(request, "tokens")
			if err != nil {
				return workflowruntime.Value{}, err
			}
			return textValue(bound.tokenizer.Decode(tokens))
		})
}

func executeChainArm(
	ctx context.Context,
	runtime *workflowruntime.Runtime,
	program recipe.Program,
	key, prompt string,
) ([]int, string, error) {
	promptValue, err := textValue(prompt)
	if err != nil {
		return nil, "", err
	}
	result, err := runtime.ExecuteProgram(ctx, key, program, map[recipe.PortName]workflowruntime.Value{
		"prompt": promptValue,
	})
	if err != nil {
		return nil, "", err
	}
	tokens, err := singleTokens(result.Outputs["context"])
	if err != nil {
		return nil, "", err
	}
	text := ""
	if value, ok := result.Outputs["context-text"]; ok {
		datum, scalar := value.Single()
		if !scalar {
			return nil, "", fmt.Errorf("composition: chain context text is not scalar")
		}
		text, _ = datum.Value.(string)
	}
	return tokens, text, nil
}

// greedyExtend appends draft tokens by deterministic argmax over the final
// position's logits, re-running the host-reference forward each step.
func greedyExtend(model *densecausal.Model, tokens []int, draft int) ([]int, error) {
	extended := slices.Clone(tokens)
	vocab := model.Dims.Vocab
	for step := 0; step < draft; step++ {
		_, logits, err := model.Loss(extended)
		if err != nil {
			return nil, err
		}
		final := logits[(len(extended)-1)*vocab:]
		best := 0
		for index := 1; index < vocab; index++ {
			if final[index] > final[best] {
				best = index
			}
		}
		extended = append(extended, best)
	}
	return extended, nil
}

// suffixCE: mean cross-entropy of the scorer over exactly the target
// positions, conditioned on the arm's context tokens.
func suffixCE(model *densecausal.Model, contextTokens, target []int) (float64, error) {
	if len(contextTokens) == 0 {
		return 0, fmt.Errorf("composition: empty scoring context")
	}
	sequence := append(slices.Clone(contextTokens), target...)
	_, logits, err := model.Loss(sequence)
	if err != nil {
		return 0, err
	}
	vocab := model.Dims.Vocab
	rows := len(target)
	start := len(contextTokens) - 1
	scratch := make([]float32, rows*vocab)
	return hostmath.SoftmaxCrossEntropy(
		scratch, logits[start*vocab:(start+rows)*vocab], sequence[start+1:], rows, vocab,
	), nil
}

func tokensValue(tokens []int) (workflowruntime.Value, error) {
	data, err := json.Marshal(tokens)
	if err != nil {
		return workflowruntime.Value{}, err
	}
	content, err := chainContent(data, "application/json")
	if err != nil {
		return workflowruntime.Value{}, err
	}
	return workflowruntime.ArtifactValue(recipe.DataTokens, tokens, content), nil
}

func textValue(text string) (workflowruntime.Value, error) {
	content, err := chainContent([]byte(text), "text/plain")
	if err != nil {
		return workflowruntime.Value{}, err
	}
	return workflowruntime.ArtifactValue(recipe.DataText, text, content), nil
}

func chainContent(data []byte, mediaType string) (artifact.Content, error) {
	id, err := artifact.IdentifyBytes(artifact.KindOutput, data)
	if err != nil {
		return artifact.Content{}, err
	}
	return artifact.Content{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: mediaType},
		Data:       data,
	}, nil
}

func scalarString(request workflowruntime.StepRequest, name recipe.PortName) (string, error) {
	datum, ok := request.Inputs[name].Single()
	if !ok {
		return "", fmt.Errorf("composition: input %q is not scalar", name)
	}
	text, ok := datum.Value.(string)
	if !ok {
		return "", fmt.Errorf("composition: input %q is not text", name)
	}
	return text, nil
}

func scalarTokens(request workflowruntime.StepRequest, name recipe.PortName) ([]int, error) {
	return singleTokens(request.Inputs[name])
}

func singleTokens(value workflowruntime.Value) ([]int, error) {
	datum, ok := value.Single()
	if !ok {
		return nil, fmt.Errorf("composition: token value is not scalar")
	}
	tokens, ok := datum.Value.([]int)
	if !ok {
		return nil, fmt.Errorf("composition: token value has invalid type")
	}
	return tokens, nil
}
