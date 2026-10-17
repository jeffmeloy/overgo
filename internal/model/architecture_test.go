package model

import "testing"

const (
	runtimePolicyFixtureArchitecture = "runtime-policy-fixture"
	runtimePolicyFixtureWidth        = uint32(16)
	runtimePolicyFixtureScale        = float32(4)
	runtimePolicyFixtureBlocks       = uint32(2)
	runtimePolicyFixtureHeads        = uint32(2)
	runtimePolicyFixtureKVHeads      = runtimePolicyFixtureHeads - 1
	runtimePolicyFixtureHeadWidth    = uint32(4)
	runtimePolicyFixtureNarrowWidth  = runtimePolicyFixtureHeadWidth / 2
	runtimePolicyFixtureTokenTypes   = uint32(1)
	runtimePolicyFixtureRank         = uint32(1)
	runtimePolicyFixturePositive     = float32(1)
	runtimePolicyFixtureTargetLayer  = int32(0)
)

func TestArchitectureRegistryProfiles(t *testing.T) {
	tests := []struct {
		name       string
		capability ArchitectureCapability
	}{
		{"llama", architectureNormalRoPE},
		{"deepseek32", architectureSparseLatent | architectureLatent},
		{"qwen35moe", architectureMoE | architectureRecurrent},
		{"qwen4exp", architectureMoE | architectureRecurrent},
		{"t5", architectureRoPEDisabled},
		{"llada", architectureNonCausal | architectureDiffusion},
		{"qwen3vl", architectureMultimodal},
		{"gemma3n", architectureSharedKV | architecturePerLayerEmbeddings},
		{"talkie", architectureEmbeddingSkip},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile, ok := LookupArchitecture(test.name)
			if !ok || profile.Name != test.name ||
				profile.Capabilities&test.capability != test.capability {
				t.Fatalf("profile = %#v", profile)
			}
		})
	}
	if _, ok := LookupArchitecture("unknown"); ok {
		t.Fatal("unknown architecture registered")
	}
	names := SupportedArchitectures()
	if len(names) != len(architectureRegistry) || len(names) < 100 {
		t.Fatalf("registered names = %d", len(names))
	}
	for index := 1; index < len(names); index++ {
		if names[index-1] >= names[index] {
			t.Fatalf("names are not sorted at %d", index)
		}
	}
}

func TestResolvedProfilePrefersExactBinding(t *testing.T) {
	bootstrap, _ := LookupArchitecture("llama")
	unbound := Spec{CommonSpec: CommonSpec{Architecture: bootstrap.Name}}
	if profile := unbound.Profile(); profile.Name != "" {
		t.Fatalf("unbound profile = %+v", profile)
	}
	bound := bootstrap
	bound.LayerTopology = LayerTopologyCausalPostQKNormSkip
	spec := Spec{CommonSpec: CommonSpec{Architecture: bootstrap.Name}}.withProfile(bound)
	resolved, ok := spec.boundProfile()
	if !ok || resolved.LayerTopology != LayerTopologyCausalPostQKNormSkip {
		t.Fatalf("resolved profile = (%+v, %t)", resolved, ok)
	}
	mismatch := spec.withProfile(ArchitectureProfile{Name: "other"})
	if _, ok := mismatch.boundProfile(); ok {
		t.Fatal("mismatched bound profile accepted")
	}
}

func TestRuntimeBehaviorUsesBoundProfilePolicies(t *testing.T) {
	profile := ArchitectureProfile{
		Name: runtimePolicyFixtureArchitecture, Normalization: NormalizationRMS,
		Rotary: rotaryPolicy{Usage: rotaryUsageSlidingOnly},
		Runtime: runtimePolicy{
			EmbeddingScale: embeddingScaleSqrtWidth, LogitScale: logitScaleDirect,
			NormalizationPlacement: normalizationPlacementPostOnly,
			NormalizationBias:      normalizationBiasAlways,
		},
	}
	spec := Spec{
		CommonSpec: CommonSpec{
			Architecture: runtimePolicyFixtureArchitecture,
			BlockCount:   runtimePolicyFixtureBlocks, EmbeddingLength: runtimePolicyFixtureWidth,
			LogitScale: runtimePolicyFixtureScale,
		},
		AttentionSpec: AttentionSpec{SlidingLayers: []bool{false, true}},
	}.withProfile(profile)
	norm := spec.NormPlan()
	if spec.InputEmbeddingScale() != runtimePolicyFixtureScale ||
		spec.OutputLogitMultiplier() != runtimePolicyFixtureScale ||
		norm.Operation != NormalizationRMS || norm.PreAttention || !norm.PostAttention || !norm.Bias ||
		spec.UsesRoPE(0) || !spec.UsesRoPE(1) {
		t.Fatalf("runtime policy result = (input=%g output=%g norm=%+v rope=%t/%t)",
			spec.InputEmbeddingScale(), spec.OutputLogitMultiplier(), norm,
			spec.UsesRoPE(0), spec.UsesRoPE(1))
	}
}

func TestEncoderValidationUsesBoundProfilePolicy(t *testing.T) {
	profile := ArchitectureProfile{
		Name:       runtimePolicyFixtureArchitecture,
		Validation: validationPolicy{Encoder: encoderValidationTokenTypesMatchingHeads},
	}
	spec := Spec{
		CommonSpec: CommonSpec{Architecture: runtimePolicyFixtureArchitecture},
		AttentionSpec: AttentionSpec{
			HeadCount: runtimePolicyFixtureHeads, HeadCountKV: runtimePolicyFixtureKVHeads,
			KeyLength: runtimePolicyFixtureHeadWidth, ValueLength: runtimePolicyFixtureHeadWidth,
		},
	}.withProfile(profile)
	if err := spec.validateEncoderMetadata(); err == nil {
		t.Fatal("bound encoder policy accepted invalid metadata")
	}
	spec.TokenTypeCount = runtimePolicyFixtureTokenTypes
	spec.HeadCountKV = spec.HeadCount
	if err := spec.validateEncoderMetadata(); err != nil {
		t.Fatalf("bound encoder policy rejected valid metadata: %v", err)
	}
}

func TestAttentionValidationUsesBoundProfilePolicy(t *testing.T) {
	profile := ArchitectureProfile{
		Name:       runtimePolicyFixtureArchitecture,
		Validation: validationPolicy{Attention: attentionValidationFullRotary},
	}
	spec := Spec{
		CommonSpec: CommonSpec{Architecture: runtimePolicyFixtureArchitecture},
		AttentionSpec: AttentionSpec{
			KeyLength: runtimePolicyFixtureHeadWidth, ValueLength: runtimePolicyFixtureNarrowWidth,
			RopeDimensionCount: runtimePolicyFixtureHeadWidth,
		},
	}.withProfile(profile)
	if err := spec.validateAttentionMetadata(); err == nil {
		t.Fatal("bound attention policy accepted invalid metadata")
	}
	spec.ValueLength = spec.KeyLength
	if err := spec.validateAttentionMetadata(); err != nil {
		t.Fatalf("bound attention policy rejected valid metadata: %v", err)
	}
}

func TestMLAValidationUsesBoundProfilePolicy(t *testing.T) {
	profile := ArchitectureProfile{
		Name:       runtimePolicyFixtureArchitecture,
		Validation: validationPolicy{MLA: mlaValidationScaledLatent},
	}
	spec := Spec{
		CommonSpec: CommonSpec{Architecture: runtimePolicyFixtureArchitecture},
	}.withProfile(profile)
	if err := spec.validateMLAMetadata(); err == nil {
		t.Fatal("bound MLA policy accepted invalid metadata")
	}
	spec.QLoRARank = runtimePolicyFixtureRank
	spec.ResidualScale = runtimePolicyFixturePositive
	spec.OriginalContextLength = runtimePolicyFixtureBlocks
	spec.RopeAttentionFactor = runtimePolicyFixturePositive
	if err := spec.validateMLAMetadata(); err != nil {
		t.Fatalf("bound MLA policy rejected valid metadata: %v", err)
	}
}

func TestRecurrentValidationUsesBoundProfilePolicy(t *testing.T) {
	profile := ArchitectureProfile{
		Name:       runtimePolicyFixtureArchitecture,
		Validation: validationPolicy{Recurrent: recurrentValidationTargetLayerBlock},
	}
	spec := Spec{
		CommonSpec: CommonSpec{Architecture: runtimePolicyFixtureArchitecture},
	}.withProfile(profile)
	if err := spec.validateRecurrentMetadata(); err == nil {
		t.Fatal("bound recurrent policy accepted invalid metadata")
	}
	spec.TargetLayers = []int32{runtimePolicyFixtureTargetLayer}
	spec.DFlashBlockSize = runtimePolicyFixtureBlocks
	if err := spec.validateRecurrentMetadata(); err != nil {
		t.Fatalf("bound recurrent policy rejected valid metadata: %v", err)
	}
}

func TestHybridValidationUsesBoundProfilePolicy(t *testing.T) {
	profile := ArchitectureProfile{
		Name:       runtimePolicyFixtureArchitecture,
		Validation: validationPolicy{Hybrid: hybridValidationSharedExperts},
	}
	spec := Spec{
		CommonSpec: CommonSpec{Architecture: runtimePolicyFixtureArchitecture},
	}.withProfile(profile)
	if err := spec.validateHybridMetadata(); err == nil {
		t.Fatal("bound hybrid policy accepted invalid metadata")
	}
	spec.ExpertCount = runtimePolicyFixtureHeads
	spec.ExpertUsedCount = runtimePolicyFixtureRank
	spec.ExpertFeedForward = runtimePolicyFixtureHeadWidth
	spec.ExpertWeightsScale = runtimePolicyFixturePositive
	if err := spec.validateHybridMetadata(); err != nil {
		t.Fatalf("bound hybrid policy rejected valid metadata: %v", err)
	}
}

func TestIndexerCadenceUsesBoundProfilePolicy(t *testing.T) {
	profile := ArchitectureProfile{
		Name:    runtimePolicyFixtureArchitecture,
		Cadence: layerCadencePolicy{FullIndexerEveryLayer: true},
	}
	spec := Spec{CommonSpec: CommonSpec{
		Architecture: runtimePolicyFixtureArchitecture,
		BlockCount:   runtimePolicyFixtureBlocks, NextNPredictLayers: runtimePolicyFixtureRank,
	}}.withProfile(profile)
	lastLayer := spec.BlockCount + spec.NextNPredictLayers - 1
	if !spec.LayerHasFullIndexer(lastLayer) || spec.LayerHasFullIndexer(lastLayer+1) {
		t.Fatal("bound full-indexer cadence is invalid")
	}
}

func TestArchitectureProfileDraftPlan(t *testing.T) {
	for architecture, want := range map[string]DraftPlan{
		"qwen35":     {Kind: draftSingleCatalog, Heads: 1, SingleCatalog: true, SupportsMTPOnly: true, Session: DraftSessionSingle, QueryCopies: 2},
		"step35":     {Kind: draftAppendedMultiCarry, Heads: 2, AppendedBlocks: true, CarryRawHidden: true, Session: DraftSessionMulti},
		"hy_v3":      {Kind: draftAppendedMulti, Heads: 2, AppendedBlocks: true, Session: DraftSessionMulti},
		"glm4":       {Kind: DraftAppendedSingle, Heads: 1, AppendedBlocks: true, Session: DraftSessionSingle},
		"cohere2moe": {Kind: draftOptionalSingleCatalog, Heads: 1, SingleCatalog: true, OptionalCatalog: true, SupportsMTPOnly: true, ScaleLogits: true, Normalization: draftNormalizationArchitecture, Session: DraftSessionSingle},
	} {
		profile, _ := LookupArchitecture(architecture)
		heads := want.Heads
		if got := profile.DraftPlan(heads); got != want || !got.SessionEligible() {
			t.Errorf("%s draft plan = %+v, want %+v", architecture, got, want)
		}
	}
	next, _ := LookupArchitecture("glm4")
	if next.DraftPlan(2).SessionEligible() {
		t.Fatal("multi-head NextN unexpectedly session-eligible")
	}
}

func TestArchitectureProfileDeepSeekLayoutPolicy(t *testing.T) {
	for architecture, wantGraph := range map[string]bool{
		"deepseek2": true, "deepseek32": true, "mistral4": true, "glm-dsa": false,
	} {
		profile, ok := LookupArchitecture(architecture)
		if !ok || !profile.Has(architectureLatentKVLayout) ||
			profile.Has(architectureLatentYaRNQuery) != wantGraph {
			t.Fatalf("%s DeepSeek policies = %064b", architecture, profile.Capabilities)
		}
	}
	for _, architecture := range []string{"deepseek", "deepseek2-ocr", "minicpm3"} {
		profile, _ := LookupArchitecture(architecture)
		if profile.Has(architectureLatentKVLayout) {
			t.Fatalf("%s unexpectedly has DeepSeek2 layout", architecture)
		}
	}
}

func TestArchitectureProfileForwardProgram(t *testing.T) {
	type expectedForward struct {
		operation ForwardOperation
		session   ForwardSession
	}
	for architecture, want := range map[string]expectedForward{
		"llama":            {operation: ForwardOperationCached},
		"bert":             {operation: ForwardOperationBidirectional},
		"dream":            {operation: ForwardOperationDiffusion},
		"llada":            {operation: ForwardOperationDiffusion},
		"dflash":           {operation: ForwardOperationSession, session: ForwardSessionPairedFeatures},
		"eagle3":           {operation: ForwardOperationSession, session: ForwardSessionFeatureDraft},
		"gemma4-assistant": {operation: ForwardOperationSession, session: ForwardSessionPairedProjection},
		"wavtokenizer-dec": {operation: ForwardOperationAudioTokens},
		"t5encoder":        {operation: ForwardOperationEncoder},
		"t5":               {operation: ForwardOperationSession, session: ForwardSessionEncoderDecoder},
	} {
		profile, ok := LookupArchitecture(architecture)
		if !ok || profile.Forward.Operation != want.operation || profile.Forward.Session != want.session || !profile.Forward.valid() {
			t.Fatalf("%s forward program = %v, want %v", architecture, profile.Forward, want)
		}
	}
}

func TestArchitectureProfileOutputNormPolicy(t *testing.T) {
	for architecture, want := range map[string]struct {
		policy OutputNormPolicy
		tensor string
	}{
		"llama":          {outputNormModel, "output_norm.weight"},
		"bert":           {OutputNormAbsent, ""},
		"nomic-bert-moe": {OutputNormAbsent, ""},
		"neo-bert":       {outputNormEncoder, "enc.output_norm.weight"},
		"t5encoder":      {outputNormEncoder, "enc.output_norm.weight"},
		"t5":             {outputNormDecoder, "dec.output_norm.weight"},
		"lfm2":           {outputNormTokenEmbedding, "token_embd_norm.weight"},
	} {
		profile, ok := LookupArchitecture(architecture)
		if !ok || profile.OutputNorm != want.policy || profile.OutputNormTensor() != want.tensor {
			t.Fatalf("%s output norm = %v/%q, want %v/%q",
				architecture, profile.OutputNorm, profile.OutputNormTensor(), want.policy, want.tensor)
		}
		if want.policy == OutputNormAbsent && !profile.Has(architectureOutputLayerNormLayout) {
			t.Fatalf("%s has no BERT normalization layout", architecture)
		}
	}
}

func TestArchitectureProfileFusedQKVPolicy(t *testing.T) {
	for architecture, capabilities := range map[string]ArchitectureCapability{
		"bloom": architectureFusedQKV | architectureRequiresFusedQKV | architectureRequiresFusedQKVBias,
		"mpt":   architectureFusedQKV | architectureRequiresFusedQKV,
		"phi3":  architectureFusedQKV | architectureRejectsOrphanFusedQKVBias,
		"llama": 0,
	} {
		profile, ok := LookupArchitecture(architecture)
		mask := architectureFusedQKV | architectureRequiresFusedQKV |
			architectureRequiresFusedQKVBias | architectureRejectsOrphanFusedQKVBias
		if !ok || profile.Capabilities&mask != capabilities {
			t.Fatalf("%s fused QKV capabilities = %064b", architecture, profile.Capabilities)
		}
	}
}

func TestArchitectureProfileProjectedInputPolicies(t *testing.T) {
	tests := []struct {
		architecture string
		overrides    EmbeddingOverridePolicy
		deepstack    DeepstackPolicy
		blocks       AttentionBlockPolicy
	}{
		{"llama", embeddingOverrideStandard, deepstackNone, AttentionBlocksNone},
		{"cogvlm", EmbeddingOverrideVisualSpan, deepstackNone, AttentionBlocksNone},
		{"gemma4", EmbeddingOverrideRawScaled, deepstackNone, AttentionBlocksUncached},
		{"granite", EmbeddingOverrideMappedBase, deepstackMappedBefore, AttentionBlocksNone},
		{"qwen3vl", embeddingOverrideStandard, deepstackSequentialAfter, AttentionBlocksNone},
	}
	for _, test := range tests {
		profile, ok := LookupArchitecture(test.architecture)
		if !ok || profile.Overrides != test.overrides || profile.Deepstack != test.deepstack ||
			profile.AttentionBlocks != test.blocks {
			t.Fatalf("%s projected-input policies = %#v", test.architecture, profile)
		}
	}
}

func TestArchitectureProfileLayerSideInputPolicies(t *testing.T) {
	for architecture, want := range map[string]AuxiliaryFlow{
		"llama": AuxiliaryNone, "rwkv7": auxiliaryRecurrentValue, "arwkv7": auxiliaryRecurrentValue,
		"glm-dsa": AuxiliarySparseTopK, "deepseek32": AuxiliaryNone,
	} {
		profile, _ := LookupArchitecture(architecture)
		if profile.Auxiliary != want {
			t.Fatalf("%s auxiliary flow = %v, want %v", architecture, profile.Auxiliary, want)
		}
	}
	for architecture, want := range map[string]AttentionTemperaturePolicy{
		"llama": AttentionTemperatureNone, "llama4": attentionTemperatureNoRoPE,
		"deepseek2": AttentionTemperatureConfigured, "mistral3": AttentionTemperatureConfigured,
		"mistral4": AttentionTemperatureConfigured,
	} {
		profile, _ := LookupArchitecture(architecture)
		if profile.Temperature != want {
			t.Fatalf("%s temperature policy = %v, want %v", architecture, profile.Temperature, want)
		}
	}
}

func TestArchitectureProfileNormTensorCatalog(t *testing.T) {
	for architecture, want := range map[string]postNormTensorNames{
		"llama": {
			AttentionWeight: "post_attention_norm.weight", FeedForwardWeight: "post_ffw_norm.weight",
		},
		"bert": {
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			AttentionBias: "attn_output_norm.bias", FeedForwardBias: "layer_output_norm.bias",
		},
		"grok": {
			AttentionWeight: "attn_output_norm.weight", FeedForwardWeight: "layer_output_norm.weight",
			FeedForwardAlternate: "ffn_post_norm.weight",
		},
	} {
		profile, _ := LookupArchitecture(architecture)
		if got := profile.PostNormTensors(); got != want {
			t.Fatalf("%s post-norm tensors = %#v, want %#v", architecture, got, want)
		}
	}
	for architecture, want := range map[string]string{
		"llama": "ffn_norm.weight", "falcon-h1": "ffn_norm", "dbrx": "attn_output_norm.weight",
		"glm4moe": "attn_post_norm.weight", "qwen35": "post_attention_norm.weight",
	} {
		profile, _ := LookupArchitecture(architecture)
		if got := profile.FeedForwardNormTensor(); got != want {
			t.Fatalf("%s FFN norm tensor = %q, want %q", architecture, got, want)
		}
	}
}
