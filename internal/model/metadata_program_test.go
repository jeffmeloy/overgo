package model

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/gguf"
)

func metadataOpSignature(op metadataOp) string {
	modes := map[metadataReadMode]string{
		metadataReadRequired:    "req",
		metadataReadAssignZero:  "zero",
		metadataReadKeepCurrent: "keep",
		metadataReadKeepNonZero: "nonzero",
	}
	switch op.kind {
	case metadataOpUint32:
		return fmt.Sprintf("%s u32 %s", modes[op.mode], op.key)
	case metadataOpFloat32:
		return fmt.Sprintf("%s f32 %s", modes[op.mode], op.key)
	case metadataOpBool:
		return fmt.Sprintf("%s bool %s", modes[op.mode], op.key)
	case metadataOpSetUint32:
		return fmt.Sprintf("set u32=%d", op.value)
	case metadataOpSetBool:
		return fmt.Sprintf("set bool=%t", op.flag)
	case metadataOpDefaults:
		return "defaults"
	case metadataOpInverseKeyScale:
		return "inverse-key-scale"
	case metadataOpEpsilonEither:
		return "epsilon-either"
	case metadataOpRopeSections:
		return fmt.Sprintf("sections %s %s", modes[op.mode], op.key)
	case metadataOpSlidingPatternType:
		return "sliding-pattern-type " + op.key
	case metadataOpHalveFeedForward:
		return "halve-feed-forward"
	case metadataOpExpertFeedForwardFromModel:
		if op.flag {
			return "expert-width-from-model if-zero"
		}
		return "expert-width-from-model"
	case metadataOpSharedFeedForwardFromModel:
		return "shared-width-from-model"
	case metadataOpSharedFeedForwardFromExpert:
		return "shared-width-from-expert"
	case metadataOpSharedFeedForwardScale:
		return "shared-width-scale"
	case metadataOpSharedFeedForwardPolicy:
		return "shared-width-policy"
	case metadataOpKeyedDeltaDimensions:
		return "keyed-delta-dimensions"
	case metadataOpSlidingSchedule:
		return "sliding-schedule " + map[slidingMetadataPolicy]string{
			slidingTargetLayer:          "target-layer",
			slidingRequiredMixed:        "required-mixed",
			slidingRequiredCompatible:   "required-compatible",
			slidingSharedKV:             "shared-kv",
			slidingDefaultMixed:         "default-mixed",
			slidingOptionalMixed:        "optional-mixed",
			slidingDualExpert:           "dual-expert",
			slidingRuntimeOptional:      "runtime-optional",
			slidingRuntimeDual:          "runtime-dual",
			slidingRuntimeRotary:        "runtime-rotary",
			slidingRuntimeRotaryExperts: "runtime-rotary-experts",
		}[op.slide]
	case metadataOpAttentionScaleFromValueWidth:
		return "attention-scale-from-value-width"
	case metadataOpSetFloat32:
		return fmt.Sprintf("set f32=%g", op.floatValue)
	case metadataOpCopyUint32:
		return "copy u32"
	case metadataOpLongRoPE:
		return "long-rope"
	case metadataOpYaRNAttentionFactor:
		return "yarn-attention-factor"
	case metadataOpVocabulary:
		return "vocabulary"
	}
	return "unknown"
}

func programSignatures(program []metadataOp) []string {
	signatures := make([]string, 0, len(program))
	for _, op := range program {
		signatures = append(signatures, metadataOpSignature(op))
	}
	return signatures
}

const (
	rmsEpsilonOp   = "req f32 attention.layer_norm_rms_epsilon"
	layerEpsilonOp = "req f32 attention.layer_norm_epsilon"
)

var rwkv6ReadOps = []string{
	"req u32 time_mix_extra_dim",
	"req u32 time_decay_extra_dim",
	"zero u32 rescale_every_n_layers",
}

var rwkv7ReadOps = []string{
	"req u32 attention.decay_lora_rank",
	"req u32 attention.iclr_lora_rank",
	"req u32 attention.value_residual_mix_lora_rank",
	"zero u32 attention.gate_lora_rank",
}

func rwkvSuffix(reads []string, shift int) []string {
	suffix := append([]string{"req u32 wkv.head_size"}, reads...)
	return append(suffix,
		fmt.Sprintf("set u32=%d", shift),
		"keep u32 token_shift_count",
	)
}

func ssmSuffix(grouped bool) []string {
	suffix := []string{
		"req u32 ssm.conv_kernel",
		"req u32 ssm.inner_size",
		"req u32 ssm.state_size",
		"req u32 ssm.time_step_rank",
	}
	if grouped {
		return append(suffix, "req u32 ssm.group_count")
	}
	return append(suffix, "set u32=1", "zero bool ssm.dt_b_c_rms")
}

func commonCoreOps(epsilon string, derives ...string) []string {
	ops := []string{
		epsilon,
		"zero f32 final_logit_softcapping",
		"zero f32 attn_logit_softcapping",
	}
	ops = append(ops, derives...)
	return append(ops, "keep f32 attention.scale", "defaults")
}

// TestProfileCompiledMetadataLoading proves the profile-compiled metadata
// programs read, derive, and validate the same spec fields the deleted
// per-family branches did, and that unmigrated families keep branch loading.
func TestProfileCompiledMetadataLoading(t *testing.T) {
	cohere2Suffix := []string{
		"req f32 logit_scale",
		"req u32 rope.dimension_count",
		"sliding-pattern-type attention.sliding_window_pattern",
	}
	ropeDimensionRequired := []string{"req u32 rope.dimension_count"}
	phi3Suffix := []string{
		"req u32 rope.dimension_count",
		"req u32 rope.scaling.original_context_length",
	}
	visualSections := []string{"sections req rope.dimension_sections"}
	programs := []struct {
		architecture string
		core         []string
		expert       []string
	}{
		{"llama", commonCoreOps(rmsEpsilonOp), nil},
		{"qwen2", commonCoreOps(rmsEpsilonOp), nil},
		{"granite", commonCoreOps(rmsEpsilonOp), nil},
		{"talkie", append(commonCoreOps(rmsEpsilonOp), "req f32 logit_scale"), nil},
		{"cohere2", append(commonCoreOps(layerEpsilonOp), cohere2Suffix...), nil},
		{"cohere2moe", append(commonCoreOps("epsilon-either"), cohere2Suffix...), nil},
		{"stablelm", append(commonCoreOps(layerEpsilonOp), ropeDimensionRequired...), nil},
		{"phi2", append(commonCoreOps(layerEpsilonOp), ropeDimensionRequired...), nil},
		{"phi3", append(commonCoreOps(rmsEpsilonOp), phi3Suffix...), nil},
		{"phimoe", append(commonCoreOps(rmsEpsilonOp), phi3Suffix...), nil},
		{"gemma-embedding", append(commonCoreOps(rmsEpsilonOp),
			"req u32 attention.sliding_window",
			"zero u32 dense_2_feat_in",
			"zero u32 dense_2_feat_out",
			"zero u32 dense_3_feat_in",
			"zero u32 dense_3_feat_out"), nil},
		{"gptneox", append(commonCoreOps(layerEpsilonOp),
			"zero u32 rope.dimension_count",
			"req bool use_parallel_residual"), nil},
		{"qwen", append(commonCoreOps(rmsEpsilonOp), "halve-feed-forward"), nil},
		{"glm4", append(commonCoreOps(rmsEpsilonOp), "sections keep rope.dimension_sections"), nil},
		{"glm4moe", append(commonCoreOps(rmsEpsilonOp), "sections keep rope.dimension_sections"), nil},
		{"falcon", append(commonCoreOps(layerEpsilonOp), "zero u32 rope.dimension_count"), nil},
		{"gptj", append(commonCoreOps(layerEpsilonOp), ropeDimensionRequired...), nil},
		{"command-r", append(commonCoreOps(layerEpsilonOp), "zero f32 logit_scale"), nil},
		{"jais", commonCoreOps(layerEpsilonOp, "inverse-key-scale"), nil},
		{"smollm3", append(commonCoreOps(rmsEpsilonOp), "set u32=4"), nil},
		{"olmo", append(commonCoreOps(layerEpsilonOp), "keep f32 attention.clamp_kqv"), nil},
		{"qwen2vl", append(commonCoreOps(rmsEpsilonOp), visualSections...), nil},
		{"paddleocr", append(commonCoreOps(rmsEpsilonOp), visualSections...), nil},
		{"qwen3vl", append(commonCoreOps(rmsEpsilonOp),
			"sections req rope.dimension_sections",
			"keep u32 n_deepstack_layers"), nil},
		{"qwen3vlmoe", append(commonCoreOps(rmsEpsilonOp),
			"sections req rope.dimension_sections",
			"keep u32 n_deepstack_layers"), nil},
		{"afmoe", append(commonCoreOps(rmsEpsilonOp), "set u32=4"), nil},
		{"mimo2", append(commonCoreOps(rmsEpsilonOp), "sliding-schedule required-mixed"), []string{
			"set u32=2", "set bool=true"}},
		{"mellum", append(commonCoreOps(rmsEpsilonOp), "sliding-schedule optional-mixed"), []string{
			"req u32 expert_feed_forward_length", "set bool=true"}},
		{"hunyuan-moe", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"shared-width-policy", "set bool=true"}},
		{"dbrx", commonCoreOps(layerEpsilonOp), []string{
			"expert-width-from-model", "set bool=true"}},
		{"smallthinker", append(commonCoreOps(rmsEpsilonOp), "sliding-schedule dual-expert"), []string{
			"expert-width-from-model", "set bool=true",
			"req u32 expert_gating_func"}},
		{"dots1", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"req u32 expert_shared_count",
			"req u32 expert_gating_func",
			"shared-width-from-expert", "shared-width-scale",
			"zero bool expert_weights_norm",
			"zero u32 leading_dense_block_count"}},
		{"bailingmoe", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"req u32 expert_shared_count",
			"shared-width-from-expert", "shared-width-scale",
			"zero bool expert_weights_norm",
			"zero u32 leading_dense_block_count"}},
		{"deepseek", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"req u32 expert_shared_count",
			"shared-width-from-expert", "shared-width-scale",
			"zero u32 leading_dense_block_count"}},
		{"lfm2moe", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"req u32 expert_gating_func",
			"zero u32 leading_dense_block_count"}},
		{"bailingmoe2", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"req u32 expert_shared_count",
			"shared-width-from-expert",
			"keep u32 expert_shared_feed_forward_length",
			"shared-width-scale",
			"req u32 expert_gating_func",
			"zero bool expert_weights_norm",
			"zero u32 leading_dense_block_count"}},
		{"qwen2moe", commonCoreOps(rmsEpsilonOp), []string{
			"expert-width-from-model if-zero",
			"set u32=1",
			"shared-width-from-model",
			"keep u32 expert_shared_feed_forward_length"}},
		{"ernie4_5-moe", commonCoreOps(rmsEpsilonOp), []string{
			"req u32 expert_feed_forward_length",
			"req u32 interleave_moe_layer_step",
			"zero u32 leading_dense_block_count",
			"zero u32 expert_shared_feed_forward_length",
			"set bool=true"}},
		{"nomic-bert-moe", commonCoreOps(layerEpsilonOp), []string{
			"req u32 moe_every_n_layers"}},
		{"rwkv6", append(commonCoreOps(layerEpsilonOp), rwkvSuffix(rwkv6ReadOps, 2)...), nil},
		{"rwkv6qwen2", append(commonCoreOps(rmsEpsilonOp), rwkvSuffix(rwkv6ReadOps, 1)...), nil},
		{"rwkv7", append(commonCoreOps(layerEpsilonOp), rwkvSuffix(rwkv7ReadOps, 2)...), nil},
		{"arwkv7", append(commonCoreOps(rmsEpsilonOp), rwkvSuffix(rwkv7ReadOps, 1)...), nil},
		{"mamba", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(false)...), nil},
		{"jamba", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(false)...), nil},
		{"mamba2", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(true)...), nil},
		{"granitehybrid", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(true)...), nil},
		{"plamo2", append(append(commonCoreOps(rmsEpsilonOp), ssmSuffix(true)...), "attention-scale-from-value-width"), nil},
		{"nemotron_h", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(true)...), nil},
		{"nemotron_h_moe", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(true)...), nil},
		{"falcon-h1", append(commonCoreOps(rmsEpsilonOp), ssmSuffix(true)...), nil},
		{"lfm2", commonCoreOps(rmsEpsilonOp), nil},
		{"gpt-oss", commonCoreOps(rmsEpsilonOp), nil},
	}
	runtimePrograms := map[string][]string{
		"lfm2": {
			"req u32 shortconv.l_cache",
			"keep u32 attention.sliding_window",
		},
		"lfm2moe": {
			"req u32 shortconv.l_cache",
			"keep u32 attention.sliding_window",
		},
		"gpt-oss": {"req u32 attention.sliding_window"},
	}
	for _, entry := range programs {
		profile, supported := LookupArchitecture(entry.architecture)
		if !supported {
			t.Fatalf("architecture %q is unsupported", entry.architecture)
		}
		core := programSignatures(compileArchitectureCoreProgram(profile))
		if !slices.Equal(core, entry.core) {
			t.Fatalf("%s core program mismatch:\n got %v\nwant %v", entry.architecture, core, entry.core)
		}
		expert := programSignatures(compileExpertProgram(profile))
		if !slices.Equal(expert, entry.expert) {
			t.Fatalf("%s expert program mismatch:\n got %v\nwant %v", entry.architecture, expert, entry.expert)
		}
		runtime := programSignatures(compileRuntimeProgram(profile))
		if len(runtime) == 0 || runtime[len(runtime)-1] != "vocabulary" || slices.Contains(runtime, "unknown") {
			t.Fatalf("%s runtime program is incomplete: %v", entry.architecture, runtime)
		}
		for _, required := range runtimePrograms[entry.architecture] {
			if !slices.Contains(runtime, required) {
				t.Fatalf("%s runtime program lacks %q: %v", entry.architecture, required, runtime)
			}
		}
	}

	llamaFile := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama"),
		metadata("llama.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("llama.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("llama.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("llama.attention.scale", gguf.ValueTypeFloat32, float32(0.5)),
	}}
	spec, err := ReadSpec(llamaFile)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RMSNormEpsilon != 1e-5 || spec.AttentionScale != 0.5 ||
		spec.FinalLogitSoftcap != 0 || spec.AttentionSoftcap != 0 {
		t.Fatalf("unexpected Llama compiled-path spec: %+v", spec)
	}
	qwen2File := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen2"),
		metadata("qwen2.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen2.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("qwen2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen2.rope.freq_base", gguf.ValueTypeFloat32, float32(1000000)),
		metadata("qwen2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen2.final_logit_softcapping", gguf.ValueTypeFloat32, float32(30)),
	}}
	if spec, err = ReadSpec(qwen2File); err != nil {
		t.Fatal(err)
	}
	if spec.RMSNormEpsilon != 1e-6 || spec.FinalLogitSoftcap != 30 {
		t.Fatalf("unexpected Qwen2 compiled-path spec: %+v", spec)
	}
	qwenFile := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen"),
		metadata("qwen.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("qwen.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("qwen.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
	}}
	if spec, err = ReadSpec(qwenFile); err != nil {
		t.Fatal(err)
	}
	if spec.FeedForwardLength != 12 {
		t.Fatalf("Qwen compiled halving produced %d", spec.FeedForwardLength)
	}
	jaisFile := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "jais"),
		metadata("jais.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jais.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("jais.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("jais.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("jais.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jais.attention.max_alibi_bias", gguf.ValueTypeFloat32, float32(8)),
		metadata("jais.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	if spec, err = ReadSpec(jaisFile); err != nil {
		t.Fatal(err)
	}
	if spec.AttentionScale != 0.25 || spec.LayerNormEpsilon != 1e-5 {
		t.Fatalf("unexpected Jais compiled-path spec: %+v", spec)
	}
	graniteFile := &gguf.File{Metadata: append(graniteMetadata(),
		metadata("granite.embedding_scale", gguf.ValueTypeFloat32, float32(2)),
	)}
	if spec, err = ReadSpec(graniteFile); err != nil {
		t.Fatal(err)
	}
	if spec.LogitScale != 8 || spec.EmbeddingScale != 2 {
		t.Fatalf("unexpected Granite branch-path spec: %+v", spec)
	}

	cohere2moeProfile, _ := LookupArchitecture("cohere2moe")
	epsilonCases := []struct {
		values    []gguf.Metadata
		wantRMS   float32
		wantLayer float32
		wantError bool
	}{
		{[]gguf.Metadata{metadata("cohere2moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6))}, 1e-6, 0, false},
		{[]gguf.Metadata{metadata("cohere2moe.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5))}, 0, 1e-5, false},
		{nil, 0, 0, true},
	}
	for index, entry := range epsilonCases {
		values := make(map[string]gguf.Value, len(entry.values))
		for _, item := range entry.values {
			values[item.Key] = item.Value
		}
		reader := specMetadata{values: values, architecture: "cohere2moe", prefix: "cohere2moe.", profile: cohere2moeProfile}
		bound := cohere2moeProfile
		probe := Spec{CommonSpec: CommonSpec{Architecture: "cohere2moe"}}
		probe.profile = &bound
		result, programErr := reader.runMetadataProgram(probe, specReadState{}, []metadataOp{{kind: metadataOpEpsilonEither}})
		if entry.wantError {
			if programErr == nil {
				t.Fatalf("epsilon case %d: missing epsilon accepted", index)
			}
			continue
		}
		if programErr != nil {
			t.Fatalf("epsilon case %d: %v", index, programErr)
		}
		if result.RMSNormEpsilon != entry.wantRMS || result.LayerNormEpsilon != entry.wantLayer {
			t.Fatalf("epsilon case %d: got rms=%g layer=%g", index, result.RMSNormEpsilon, result.LayerNormEpsilon)
		}
		if entry.wantRMS > 0 && bound.Normalization != NormalizationRMS {
			t.Fatalf("epsilon case %d: profile normalization was not rebound", index)
		}
	}

	dots1Profile, _ := LookupArchitecture("dots1")
	dots1Values := map[string]gguf.Value{
		"dots1.expert_feed_forward_length": {Type: gguf.ValueTypeUint32, Data: uint32(64)},
		"dots1.expert_shared_count":        {Type: gguf.ValueTypeUint32, Data: uint32(2)},
		"dots1.expert_gating_func":         {Type: gguf.ValueTypeUint32, Data: expertGatingSigmoid},
	}
	dots1Reader := specMetadata{values: dots1Values, architecture: "dots1", prefix: "dots1.", profile: dots1Profile}
	dots1Spec, dots1Err := dots1Reader.runMetadataProgram(Spec{}, specReadState{}, compileExpertProgram(dots1Profile))
	if dots1Err != nil {
		t.Fatal(dots1Err)
	}
	if dots1Spec.ExpertFeedForward != 64 || dots1Spec.SharedExpertCount != 2 ||
		dots1Spec.SharedExpertFF != 128 || dots1Spec.ExpertGatingFunc != expertGatingSigmoid ||
		dots1Spec.ExpertWeightsNorm || dots1Spec.LeadingDenseBlocks != 0 {
		t.Fatalf("unexpected DOTS1 compiled expert spec: %+v", dots1Spec)
	}

	qwen2moeProfile, _ := LookupArchitecture("qwen2moe")
	qwen2moeReader := specMetadata{values: map[string]gguf.Value{}, architecture: "qwen2moe", prefix: "qwen2moe.", profile: qwen2moeProfile}
	qwen2moeSpec, qwen2moeErr := qwen2moeReader.runMetadataProgram(
		Spec{CommonSpec: CommonSpec{FeedForwardLength: 48}}, specReadState{}, compileExpertProgram(qwen2moeProfile),
	)
	if qwen2moeErr != nil {
		t.Fatal(qwen2moeErr)
	}
	if qwen2moeSpec.ExpertFeedForward != 48 || qwen2moeSpec.SharedExpertCount != 1 ||
		qwen2moeSpec.SharedExpertFF != 48 {
		t.Fatalf("unexpected Qwen2-MoE compiled expert spec: %+v", qwen2moeSpec)
	}
}
