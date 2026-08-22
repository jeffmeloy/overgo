package thoughtbank

import (
	"fmt"
	"sort"

	"overgo/internal/checked"
	"overgo/internal/pytorchzip"
)

// LoadCheckpoint binds a self-describing PyTorch checkpoint to the shared
// ThoughtBank runtime. Architecture and execution settings come from the file.
// Port source: adaptive_new 0ff1e66b9.
func LoadCheckpoint(path string) (*FastWeightBankLMWeights, ArchConfig, error) {
	catalog, err := pytorchzip.ReadCatalog(path)
	if err != nil {
		return nil, ArchConfig{}, fmt.Errorf("thoughtbank load metadata: %w", err)
	}
	metas := catalog.Tensors
	config, err := checkpointConfig(metas, catalog.Scalars)
	if err != nil {
		return nil, ArchConfig{}, err
	}
	names := ExpectedTensorNames(config)
	if err := validateInventory(metas, names); err != nil {
		return nil, ArchConfig{}, err
	}
	bindings, err := pytorchzip.CompileBindings(metas, names)
	if err != nil {
		return nil, ArchConfig{}, fmt.Errorf("thoughtbank compile bindings: %w", err)
	}
	reader, err := pytorchzip.Open(path)
	if err != nil {
		return nil, ArchConfig{}, fmt.Errorf("thoughtbank open checkpoint: %w", err)
	}
	defer reader.Close()
	values, err := reader.ReadBindingValues(bindings)
	if err != nil {
		return nil, ArchConfig{}, fmt.Errorf("thoughtbank read weights: %w", err)
	}
	tensors := make(map[string][]float32, len(bindings))
	for index, binding := range bindings {
		tensors[binding.Identity.Name] = values[index]
	}
	weights, err := bindCheckpoint(config, tensors)
	if err != nil {
		return nil, ArchConfig{}, err
	}
	return weights, config, nil
}

func checkpointConfig(metas []pytorchzip.TensorMeta, scalars map[string]any) (ArchConfig, error) {
	shapes := make(map[string][]int64, len(metas))
	for _, meta := range metas {
		shapes[meta.Name] = meta.Shape
	}
	swiGLU, ok := scalars["mem_read_swiglu"].(bool)
	if !ok {
		return ArchConfig{}, fmt.Errorf("thoughtbank config: mem_read_swiglu is missing or not boolean")
	}
	config, err := DeriveArchFromShapes(shapes, swiGLU)
	if err != nil {
		return ArchConfig{}, err
	}
	fields := []struct {
		name string
		dst  *int
	}{
		{"max_mem", &config.MaxMem},
		{"top_k_csa", &config.TopKCSA},
		{"top_k_experts", &config.TopKExperts},
		{"n_win", &config.NWin},
		{"sinkhorn_iters", &config.SinkhornIters},
		{"mem_seed_slots", &config.MemSeedSlots},
	}
	for _, field := range fields {
		value, err := scalarInt(scalars, field.name)
		if err != nil {
			return ArchConfig{}, err
		}
		*field.dst = value
	}
	return config, nil
}

func scalarInt(values map[string]any, name string) (int, error) {
	value, ok := values[name]
	if !ok {
		return 0, fmt.Errorf("thoughtbank config: missing %q", name)
	}
	switch number := value.(type) {
	case int64:
		return int(number), nil
	case float64:
		converted := int(number)
		if float64(converted) != number {
			return 0, fmt.Errorf("thoughtbank config: %q is not an integer", name)
		}
		return converted, nil
	default:
		return 0, fmt.Errorf("thoughtbank config: %q has type %T", name, value)
	}
}

func validateInventory(metas []pytorchzip.TensorMeta, expected []string) error {
	actual := make([]string, len(metas))
	for index, meta := range metas {
		actual[index] = meta.Name
	}
	sort.Strings(actual)
	if len(actual) != len(expected) {
		return fmt.Errorf("thoughtbank inventory: checkpoint has %d tensors, runtime requires %d", len(actual), len(expected))
	}
	for index := range expected {
		if actual[index] != expected[index] {
			return fmt.Errorf("thoughtbank inventory: tensor %q does not match required %q", actual[index], expected[index])
		}
	}
	return nil
}

func bindCheckpoint(config ArchConfig, tensors map[string][]float32) (*FastWeightBankLMWeights, error) {
	var bindErr error
	get := func(name string) []float32 {
		values, ok := tensors[name]
		if !ok && bindErr == nil {
			bindErr = fmt.Errorf("thoughtbank checkpoint: missing tensor %q", name)
		}
		return values
	}
	weights := &FastWeightBankLMWeights{
		VocabSize: config.Vocab,
		DModel:    config.DModel,
		NLayers:   config.NLayers,
		NHC:       config.NHC,
		MemDim:    config.MemDim,
		MaxMem:    config.MaxMem,
		Embed:     get("embed.weight"),
		AOutNet:   get("A_out_net.weight"),
		NormOut:   get("norm_out.weight"),
		Write: &thoughtWriteWeights{
			WriteCtxQ:     get("thought_stream.write_ctx_q.weight"),
			WriteGate:     get("thought_stream.write_gate.weight"),
			WriteGateBias: get("thought_stream.write_gate.bias"),
			ThoughtHead:   get("thought_stream.thought_head.weight"),
			NormWrite:     get("thought_stream.norm_write.weight"),
			WriteDecision: get("thought_stream.write_decision.weight"),
			WriteDecBias:  get("thought_stream.write_decision.bias"),
		},
		NormEps: config.NormEps,
	}
	if config.Untied {
		weights.LMHead = get("lm_head.weight")
	}
	for layer := 0; layer < config.NLayers; layer++ {
		prefix := fmt.Sprintf("blocks.%d.", layer)
		value := func(name string) []float32 { return get(prefix + name) }
		scalar := func(name string) float32 {
			values := value(name)
			first, ok := checked.First(values)
			if !ok {
				if bindErr == nil {
					bindErr = fmt.Errorf("thoughtbank checkpoint: tensor %q is empty", prefix+name)
				}
				return first
			}
			return first
		}
		sparse := layer%2 == 0
		blockSize := config.HCAm
		if sparse {
			blockSize = config.CSAm
		}
		attention := &CompressedHybridAttnWeights{
			Sparse: sparse,
			DModel: config.DModel, NHeads: config.NHeads, DHead: config.DHead,
			M: blockSize, NWin: config.NWin, DLatentQ: config.DLatentQ,
			NGroups: config.NGroups, TopK: config.TopKCSA, NIdxHeads: config.NIdxHeads,
			WDq: value("_attn.W_dq.weight"), WUq: value("_attn.W_uq.weight"),
			WWk: value("_attn.W_wk.weight"), WWv: value("_attn.W_wv.weight"),
			OutProj: value("_attn.out_proj.weight"), QNorm: value("_attn.q_norm.weight"),
			KVNorm: value("_attn.kv_norm.weight"), SinkLogits: value("_attn.sink_logits"),
			NormEps: config.NormEps,
		}
		for group := 0; group < config.NGroups; group++ {
			attention.OutGroup = append(attention.OutGroup, value(fmt.Sprintf("_attn.out_group.%d.weight", group)))
		}
		if sparse {
			attention.WKVa = value("_attn.W_kv_a.weight")
			attention.WKVb = value("_attn.W_kv_b.weight")
			attention.WZa = value("_attn.W_z_a.weight")
			attention.WZb = value("_attn.W_z_b.weight")
			attention.PosA = value("_attn.pos_a")
			attention.PosB = value("_attn.pos_b")
			attention.WIq = value("_attn.W_iq.weight")
			attention.WW = value("_attn.W_w.weight")
		} else {
			attention.WKV = value("_attn.W_kv.weight")
			attention.WZ = value("_attn.W_z.weight")
			attention.Pos = value("_attn.pos")
		}
		moe := &sharedRoutedMoEWeights{
			DModel: config.DModel, DFF: config.DFF, NExperts: config.NExperts,
			NShared: config.NShared, TopK: config.TopKExperts, WGate: value("_moe.W_gate.weight"),
		}
		for expert := 0; expert < config.NExperts; expert++ {
			moe.ExpertW12 = append(moe.ExpertW12, value(fmt.Sprintf("_moe.experts.%d.w12.weight", expert)))
			moe.ExpertW3 = append(moe.ExpertW3, value(fmt.Sprintf("_moe.experts.%d.w3.weight", expert)))
		}
		for shared := 0; shared < config.NShared; shared++ {
			moe.SharedW12 = append(moe.SharedW12, value(fmt.Sprintf("_moe.shared.%d.w12.weight", shared)))
			moe.SharedW3 = append(moe.SharedW3, value(fmt.Sprintf("_moe.shared.%d.w3.weight", shared)))
		}
		hyper := func(name string) *HyperConnectionWeights {
			return &HyperConnectionWeights{
				NHC: config.NHC, DModel: config.DModel, SinkhornIters: config.SinkhornIters,
				WPre: value(name + ".W_pre.weight"), WRes: value(name + ".W_res.weight"),
				WPost: value(name + ".W_post.weight"), SPre: value(name + ".S_pre"),
				SRes: value(name + ".S_res"), SPost: value(name + ".S_post"),
				AlphaPre: scalar(name + ".alpha_pre"), AlphaRes: scalar(name + ".alpha_res"),
				AlphaPost: scalar(name + ".alpha_post"), NormWeight: value(name + ".norm.weight"),
				NormEps: config.NormEps,
			}
		}
		block := &HyperConnectionBlockWeights{
			NHC: config.NHC, DModel: config.DModel, Attn: attention, MoE: moe,
			Bank: &FastWeightBankWeights{
				DModel: config.DModel, MemDim: config.MemDim, Rank: config.MemReadRank,
				SwiGLU: config.SwiGLU, FWA: value("fw_A.weight"), FWB: value("fw_B.weight"),
				FWO: value("fw_o.weight"), NormWeight: value("norm_fw.weight"), NormEps: config.NormEps,
			},
			MHCAttn: hyper("mhc_attn"), MHCMoE: hyper("mhc_moe"),
			NormAttn: value("norm_attn.weight"), NormMoE: value("norm_moe.weight"),
			ACrossNet: value("A_cross_net.weight"), ReadBank: true, NormEps: config.NormEps,
		}
		if err := block.Bank.Validate(); err != nil && bindErr == nil {
			bindErr = fmt.Errorf("thoughtbank layer %d bank: %w", layer, err)
		}
		if err := block.MHCAttn.Validate(); err != nil && bindErr == nil {
			bindErr = fmt.Errorf("thoughtbank layer %d attention connection: %w", layer, err)
		}
		if err := block.MHCMoE.Validate(); err != nil && bindErr == nil {
			bindErr = fmt.Errorf("thoughtbank layer %d MoE connection: %w", layer, err)
		}
		weights.Blocks = append(weights.Blocks, block)
	}
	if bindErr != nil {
		return nil, bindErr
	}
	return weights, nil
}
