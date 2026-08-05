package model

type architectureRegistryEditor map[string]ArchitectureProfile

func (e architectureRegistryEditor) update(names []string, apply func(*ArchitectureProfile)) {
	for _, name := range names {
		profile, ok := e[name]
		if !ok {
			panic("unknown architecture profile: " + name)
		}
		apply(&profile)
		e[name] = profile
	}
}

func (e architectureRegistryEditor) setCapabilities(value ArchitectureCapability, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Capabilities |= value })
}

func (e architectureRegistryEditor) setFamily(value ArchitectureFamily, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) {
		profile.Family, profile.GraphFamily, profile.CatalogFamily = value, value, value
	})
}

func (e architectureRegistryEditor) setDraftKind(value DraftKind, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.DraftKind = value })
}

func (e architectureRegistryEditor) setForward(value ForwardPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Forward = value })
}

func (e architectureRegistryEditor) setOutputNorm(value OutputNormPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.OutputNorm = value })
}

func (e architectureRegistryEditor) setNormalization(value NormalizationPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Normalization = value })
}

func (e architectureRegistryEditor) setFeedForward(value FeedForwardPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.FeedForward = value })
}

func (e architectureRegistryEditor) setOverrides(value EmbeddingOverridePolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Overrides = value })
}

func (e architectureRegistryEditor) setDeepstack(value DeepstackPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Deepstack = value })
}

func (e architectureRegistryEditor) setAttentionBlocks(value AttentionBlockPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.AttentionBlocks = value })
}

func (e architectureRegistryEditor) setAuxiliary(value AuxiliaryFlow, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Auxiliary = value })
}

func (e architectureRegistryEditor) setTemperature(value AttentionTemperaturePolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Temperature = value })
}

func (e architectureRegistryEditor) setPostNormLayout(value PostNormLayoutPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.PostNormLayout = value })
}

func (e architectureRegistryEditor) setFFNNormLayout(value FeedForwardNormLayoutPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.FFNNormLayout = value })
}

func (e architectureRegistryEditor) setBlock(value BlockPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Block = value })
}

func (e architectureRegistryEditor) setRecurrentBlock(value BlockPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.RecurrentBlock = value })
}

func (e architectureRegistryEditor) setCache(value CachePolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Cache = value })
}

func (e architectureRegistryEditor) setRecurrentCache(value CachePolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.RecurrentCache = value })
}

func (e architectureRegistryEditor) setCacheFallback(value CacheFallbackPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.CacheFallback = value })
}

func (e architectureRegistryEditor) setDenseGraph(value DenseGraphPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.DenseGraph = value })
}

func (e architectureRegistryEditor) setExperts(value ExpertPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Experts = value })
}

func (e architectureRegistryEditor) updateExperts(names []string, apply func(*ExpertPolicy)) {
	e.update(names, func(profile *ArchitectureProfile) { apply(&profile.Experts) })
}

func (e architectureRegistryEditor) setCadence(value LayerCadencePolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Cadence = value })
}

func (e architectureRegistryEditor) setExpertCatalog(value expertCatalogPolicy, names ...string) {
	e.updateExperts(names, func(experts *ExpertPolicy) { experts.Catalog = value })
}

func (e architectureRegistryEditor) setMetadataShape(value MetadataShapePolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.Metadata = value })
}

func (e architectureRegistryEditor) setEncoderGraph(value encoderGraphKind, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.EncoderGraph.Kind = value })
}

func (e architectureRegistryEditor) setMLAVariant(value mlaVariantPolicy, names ...string) {
	e.update(names, func(profile *ArchitectureProfile) { profile.MLAVariant = value })
}

func (e architectureRegistryEditor) updateDenseStages(names []string, apply func(*DenseStagePolicy)) {
	e.update(names, func(profile *ArchitectureProfile) { apply(&profile.DenseStages) })
}

func (e architectureRegistryEditor) updateDenseWeights(names []string, apply func(*DenseWeightPolicy)) {
	e.update(names, func(profile *ArchitectureProfile) { apply(&profile.DenseWeights) })
}

func (e architectureRegistryEditor) updateRotary(names []string, apply func(*RotaryPolicy)) {
	e.update(names, func(profile *ArchitectureProfile) { apply(&profile.Rotary) })
}

func (e architectureRegistryEditor) updateAttentionGraph(names []string, apply func(*AttentionGraphPolicy)) {
	e.update(names, func(profile *ArchitectureProfile) { apply(&profile.AttentionGraph) })
}
