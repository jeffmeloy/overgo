package model

// Spec: architecture metadata grouped by runtime concern.
type Spec struct {
	CommonSpec
	AttentionSpec
	MoESpec
	RecurrentSpec
	EncoderSpec
	MultimodalSpec
}

// CommonSpec: shared model metadata.
type CommonSpec struct {
	Architecture          string
	Name                  string
	BlockCount            uint32
	ContextLength         uint32
	EmbeddingLength       uint32
	FeedForwardLength     uint32
	EmbeddingScale        float32
	ResidualScale         float32
	LogitScale            float32
	FinalLogitSoftcap     float32
	RMSNormEpsilon        float32
	LayerNormEpsilon      float32
	VocabularySize        uint32
	OutputEmbeddingLength uint32
	NextNPredictLayers    uint32
	TargetLayers          []int32
	TargetHiddenSize      uint32
	NormBeforeResidual    bool
	HiddenActivation      string
	Dense2FeatureIn       uint32
	Dense2FeatureOut      uint32
	Dense3FeatureIn       uint32
	Dense3FeatureOut      uint32
	ParallelResidual      bool
	SandwichNorm          bool
	HyperConnectionCount  uint32
	HyperSinkhornIters    uint32
	HyperConnectionEps    float32
	HashLayerCount        uint32
}

// AttentionSpec: attention and position metadata.
type AttentionSpec struct {
	HeadCount             uint32
	HeadCountKV           uint32
	KeyLength             uint32
	ValueLength           uint32
	KeyLengthSWA          uint32
	ValueLengthSWA        uint32
	RopeFrequencyBase     float32
	RopeFrequencySWA      float32
	RopeScalingType       string
	RopeScalingFactor     float32
	RopeAttentionFactor   float32
	RopeYaRNLogMultiplier float32
	OriginalContextLength uint32
	AttentionScale        float32
	AttentionTempScale    float32
	AttentionTempFloor    uint32
	AttentionTempOffset   float32
	AttentionValueScale   float32
	AttentionClamp        float32
	AttentionSoftcap      float32
	MaxALiBiBias          float32
	QLoRARank             uint32
	KVLoRARank            uint32
	QKNormEpsilon         float32
	SlidingWindow         uint32
	SlidingPattern        uint32
	NoRopeLayerStep       uint32
	RopeDisabled          bool
	NonCausalAttention    bool
	YaRNExtFactor         float32
	YaRNAttentionFactor   float32
	YaRNBetaFast          float32
	YaRNBetaSlow          float32
	RopeDimensionSWA      uint32
	LayerHeadCounts       []uint32
	LayerKVHeadCounts     []uint32
	SlidingLayers         []bool
	RopeDimensionCount    uint32
	RopeSections          [4]int32
	IndexerHeadCount      uint32
	IndexerKeyLength      uint32
	IndexerTopK           uint32
	IndexerFullLayers     []bool
	AttentionOutputGroups uint32
	AttentionOutputRank   uint32
	CompressRopeBase      float32
	CompressRatios        []uint32
}

// MoESpec: routed-expert metadata.
type MoESpec struct {
	ExpertCount            uint32
	ExpertUsedCount        uint32
	ExpertFeedForward      uint32
	MoELatentSize          uint32
	ExpertChunkFeedForward uint32
	ExpertWeightsScale     float32
	ExpertGroupScale       float32
	ExpertsPerGroup        uint32
	LeadingDenseBlocks     uint32
	MoELayerStep           uint32
	SharedExpertFF         uint32
	SharedExpertCount      uint32
	ExpertGatingFunc       uint32
	ExpertWeightsNorm      bool
	LayerFeedForward       []uint32
	LayerSwiGLUClamp       []float32
	LayerSharedSwiGLUClamp []float32
	XIELUAlphaN            []float32
	XIELUAlphaP            []float32
	XIELUBeta              []float32
	XIELUEpsilon           []float32
}

// RecurrentSpec: state-space and recurrent metadata.
type RecurrentSpec struct {
	ShortConvCacheLength  uint32
	SSMConvKernel         uint32
	SSMInnerSize          uint32
	SSMStateSize          uint32
	SSMTimeStepRank       uint32
	SSMGroupCount         uint32
	KDAHeadDim            uint32
	SSMDtBCNorm           bool
	FullAttentionInterval uint32
	RecurrentLayers       []bool
	WKVHeadSize           uint32
	TimeMixExtraDim       uint32
	TimeDecayExtraDim     uint32
	RescaleEvery          uint32
	TokenShiftCount       uint32
	DecayLoRARank         uint32
	ICLRLoRARank          uint32
	ValueMixLoRARank      uint32
	GateLoRARank          uint32
}

// EncoderSpec: encoder, decoder, and pooling metadata.
type EncoderSpec struct {
	TokenTypeCount      uint32
	RelativeBuckets     uint32
	DecoderBlockCount   uint32
	DecoderStartTokenID uint32
	PoolingType         uint32
	ClassifierLabels    []string
	DFlashBlockSize     uint32
}

// MultimodalSpec: projector-fusion metadata.
type MultimodalSpec struct {
	PosNetEmbeddingLength   uint32
	PosNetBlockCount        uint32
	ConvNextEmbeddingLength uint32
	ConvNextBlockCount      uint32
	GroupNormGroups         uint32
	GroupNormEpsilon        float32
	DeepstackLayerCount     uint32
	DeepstackMapping        []int32
	EmbeddingPerLayer       uint32
	SharedKVLayers          uint32
	KVFromStart             uint32
	AltUpCount              uint32
	AltUpActive             uint32
	LaurelRank              uint32
	SparseLayerCount        uint32
	SparsityStdMultiplier   float32
}
