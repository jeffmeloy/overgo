package sampling

// Policy defines recipe-owned sampler defaults. Request-only grammar, bias,
// infill, and tokenized DRY state remain explicit inputs.
type Policy struct {
	Temperature       float32        `json:"temperature"`
	DynatempRange     float32        `json:"dynatemp_range"`
	DynatempExponent  float32        `json:"dynatemp_exponent"`
	TopK              int            `json:"top_k"`
	TopP              float32        `json:"top_p"`
	MinP              float32        `json:"min_p"`
	TypicalP          float32        `json:"typical_p"`
	TopNSigma         float32        `json:"top_n_sigma"`
	XTCProbability    float32        `json:"xtc_probability"`
	XTCThreshold      float32        `json:"xtc_threshold"`
	MinKeep           int            `json:"min_keep"`
	AdaptiveTarget    float32        `json:"adaptive_target"`
	AdaptiveDecay     float32        `json:"adaptive_decay"`
	RepeatLastN       int            `json:"repeat_last_n"`
	RepeatPenalty     float32        `json:"repeat_penalty"`
	PresencePenalty   float32        `json:"presence_penalty"`
	FrequencyPenalty  float32        `json:"frequency_penalty"`
	NoRepeatNgramSize int            `json:"no_repeat_ngram_size"`
	NgramWindow       int            `json:"ngram_window"`
	DryMultiplier     float32        `json:"dry_multiplier"`
	DryBase           float32        `json:"dry_base"`
	DryAllowedLength  int            `json:"dry_allowed_length"`
	DryPenaltyLastN   int            `json:"dry_penalty_last_n"`
	DryBreakers       []string       `json:"dry_breakers"`
	Mirostat          int            `json:"mirostat"`
	MirostatTau       float32        `json:"mirostat_tau"`
	MirostatEta       float32        `json:"mirostat_eta"`
	Seed              int64          `json:"seed"`
	Samplers          []SamplerStage `json:"samplers"`
}

// Config compiles policy into one request sampler configuration.
func (p Policy) Config() Config {
	return Config{
		Temperature: p.Temperature, DynatempRange: p.DynatempRange, DynatempExponent: p.DynatempExponent,
		TopK: p.TopK, TopP: p.TopP, MinP: p.MinP, TypicalP: p.TypicalP, TopNSigma: p.TopNSigma,
		XTCProbability: p.XTCProbability, XTCThreshold: p.XTCThreshold, MinKeep: p.MinKeep,
		AdaptiveTarget: p.AdaptiveTarget, AdaptiveDecay: p.AdaptiveDecay,
		RepeatLastN: p.RepeatLastN, RepeatPenalty: p.RepeatPenalty,
		PresencePenalty: p.PresencePenalty, FrequencyPenalty: p.FrequencyPenalty,
		NoRepeatNgramSize: p.NoRepeatNgramSize, NgramWindow: p.NgramWindow,
		DryMultiplier: p.DryMultiplier, DryBase: p.DryBase,
		DryAllowedLength: p.DryAllowedLength, DryPenaltyLastN: p.DryPenaltyLastN,
		Mirostat: p.Mirostat, MirostatTau: p.MirostatTau, MirostatEta: p.MirostatEta,
		Seed: p.Seed, Samplers: cloneSamplerOrder(p.Samplers),
	}
}

// Validate checks policy through the sampler contract.
func (p Policy) Validate() error {
	_, err := New(p.Config())
	return err
}
