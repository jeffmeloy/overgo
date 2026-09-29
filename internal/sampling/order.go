package sampling

import (
	"fmt"
	"slices"
	"strings"
)

// SamplerStage names repeatable transform in non-Mirostat sampler
// chain; spellings follow llama.cpp's public CLI and server API
type SamplerStage string

const (
	SamplerPenalties   SamplerStage = "penalties"
	SamplerDry         SamplerStage = "dry"
	SamplerTopNSigma   SamplerStage = "top_n_sigma"
	SamplerTopK        SamplerStage = "top_k"
	SamplerTypicalP    SamplerStage = "typ_p"
	SamplerTopP        SamplerStage = "top_p"
	SamplerMinP        SamplerStage = "min_p"
	SamplerXTC         SamplerStage = "xtc"
	SamplerTemperature SamplerStage = "temperature"
	SamplerAdaptiveP   SamplerStage = "adaptive_p"
	SamplerInfill      SamplerStage = "infill"
)

var defaultSamplerOrder = []SamplerStage{
	SamplerPenalties,
	SamplerDry,
	SamplerTopNSigma,
	SamplerTopK,
	SamplerTypicalP,
	SamplerTopP,
	SamplerMinP,
	SamplerXTC,
	SamplerTemperature,
}

// DefaultSamplerOrder: returns supported subset of llama.cpp's default
// chain; Unsupported disabled-by-default stages are omitted
func DefaultSamplerOrder() []SamplerStage {
	return slices.Clone(defaultSamplerOrder)
}

// configuredOrder is the default order narrowed to the stages a config
// sets, temperature always last. A config that names no order samples
// through only what it configures: an unset (zero) top-p or typical-p never
// truncates, and Temperature zero stays greedy when a grammar or bias takes
// sampling off the raw argmax path, as IsRawGreedy reads an empty order.
func configuredOrder(config Config) []SamplerStage {
	return slices.DeleteFunc(DefaultSamplerOrder(), func(stage SamplerStage) bool {
		switch stage {
		case SamplerPenalties:
			return (config.RepeatPenalty == 0 || config.RepeatPenalty == 1) &&
				config.PresencePenalty == 0 && config.FrequencyPenalty == 0
		case SamplerDry:
			return config.DryMultiplier == 0
		case SamplerTopNSigma:
			return config.TopNSigma <= 0
		case SamplerTopK:
			return config.TopK <= 0
		case SamplerTypicalP:
			return config.TypicalP <= 0 || config.TypicalP >= 1
		case SamplerTopP:
			return config.TopP <= 0 || config.TopP >= 1
		case SamplerMinP:
			return config.MinP <= 0
		case SamplerXTC:
			return config.XTCProbability <= 0
		}
		return false
	})
}

// FormatSamplerOrder returns the semicolon-delimited public stage names.
func FormatSamplerOrder(order []SamplerStage) string {
	names := make([]string, len(order))
	for index, stage := range order {
		names[index] = string(stage)
	}
	return strings.Join(names, ";")
}

// ParseSamplerOrder: parses llama.cpp's semicolon-delimited sampler names
// empty string denotes explicitly empty chain
func ParseSamplerOrder(value string) ([]SamplerStage, error) {
	if value == "" || value == "none" {
		return []SamplerStage{}, nil
	}
	return ParseSamplerNames(strings.Split(value, ";"))
}

// ParseSamplerNames: validates server API's ordered sampler-name array
func ParseSamplerNames(names []string) ([]SamplerStage, error) {
	result := make([]SamplerStage, 0, len(names))
	for index, part := range names {
		stage := SamplerStage(strings.TrimSpace(part))
		if err := validateSamplerStage(stage); err != nil {
			return nil, fmt.Errorf("sampler stage %d: %w", index, err)
		}
		result = append(result, stage)
	}
	return result, nil
}

func validateSamplerStage(stage SamplerStage) error {
	switch stage {
	case SamplerPenalties, SamplerDry, SamplerTopNSigma, SamplerTopK,
		SamplerTypicalP, SamplerTopP, SamplerMinP, SamplerXTC,
		SamplerTemperature, SamplerAdaptiveP:
		return nil
	case SamplerInfill:
		return nil
	default:
		return fmt.Errorf("unsupported sampler %q", stage)
	}
}

func cloneSamplerOrder(order []SamplerStage) []SamplerStage {
	if order == nil {
		return nil
	}
	return slices.Clone(order)
}
