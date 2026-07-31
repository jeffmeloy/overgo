package sampling

import (
	"fmt"
	"strings"
)

// SamplerStage names a repeatable transform in the non-Mirostat sampler
// chain. The spellings follow llama.cpp's public CLI and server API.
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

// DefaultSamplerOrder returns the supported subset of llama.cpp's default
// chain. Unsupported disabled-by-default stages are intentionally omitted.
func DefaultSamplerOrder() []SamplerStage {
	return append([]SamplerStage(nil), defaultSamplerOrder...)
}

// ParseSamplerOrder parses llama.cpp's semicolon-delimited sampler names.
// An empty string denotes an explicitly empty chain.
func ParseSamplerOrder(value string) ([]SamplerStage, error) {
	if value == "" || value == "none" {
		return []SamplerStage{}, nil
	}
	return ParseSamplerNames(strings.Split(value, ";"))
}

// ParseSamplerNames validates the server API's ordered sampler-name array.
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
	return append([]SamplerStage{}, order...)
}
