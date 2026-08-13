package modelrecipe

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sync"
)

//go:embed image_profiles.json
var imageProfileData []byte

type ImageRecognition struct {
	Pipeline    string `json:"pipeline"`
	Transformer string `json:"transformer"`
	VAE         string `json:"vae"`
	TextEncoder string `json:"text_encoder"`
	Tokenizer   string `json:"tokenizer"`
	Scheduler   string `json:"scheduler"`
}

type ImageConditioning struct {
	Prefix          string `json:"prefix"`
	Suffix          string `json:"suffix"`
	MaxPromptTokens int    `json:"max_prompt_tokens"`
	PadToken        string `json:"pad_token"`
	PrefixTokens    int    `json:"prefix_tokens"`
	SuffixTokens    int    `json:"suffix_tokens"`
}

type ImageSampling struct {
	Steps             int     `json:"steps"`
	NumTrainTimesteps int     `json:"num_train_timesteps"`
	DynamicShiftMu    float64 `json:"dynamic_shift_mu"`
}

// ImageProfile owns recognition, conditioning, and sampling policy.
type ImageProfile struct {
	ID           string            `json:"id"`
	Family       string            `json:"family"`
	Recognition  ImageRecognition  `json:"recognition"`
	Conditioning ImageConditioning `json:"conditioning"`
	Sampling     ImageSampling     `json:"sampling"`
}

var imageProfiles = sync.OnceValues(loadImageProfiles)

func loadImageProfiles() ([]ImageProfile, error) {
	var profiles []ImageProfile
	if err := json.Unmarshal(imageProfileData, &profiles); err != nil {
		return nil, fmt.Errorf("model recipe image profiles: %w", err)
	}
	seen := make(map[string]struct{}, len(profiles))
	for _, profile := range profiles {
		if err := profile.validate(); err != nil {
			return nil, err
		}
		if _, exists := seen[profile.Recognition.Pipeline]; exists {
			return nil, fmt.Errorf("model recipe image profiles: duplicate pipeline %q", profile.Recognition.Pipeline)
		}
		seen[profile.Recognition.Pipeline] = struct{}{}
	}
	return profiles, nil
}

func (p ImageProfile) validate() error {
	r, c, s := p.Recognition, p.Conditioning, p.Sampling
	if p.ID == "" || p.Family == "" || r.Pipeline == "" || r.Transformer == "" || r.VAE == "" ||
		r.TextEncoder == "" || r.Tokenizer == "" || r.Scheduler == "" || c.Prefix == "" || c.Suffix == "" ||
		c.PadToken == "" || c.MaxPromptTokens <= 0 || c.PrefixTokens <= 0 || c.SuffixTokens <= 0 ||
		s.Steps <= 0 || s.NumTrainTimesteps <= 0 || s.DynamicShiftMu <= 0 {
		return fmt.Errorf("model recipe image profile %q: incomplete", p.ID)
	}
	return nil
}

// ImageProfileForPipeline resolves artifact-bound policy by pipeline class.
func ImageProfileForPipeline(pipeline string) (ImageProfile, bool, error) {
	profiles, err := imageProfiles()
	if err != nil {
		return ImageProfile{}, false, err
	}
	for _, profile := range profiles {
		if profile.Recognition.Pipeline == pipeline {
			return profile, true, nil
		}
	}
	return ImageProfile{}, false, nil
}
