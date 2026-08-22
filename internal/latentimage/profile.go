package latentimage

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/strictjson"
)

const (
	profileVersion   uint16 = 1
	profileMediaType        = "application/vnd.overgo.image-profile+json"
	profileSchema           = "overgo/image-profile/v1"
)

//go:embed profiles.json
var profileCatalogJSON []byte

var profileContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: profileMediaType, Schema: profileSchema,
}

var profileCodec = artifact.DocumentCodec[Profile]{
	Name: "image profile", Contract: profileContract,
	Decode: func(data []byte, value *Profile) error { return strictjson.DecodeBytes(data, value) },
	Encode: func(value Profile) ([]byte, error) {
		return json.Marshal(value)
	},
	Canonicalize: func(value *Profile) error { return value.validate() },
	Identity:     func(value Profile) artifact.ID { return value.ID },
	SetIdentity:  func(value *Profile, id artifact.ID) { value.ID = id },
}

type classPolicy struct {
	Pipeline          string `json:"pipeline"`
	Transformer       string `json:"transformer"`
	VAE               string `json:"vae"`
	TextEncoderModule string `json:"text_encoder_module"`
	TextEncoder       string `json:"text_encoder"`
	Tokenizer         string `json:"tokenizer"`
	Scheduler         string `json:"scheduler"`
}

type samplingPolicy struct {
	DefaultSteps          int     `json:"default_steps"`
	TrainTimesteps        int     `json:"train_timesteps"`
	TimestepFrequencyBase float64 `json:"timestep_frequency_base"`
	BaseImageSequence     int     `json:"base_image_sequence"`
	MaxImageSequence      int     `json:"max_image_sequence"`
	BaseShift             float64 `json:"base_shift"`
	MaxShift              float64 `json:"max_shift"`
}

// Profile: recipe-bound image execution facts.
type Profile struct {
	ID       artifact.ID    `json:"-"`
	Version  uint16         `json:"version"`
	Classes  classPolicy    `json:"classes"`
	Prompt   textTemplate   `json:"prompt"`
	Sampling samplingPolicy `json:"sampling"`
}

type catalogProfile struct {
	Pipeline              string       `json:"pipeline"`
	Transformer           string       `json:"transformer"`
	VAE                   string       `json:"vae"`
	TextEncoderModule     string       `json:"text_encoder_module"`
	TextEncoder           string       `json:"text_encoder"`
	Tokenizer             string       `json:"tokenizer"`
	Scheduler             string       `json:"scheduler"`
	DefaultSteps          int          `json:"default_steps"`
	TimestepFrequencyBase float64      `json:"timestep_frequency_base"`
	Prompt                textTemplate `json:"prompt"`
}

type schedulerProfile struct {
	ClassName       string  `json:"_class_name"`
	BaseImageSeqLen int     `json:"base_image_seq_len"`
	BaseShift       float64 `json:"base_shift"`
	MaxImageSeqLen  int     `json:"max_image_seq_len"`
	MaxShift        float64 `json:"max_shift"`
	TrainTimesteps  int     `json:"num_train_timesteps"`
	TimeShiftType   string  `json:"time_shift_type"`
	DynamicShifting bool    `json:"use_dynamic_shifting"`
}

func ResolveProfile(modelDir string) (Profile, error) {
	var index modelIndex
	if err := readJSON(filepath.Join(modelDir, "model_index.json"), &index); err != nil {
		return Profile{}, err
	}
	catalog, err := lookupCatalogProfile(index.ClassName)
	if err != nil {
		return Profile{}, err
	}
	var scheduler schedulerProfile
	if err := readJSON(filepath.Join(modelDir, "scheduler", "scheduler_config.json"), &scheduler); err != nil {
		return Profile{}, err
	}
	var tokenizer struct {
		ClassName string `json:"tokenizer_class"`
		PadToken  string `json:"pad_token"`
	}
	if err := readJSON(filepath.Join(modelDir, "tokenizer", "tokenizer_config.json"), &tokenizer); err != nil {
		return Profile{}, err
	}
	classes := classPolicy{
		Pipeline: catalog.Pipeline, Transformer: catalog.Transformer, VAE: catalog.VAE,
		TextEncoderModule: catalog.TextEncoderModule, TextEncoder: catalog.TextEncoder,
		Tokenizer: catalog.Tokenizer, Scheduler: catalog.Scheduler,
	}
	if index.Transformer[1] != classes.Transformer || index.VAE[1] != classes.VAE ||
		index.TextEncoder[1] != classes.TextEncoderModule || index.Tokenizer[1] != classes.Tokenizer ||
		index.Scheduler[1] != classes.Scheduler || tokenizer.ClassName != classes.Tokenizer ||
		scheduler.ClassName != classes.Scheduler || !scheduler.DynamicShifting || scheduler.TimeShiftType != "exponential" {
		return Profile{}, errors.New("latent image: artifact classes differ from image profile")
	}
	prompt := catalog.Prompt
	prompt.PadToken = tokenizer.PadToken
	return profileCodec.New(Profile{
		Version: profileVersion, Classes: classes, Prompt: prompt,
		Sampling: samplingPolicy{
			DefaultSteps: catalog.DefaultSteps, TrainTimesteps: scheduler.TrainTimesteps,
			TimestepFrequencyBase: catalog.TimestepFrequencyBase,
			BaseImageSequence:     scheduler.BaseImageSeqLen, MaxImageSequence: scheduler.MaxImageSeqLen,
			BaseShift: scheduler.BaseShift, MaxShift: scheduler.MaxShift,
		},
	})
}

func lookupCatalogProfile(pipeline string) (catalogProfile, error) {
	catalog, err := profileCatalog()
	if err != nil {
		return catalogProfile{}, err
	}
	for _, profile := range catalog {
		if profile.Pipeline == pipeline {
			return profile, nil
		}
	}
	return catalogProfile{}, fmt.Errorf("latent image: unsupported pipeline class %q", pipeline)
}

func supportsPipeline(pipeline string) (bool, error) {
	catalog, err := profileCatalog()
	if err != nil {
		return false, err
	}
	for _, profile := range catalog {
		if profile.Pipeline == pipeline {
			return true, nil
		}
	}
	return false, nil
}

func profileCatalog() ([]catalogProfile, error) {
	var catalog []catalogProfile
	if err := strictjson.DecodeBytes(profileCatalogJSON, &catalog); err != nil {
		return nil, fmt.Errorf("latent image: decode profile catalog: %w", err)
	}
	return catalog, nil
}

func ReadProfile(ctx context.Context, store artifact.Reader, id artifact.ID) (Profile, error) {
	return profileCodec.Require(ctx, store, id)
}

func (p Profile) Content() (artifact.Content, error) { return profileCodec.Content(p) }

func (p Profile) validateIdentity() error { return profileCodec.ValidateIdentity(p) }

func (p Profile) validate() error {
	if !checked.Equal(p.Version, profileVersion) || !checked.NonzeroAll(
		p.Classes.Pipeline, p.Classes.Transformer, p.Classes.VAE, p.Classes.TextEncoderModule,
		p.Classes.TextEncoder, p.Classes.Tokenizer, p.Classes.Scheduler, p.Prompt.Prefix,
		p.Prompt.Suffix, p.Prompt.PadToken) ||
		!checked.PositiveInts(p.Prompt.MaxTokens, p.Prompt.PrefixTokens, p.Prompt.SuffixTokens) {
		return errors.New("latent image: incomplete image profile")
	}
	sampling := p.Sampling
	if !checked.PositiveInts(sampling.DefaultSteps, sampling.TrainTimesteps, sampling.BaseImageSequence) ||
		!checked.GreaterInt(sampling.MaxImageSequence, sampling.BaseImageSequence) ||
		!checked.PositiveFinite64(sampling.TimestepFrequencyBase) || !checked.PositiveFinite64(sampling.BaseShift) ||
		!checked.AtLeastFinite64(sampling.MaxShift, sampling.BaseShift) {
		return errors.New("latent image: invalid sampling profile")
	}
	return nil
}

func (p samplingPolicy) dynamicShiftMu(imageSequence int) float64 {
	if checked.AtMostInt(imageSequence, p.BaseImageSequence) {
		return p.BaseShift
	}
	if checked.AtLeastInt(imageSequence, p.MaxImageSequence) {
		return p.MaxShift
	}
	position := float64(imageSequence-p.BaseImageSequence) / float64(p.MaxImageSequence-p.BaseImageSequence)
	return p.BaseShift + position*(p.MaxShift-p.BaseShift)
}
