package scratchmodel

import (
	"context"
	_ "embed"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
)

const (
	derivationProfileMediaType = "application/vnd.overgo.derivation-profile+json"
	derivationProfileSchema    = "overgo/derivation-profile/v1"
	activeDerivationProfile    = "profile.active.scratch-derivation"
)

//go:embed derivation_profiles.json
var derivationProfileCatalogJSON []byte

// DerivationProfile owns corpus-to-topology policy.
type DerivationProfile struct {
	Version            string      `json:"version"`
	SplitDenominator   int         `json:"split_denominator"`
	StableSplitMinimum int         `json:"stable_split_minimum"`
	MinimumLayers      int         `json:"minimum_layers"`
	MinimumMLPFactor   int         `json:"minimum_mlp_factor"`
	MLPBudget          int         `json:"mlp_budget"`
	Epsilon            float64     `json:"epsilon"`
	Optimizer          artifact.ID `json:"optimizer"`
	ID                 artifact.ID `json:"-"`
}

var derivationProfileCodec = artifact.JSONDocumentCodec(
	"derivation profile", artifact.KindProfile, derivationProfileMediaType, derivationProfileSchema,
	func(profile *DerivationProfile) error { return profile.validate() },
	func(profile DerivationProfile) artifact.ID { return profile.ID },
	func(profile *DerivationProfile, id artifact.ID) { profile.ID = id }, nil,
)

// NewDerivationProfile identifies validated policy.
func NewDerivationProfile(profile DerivationProfile) (DerivationProfile, error) {
	profile.ID = artifact.ID{}
	return derivationProfileCodec.New(profile)
}

// ValidateIdentity checks content identity.
func (profile DerivationProfile) ValidateIdentity() error {
	return derivationProfileCodec.ValidateIdentity(profile)
}

// Content returns publication bytes.
func (profile DerivationProfile) Content() (artifact.Content, error) {
	return derivationProfileCodec.Content(profile)
}

// PublishDerivationProfileCatalog publishes missing bootstrap authority.
func PublishDerivationProfileCatalog(
	ctx context.Context,
	repository artifact.Repository,
) (DerivationProfile, error) {
	if ctx == nil || repository == nil {
		return DerivationProfile{}, errors.New("scratch model: profile repository required")
	}
	if current, found, err := repository.ResolveAlias(ctx, activeDerivationProfile); err != nil {
		return DerivationProfile{}, err
	} else if found {
		return derivationProfileCodec.Require(ctx, repository, current)
	}
	var declaration DerivationProfile
	if err := strictjson.DecodeBytes(derivationProfileCatalogJSON, &declaration); err != nil {
		return DerivationProfile{}, err
	}
	optimizer := trainingprogram.BuiltinOptimizerPolicy()
	declaration.Optimizer = optimizer.ID
	profile, err := NewDerivationProfile(declaration)
	if err != nil {
		return DerivationProfile{}, err
	}
	content, err := profile.Content()
	if err != nil {
		return DerivationProfile{}, err
	}
	optimizerContent, err := optimizer.Content()
	if err != nil {
		return DerivationProfile{}, err
	}
	_, err = repository.Commit(ctx, artifact.Batch{
		Key:      "catalog/scratch-derivation-profile/" + profile.ID.String(),
		Contents: []artifact.Content{content, optimizerContent},
		Aliases:  []artifact.AliasBinding{{Name: activeDerivationProfile, Target: profile.ID}},
	})
	return profile, err
}

// ResolveActiveDerivationProfile loads current construction authority.
func ResolveActiveDerivationProfile(ctx context.Context, reader artifact.Reader) (DerivationProfile, error) {
	id, found, err := artifact.ResolveAlias(ctx, reader, activeDerivationProfile)
	if err != nil {
		return DerivationProfile{}, err
	}
	if !found {
		return DerivationProfile{}, errors.New("scratch model: active derivation profile absent")
	}
	return derivationProfileCodec.Require(ctx, reader, id)
}

func (profile DerivationProfile) validate() error {
	if strings.TrimSpace(profile.Version) == "" || strings.ContainsAny(profile.Version, "\x00\r\n") ||
		profile.SplitDenominator <= 0 || profile.StableSplitMinimum <= 0 ||
		profile.MinimumLayers <= 0 || profile.MinimumMLPFactor <= 0 || profile.MLPBudget <= 0 ||
		!checked.PositiveFinite64(profile.Epsilon) || profile.Optimizer.Kind() != artifact.KindProfile {
		return errors.New("scratch model: invalid derivation profile")
	}
	return nil
}
