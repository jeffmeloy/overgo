package projector

import (
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/strictjson"
)

const (
	audioProjectionProfileVersion   uint16 = 1
	audioProjectionProfileMediaType        = "application/vnd.overgo.audio-projection-profile+json"
	audioProjectionProfileSchema           = "overgo/audio-projection-profile/v1"
)

var audioProjectionProfileCodec = artifact.DocumentCodec[AudioProjectionProfile]{
	Name: "audio projection profile",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: audioProjectionProfileMediaType,
		Schema: audioProjectionProfileSchema,
	},
	Decode: func(data []byte, value *AudioProjectionProfile) error {
		var body audioProjectionProfileBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = AudioProjectionProfile{
			Version: body.Version, AttentionRopeFreqBase: body.AttentionRopeFreqBase,
		}
		return nil
	},
	Encode: func(value AudioProjectionProfile) ([]byte, error) {
		data, err := json.Marshal(audioProjectionProfileBody{
			Version: value.Version, AttentionRopeFreqBase: value.AttentionRopeFreqBase,
		})
		if err != nil {
			return nil, fmt.Errorf("projector: encode audio projection profile: %w", err)
		}
		return data, nil
	},
	Canonicalize: func(value *AudioProjectionProfile) error { return value.validate() },
	Clone:        artifact.DeepCopy[AudioProjectionProfile](),
	Identity:     func(value AudioProjectionProfile) artifact.ID { return value.ID },
	SetIdentity:  func(value *AudioProjectionProfile, id artifact.ID) { value.ID = id },
}

type audioProjectionProfileBody struct {
	Version               uint16  `json:"version"`
	AttentionRopeFreqBase float32 `json:"attention_rope_freq_base"`
}

// AudioProjectionProfile: immutable audio projector policy.
type AudioProjectionProfile struct {
	ID                    artifact.ID
	Version               uint16
	AttentionRopeFreqBase float32
}

func NewAudioProjectionProfile(attentionRopeFreqBase float32) (AudioProjectionProfile, error) {
	return audioProjectionProfileCodec.New(AudioProjectionProfile{
		Version: audioProjectionProfileVersion, AttentionRopeFreqBase: attentionRopeFreqBase,
	})
}

func (p AudioProjectionProfile) ValidateIdentity() error {
	return audioProjectionProfileCodec.ValidateIdentity(p)
}

func (p AudioProjectionProfile) Content() (artifact.Content, error) {
	return audioProjectionProfileCodec.Content(p)
}

func (p AudioProjectionProfile) validate() error {
	if p.Version != audioProjectionProfileVersion || !checked.PositiveFinite32(p.AttentionRopeFreqBase) {
		return errors.New("projector: invalid audio projection profile")
	}
	return nil
}
