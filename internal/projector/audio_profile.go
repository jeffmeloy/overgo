package projector

import (
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	AudioProjectionProfileVersion   uint16 = 1
	AudioProjectionProfileMediaType        = "application/vnd.overgo.audio-projection-profile+json"
	AudioProjectionProfileSchema           = "overgo/audio-projection-profile/v1"
)

var audioProjectionProfileCodec = artifact.DocumentCodec[AudioProjectionProfile]{
	Name: "audio projection profile",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: AudioProjectionProfileMediaType,
		Schema: AudioProjectionProfileSchema,
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
	Canonicalize: func(value *AudioProjectionProfile) error { return value.validateShape() },
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
		Version: AudioProjectionProfileVersion, AttentionRopeFreqBase: attentionRopeFreqBase,
	})
}

func ParseAudioProjectionProfile(data []byte) (AudioProjectionProfile, error) {
	return audioProjectionProfileCodec.Parse(data)
}

func (p AudioProjectionProfile) ValidateIdentity() error {
	return audioProjectionProfileCodec.ValidateIdentity(p)
}

func (p AudioProjectionProfile) Content() (artifact.Content, error) {
	return audioProjectionProfileCodec.Content(p)
}

func (p AudioProjectionProfile) validateShape() error {
	if p.Version != AudioProjectionProfileVersion || p.AttentionRopeFreqBase <= 0 ||
		!finite32(p.AttentionRopeFreqBase) {
		return errors.New("projector: invalid audio projection profile")
	}
	return nil
}
