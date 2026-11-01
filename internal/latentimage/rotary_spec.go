package latentimage

import (
	"fmt"
	"overgo/internal/checked"
	"path/filepath"
)

// mossRefinementBlocks and mossRefinementEpsilon belong to the pinned TextEmbedder.
// Reconsider these constants only when admitting a different source revision.
const (
	mossRefinementBlocks  = 2
	mossRefinementEpsilon = 1e-6
)

func readRotaryTransformer(root string) (TransformerSpec, error) {
	var config struct {
		Class     string  `json:"_class_name"`
		Layers    int     `json:"num_layers"`
		Dim       int     `json:"dim"`
		FFN       int     `json:"ffn_dim"`
		Heads     int     `json:"num_heads"`
		Text      int     `json:"text_dim"`
		Channels  int     `json:"latent_channels"`
		Frequency int     `json:"freq_dim"`
		RopeBase  float64 `json:"rope_base"`
	}
	if err := readJSON(filepath.Join(root, "transformer", "config.json"), &config); err != nil {
		return TransformerSpec{}, err
	}
	if config.Class != "MOSSImageTransformer" || !checked.PositiveInts(config.Layers, config.Dim, config.FFN, config.Heads, config.Text, config.Channels, config.Frequency) || config.Dim%config.Heads != 0 || !checked.PositiveFinite64(config.RopeBase) {
		return TransformerSpec{}, fmt.Errorf("rotary image: invalid transformer configuration")
	}
	return TransformerSpec{Layers: config.Layers, Hidden: config.Dim, Heads: config.Heads, HeadDim: config.Dim / config.Heads, Intermediate: config.FFN, TextHidden: config.Text, RefinerTextBlocks: mossRefinementBlocks, InChannels: config.Channels, TimestepEmbed: config.Frequency, RopeTheta: config.RopeBase}, nil
}
