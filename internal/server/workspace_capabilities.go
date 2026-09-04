package server

import (
	"context"

	"overgo/internal/media"
)

// The capability document (professional GUI campaign, gui-simplify/
// one-capability-source): one server-declared statement of what the served
// model accepts and can do, published with the workspace manifest and read
// by the shell, the composer and the model selector. The client makes no
// capability decision of its own: an attachment kind, a composer mode or a
// tab is accepted or refused from this document before any request is
// sent, and a new server capability reaches the GUI by declaration alone.
type workspaceModelCapabilities struct {
	ID            string                   `json:"id"`
	Name          string                   `json:"name"`
	ContextLength uint32                   `json:"context_length"`
	Generation    propertiesSamplingParams `json:"generation"`
	Modalities    map[string]bool          `json:"modalities"`
	Media         workspaceMediaLimits     `json:"media"`
	Modes         []workspaceMode          `json:"modes"`
}

// workspaceMediaLimits: the media kinds the served model accepts as prompt
// content parts and the bounds the server enforces on them, so the
// composer refuses an oversized or unsupported attachment at attach time
// with the same rule the request would meet.
type workspaceMediaLimits struct {
	Accept            []string `json:"accept"`
	MaxImageBytes     uint64   `json:"max_image_bytes"`
	MaxMediaBytes     uint64   `json:"max_media_bytes"`
	MaxImageDimension uint64   `json:"max_image_dimension"`
	MaxImagePixels    uint64   `json:"max_image_pixels"`
}

// workspaceMode: one thing the composer can ask the served model for.
type workspaceMode struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Enabled bool   `json:"enabled"`
	Refusal string `json:"refusal,omitzero"`
}

// workspaceModelCapabilities derives the document from the served
// generator, the loaded projectors, the media decoders this binary
// registers, and the same capability switch the tabs use.
func (h *Handler) workspaceModelCapabilities(ctx context.Context) (workspaceModelCapabilities, bool) {
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		return workspaceModelCapabilities{}, false
	}
	model := api.ModelProperties()
	image := h.config.ImageProjector != nil || h.config.Qwen3VLProjector != nil
	audio := h.config.AudioProjector != nil
	// Video rides the image projector frame by frame: GIF frames decode
	// natively, MP4 only through FFmpeg.
	video := image
	accept := []string{}
	if image {
		accept = append(accept, media.PNGMediaType, "image/jpeg", media.GIFMediaType)
	}
	if audio {
		accept = append(accept, "audio/wav")
	}
	if video {
		accept = append(accept, media.GIFMediaType)
		if h.config.FFmpegPath != "" {
			accept = append(accept, media.MP4MediaType)
		}
	}
	document := workspaceModelCapabilities{
		ID:            h.config.ModelID,
		Name:          model.Name,
		ContextLength: model.ContextLength,
		Generation:    h.defaultSamplingParams(),
		Modalities:    map[string]bool{"text": true, "image": image, "audio": audio, "video": video},
		Media: workspaceMediaLimits{
			Accept:        dedupeStrings(accept),
			MaxImageBytes: maxImageBytes, MaxMediaBytes: maxMediaBytes,
			MaxImageDimension: maxImageDimension, MaxImagePixels: maxImagePixels,
		},
	}
	modes := []struct {
		id, label, capability string
	}{
		{"chat", "Chat", "analysis.logits"},
		{"image-gen", "Images", "workflow.image"},
		{"video-gen", "Video", "workflow.video"},
		{"video-edit", "Video edit", "workflow.video-edit"},
		{"speech", "Speech", "workflow.speech"},
		{"embeddings", "Embeddings", "embeddings"},
		{"rerank", "Rerank", "rerank"},
	}
	for _, mode := range modes {
		enabled, refusal := h.workspaceCapability(ctx, mode.capability)
		document.Modes = append(document.Modes, workspaceMode{ID: mode.id, Label: mode.label, Enabled: enabled, Refusal: refusal})
	}
	return document, true
}

func dedupeStrings(values []string) []string {
	seen := map[string]bool{}
	result := make([]string, 0, len(values))
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	return result
}
