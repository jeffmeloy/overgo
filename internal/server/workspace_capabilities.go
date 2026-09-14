package server

import (
	"context"
	"slices"

	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// The capability document (professional GUI campaign, gui-simplify/
// one-capability-source): one server-declared statement of what the served
// model accepts and can do, published with the workspace manifest and read
// by the shell, the composer and the model selector. The client makes no
// capability decision of its own: an attachment kind, a composer mode or a
// tab is accepted or refused from this document before any request is
// sent, and a new server capability reaches the GUI by declaration alone.
type workspaceModelCapabilities struct {
	ID              string                   `json:"id"`
	Name            string                   `json:"name"`
	Recipe          string                   `json:"recipe,omitzero"`
	Model           string                   `json:"model,omitzero"`
	MaxOutputTokens int                      `json:"max_output_tokens"`
	ContextLength   uint32                   `json:"context_length"`
	Generation      propertiesSamplingParams `json:"generation"`
	Modalities      map[string]bool          `json:"modalities"`
	Media           workspaceMediaLimits     `json:"media"`
	Modes           []workspaceMode          `json:"modes"`
	// Remote: served through the relay at a hosted provider; the page marks
	// its turns (not reproducible from the store).
	Remote bool `json:"remote"`
}

// workspaceMediaLimits: the media kinds the served model accepts as prompt
// content parts and the bounds the server enforces on them, so the
// composer refuses an oversized or unsupported attachment at attach time
// with the same rule the request would meet.
type workspaceMediaLimits struct {
	Accept            []string `json:"accept"`
	IntakeAccept      []string `json:"intake_accept"`
	MaxImageBytes     uint64   `json:"max_image_bytes"`
	MaxMediaBytes     uint64   `json:"max_media_bytes"`
	MaxImageDimension uint64   `json:"max_image_dimension"`
	MaxImagePixels    uint64   `json:"max_image_pixels"`
	// Refusals: why a kind the model does not accept is refused, by kind
	// (image, audio, video) or by media type.
	Refusals map[string]string `json:"refusals"`
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
	var model inference.ModelProperties
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		model = api.ModelProperties()
	} else if !h.hasWorkspaceCapability(ctx, WorkflowGeneration, "") {
		return workspaceModelCapabilities{}, false
	}
	image := h.config.ImageProjector != nil || h.config.Qwen3VLProjector != nil
	audio := h.config.AudioProjector != nil
	// Video rides the image projector frame by frame: GIF frames decode
	// natively, MP4 only through FFmpeg.
	video := image
	refusals := map[string]string{}
	if !image {
		refusals["image"] = "the served model has no image projector loaded"
	}
	if !audio {
		refusals["audio"] = "the served model has no audio projector loaded"
	}
	if !video {
		refusals["video"] = "video rides the image projector, and none is loaded"
	} else if h.config.FFmpegPath == "" {
		refusals[media.MP4MediaType] = "MP4 decoding needs FFmpeg, which is not configured"
	}
	document := workspaceModelCapabilities{
		ID:              h.config.ModelID,
		Name:            model.Name,
		ContextLength:   model.ContextLength,
		Remote:          model.Architecture == runrecord.BackendRemote,
		Generation:      h.defaultSamplingParams(),
		MaxOutputTokens: h.config.MaxTokens,
		Modalities:      map[string]bool{"text": true, "image": image, "audio": audio, "video": video, "document": true},
		Media: workspaceMediaLimits{
			Accept: h.acceptedMedia(image, audio, video), Refusals: refusals,
			IntakeAccept:  h.acceptedMedia(true, true, true),
			MaxImageBytes: maxImageBytes, MaxMediaBytes: maxMediaBytes,
			MaxImageDimension: maxImageDimension, MaxImagePixels: maxImagePixels,
		},
	}
	if modelID, recipeID, ok := h.servingIdentity(recipe.TaskInference); ok {
		document.Model, document.Recipe = modelID.String(), recipeID.String()
	}
	document.Generation.MaxTokens = min(document.Generation.MaxTokens, document.MaxOutputTokens)
	document.Generation.NPredict = min(document.Generation.NPredict, document.MaxOutputTokens)
	modes := []struct {
		id, label, capability string
	}{
		{"chat", "Chat", "analysis.logits"},
		{"image-gen", "Images", "workflow.image"},
		{"video-gen", "Video", "workflow.video"},
		{"speech", "Speech", "workflow.speech"},
		{"vqa", "Ask about an image", "workflow.vqa"},
		{"transcription", "Transcribe", "workflow.transcription"},
		{"embeddings", "Embeddings", "embeddings"},
		{"rerank", "Rerank", "rerank"},
	}
	for _, mode := range modes {
		enabled, refusal := h.workspaceCapability(ctx, mode.capability)
		document.Modes = append(document.Modes, workspaceMode{ID: mode.id, Label: mode.label, Enabled: enabled, Refusal: refusal})
	}
	// Agent mode is declared with the rest: enabled when an active agent
	// definition exists, refused with the reason otherwise.
	agentMode := workspaceMode{ID: "agent", Label: "Agent", Refusal: "no active agent definition"}
	if h.repository != nil {
		if agents, err := h.agentInventory(ctx); err == nil && slices.ContainsFunc(agents, func(entry AgentInventoryEntry) bool { return entry.State == "active" }) {
			agentMode.Enabled, agentMode.Refusal = true, ""
		}
	}
	document.Modes = append(document.Modes, agentMode)
	return document, true
}

// acceptedMedia: media types accepted for the given projectors; documents
// need none (text passes through, PDF is extracted); MP4 needs FFmpeg.
// All three true = every type the server decodes (the attachment intake).
func (h *Handler) acceptedMedia(image, audio, video bool) []string {
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
	return dedupeStrings(append(accept, "text/plain", "text/markdown", "text/csv", "application/json", media.PDFMediaType))
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
