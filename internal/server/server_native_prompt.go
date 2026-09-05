package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/projector"
	"overgo/internal/strictjson"
	"overgo/internal/tokenizer"
)

func (h *Handler) parseNativePrompts(ctx context.Context, raw json.RawMessage) ([]nativePrompt, error) {
	if len(raw) == 0 {
		return nil, errors.New("prompt is required")
	}
	trimmedRaw := bytes.TrimSpace(raw)
	if len(trimmedRaw) > 0 && trimmedRaw[0] == '{' {
		prompt, err := h.parseNativeMultimodalPrompt(ctx, trimmedRaw)
		if err != nil {
			return nil, err
		}
		return []nativePrompt{prompt}, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		prompt, err := h.parseNativePrompt(raw)
		if err != nil {
			return nil, err
		}
		return []nativePrompt{prompt}, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		_, promptErr := h.parseNativePrompt(raw)
		return nil, promptErr
	}
	if len(parts) == 0 {
		return nil, errors.New("prompt list must not be empty")
	}
	allStrings := true
	hasNested := false
	for _, part := range parts {
		var partText string
		if json.Unmarshal(part, &partText) != nil {
			allStrings = false
		}
		trimmed := strings.TrimSpace(string(part))
		if strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") {
			hasNested = true
		}
	}
	if !allStrings && !hasNested {
		prompt, err := h.parseNativePrompt(raw)
		if err != nil {
			return nil, err
		}
		return []nativePrompt{prompt}, nil
	}
	if len(parts) > 64 {
		return nil, errors.New("prompt batch count exceeds 64")
	}
	result := make([]nativePrompt, 0, len(parts))
	for index, part := range parts {
		prompt, err := h.parseNativePrompt(part)
		if err != nil {
			return nil, fmt.Errorf("prompt %d: %w", index, err)
		}
		result = append(result, prompt)
	}
	return result, nil
}

func (h *Handler) parseNativeMultimodalPrompt(ctx context.Context, raw json.RawMessage) (nativePrompt, error) {
	if h.config.Qwen3VLProjector == nil && h.config.ImageProjector == nil && h.config.AudioProjector == nil {
		return nativePrompt{}, errors.New("multimodal data provided, but the server has no multimodal projector")
	}
	var document struct {
		PromptString   string   `json:"prompt_string"`
		MultimodalData []string `json:"multimodal_data"`
	}
	if err := strictjson.DecodeBytes(raw, &document); err != nil {
		return nativePrompt{}, errors.New("prompt object must contain prompt_string and multimodal_data")
	}
	if document.PromptString == "" {
		return nativePrompt{}, errors.New("prompt_string must not be empty")
	}
	if len(document.MultimodalData) == 0 || len(document.MultimodalData) > 8 {
		return nativePrompt{}, errors.New("multimodal prompt requires one to eight media items")
	}
	const marker = "<__media__>"
	if strings.Count(document.PromptString, marker) != len(document.MultimodalData) {
		return nativePrompt{}, errors.New("prompt_string media marker count must match multimodal_data")
	}
	segments := strings.Split(document.PromptString, marker)
	before, after := segments[0], segments[1]
	images := make([][]byte, 0, len(document.MultimodalData))
	media := make([]nativeMedia, 0, len(document.MultimodalData))
	hasAudio := false
	for index, encoded := range document.MultimodalData {
		if strings.HasPrefix(encoded, "data:audio/") {
			if h.config.AudioProjector == nil {
				return nativePrompt{}, errors.New("audio data provided, but the server has no audio projector")
			}
			audio, err := decodeNativeAudioData(ctx, encoded)
			if err != nil {
				return nativePrompt{}, fmt.Errorf("multimodal_data audio %d: %w", index, err)
			}
			hasAudio = true
			media = append(media, nativeMedia{Kind: projector.MediaAudio, Audio: audio})
			continue
		}
		if strings.HasPrefix(encoded, "data:video/") {
			if len(document.MultimodalData) != 1 {
				return nativePrompt{}, errors.New("encoded video cannot be combined with other media")
			}
			if h.config.ImageProjector == nil || !h.config.ImageProjector.Capabilities().Video {
				return nativePrompt{}, errors.New("video data provided, but the server has no video projector")
			}
			video, err := h.resolveVideoData(ctx, encoded)
			if err != nil {
				return nativePrompt{}, fmt.Errorf("multimodal_data video: %w", err)
			}
			return nativePrompt{
				Text: document.PromptString, Response: document.PromptString,
				Video: video, BeforeMedia: before, AfterMedia: after,
			}, nil
		}
		if h.config.Qwen3VLProjector == nil && h.config.ImageProjector == nil {
			return nativePrompt{}, errors.New("image data provided, but the server has no image projector")
		}
		imageData, err := h.resolveImageData(ctx, encoded)
		if err != nil {
			return nativePrompt{}, fmt.Errorf("multimodal_data image %d: %w", index, err)
		}
		images = append(images, imageData)
		media = append(media, nativeMedia{Kind: projector.MediaImage, Image: imageData})
	}
	if len(images) != 0 {
		if err := validateMultimodalImages(images); err != nil {
			return nativePrompt{}, err
		}
	}
	if hasAudio && len(media) == 1 {
		return nativePrompt{
			Text: document.PromptString, Response: document.PromptString,
			Audio: media[0].Audio, BeforeMedia: before, AfterMedia: after,
		}, nil
	}
	if hasAudio {
		return nativePrompt{
			Text: document.PromptString, Response: document.PromptString,
			Media: media, MediaText: segments, BeforeMedia: before, AfterMedia: after,
			MediaHistory: true,
		}, nil
	}
	if len(images) == 0 {
		return nativePrompt{}, errors.New("multimodal media data is empty")
	}
	return nativePrompt{
		Text: document.PromptString, Response: document.PromptString,
		Image: images[0], Images: images, MediaText: segments, BeforeMedia: before, AfterMedia: after,
	}, nil
}

func decodeNativeAudioData(ctx context.Context, encoded string) ([]float32, error) {
	header, payload, ok := strings.Cut(encoded, ",")
	if !ok || !strings.HasPrefix(header, "data:audio/") || !strings.HasSuffix(header, ";base64") {
		return nil, errors.New("multimodal_data audio must use a base64 data URI")
	}
	if base64.StdEncoding.DecodedLen(len(payload)) > maxMediaBytes+2 {
		return nil, errors.New("multimodal_data audio exceeds decoded media limit")
	}
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil || len(decoded) == 0 {
		return nil, errors.New("multimodal_data audio is not valid base64")
	}
	if len(decoded) > maxMediaBytes {
		return nil, errors.New("multimodal_data audio exceeds decoded media limit")
	}
	return decodeNativeAudioBytes(ctx, decoded)
}

func decodeNativeAudioBytes(ctx context.Context, decoded []byte) ([]float32, error) {
	if len(decoded) == 0 {
		return nil, errors.New("multimodal_data audio is empty")
	}
	if len(decoded) > maxMediaBytes {
		return nil, errors.New("multimodal_data audio exceeds decoded media limit")
	}
	if !bytes.HasPrefix(decoded, []byte("RIFF")) {
		return nil, errors.New("multimodal_data audio must contain RIFF/WAVE audio")
	}
	// Every supported WAV scalar consumes at least one encoded byte; the
	// existing encoded-media bound therefore also bounds scalar expansion.
	audio, _, err := media.DecodeAudio(ctx, decoded, uint64(len(decoded)))
	if err != nil {
		return nil, fmt.Errorf("multimodal_data audio: %w", err)
	}
	if audio.Format.Channels != 1 {
		return nil, errors.New("multimodal_data audio requires mono samples")
	}
	if audio.Format.SampleRate != 16000 {
		return nil, fmt.Errorf("multimodal_data audio sample rate %d Hz; want 16000 Hz", audio.Format.SampleRate)
	}
	return audio.Samples, nil
}

func (h *Handler) resolveAudioData(ctx context.Context, source, format string) ([]float32, int, error) {
	if isRemoteMediaSource(source) {
		if format != "" && !strings.EqualFold(format, "wav") {
			return nil, 0, errors.New("input_audio.format must be wav")
		}
		data, err := h.mediaFetcher.fetch(ctx, source, "audio")
		if err != nil {
			return nil, 0, err
		}
		samples, err := decodeNativeAudioBytes(ctx, data)
		return samples, len(data), err
	}
	if !strings.EqualFold(format, "wav") {
		return nil, 0, errors.New("input_audio.format must be wav")
	}
	if strings.HasPrefix(source, "data:audio/") {
		samples, err := decodeNativeAudioData(ctx, source)
		return samples, encodedMediaSize(source), err
	}
	samples, err := decodeNativeAudioData(ctx, "data:audio/wav;base64,"+source)
	return samples, base64.StdEncoding.DecodedLen(len(source)), err
}

func encodedMediaSize(source string) int {
	_, payload, ok := strings.Cut(source, ",")
	if !ok {
		return 0
	}
	return base64.StdEncoding.DecodedLen(len(payload))
}

func decodeNativeImageData(encoded string) ([]byte, error) {
	if strings.HasPrefix(encoded, "data:") {
		header, payload, ok := strings.Cut(encoded, ",")
		if !ok || !strings.HasPrefix(header, "data:image/") || !strings.HasSuffix(header, ";base64") {
			return nil, errors.New("multimodal_data data URI must contain a base64 image")
		}
		encoded = payload
	}
	if encoded == "" {
		return nil, errors.New("multimodal_data image is empty")
	}
	if base64.StdEncoding.DecodedLen(len(encoded)) > maxImageBytes+2 {
		return nil, errors.New("multimodal_data image exceeds decoded image limit")
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, errors.New("multimodal_data image is not valid base64")
	}
	if len(decoded) == 0 {
		return nil, errors.New("multimodal_data image is empty")
	}
	if len(decoded) > maxImageBytes {
		return nil, errors.New("multimodal_data image exceeds decoded image limit")
	}
	return decoded, nil
}

func (h *Handler) resolveImageData(ctx context.Context, source string) ([]byte, error) {
	if !isRemoteMediaSource(source) {
		return decodeNativeImageData(source)
	}
	data, err := h.mediaFetcher.fetch(ctx, source, "image")
	if err != nil {
		return nil, err
	}
	if len(data) > maxImageBytes {
		return nil, errors.New("multimodal_data image exceeds decoded image limit")
	}
	return data, nil
}

func (h *Handler) resolveVideoData(ctx context.Context, source string) ([]byte, error) {
	if isRemoteMediaSource(source) {
		data, err := h.mediaFetcher.fetch(ctx, source, "video")
		if err != nil {
			return nil, err
		}
		if len(data) > maxMediaBytes {
			return nil, errors.New("video exceeds decoded media limit")
		}
		return data, nil
	}
	comma := strings.IndexByte(source, ',')
	if comma >= 0 {
		metadata := strings.ToLower(source[:comma])
		if !strings.HasPrefix(metadata, "data:video/") || !strings.HasSuffix(metadata, ";base64") {
			return nil, errors.New("video data URI must contain base64 video")
		}
		source = source[comma+1:]
	}
	if base64.StdEncoding.DecodedLen(len(source)) > maxMediaBytes+2 {
		return nil, errors.New("video exceeds decoded media limit")
	}
	data, err := base64.StdEncoding.DecodeString(source)
	if err != nil {
		return nil, errors.New("video is not valid base64")
	}
	if len(data) == 0 {
		return nil, errors.New("video is empty")
	}
	if len(data) > maxMediaBytes {
		return nil, errors.New("video exceeds decoded media limit")
	}
	return data, nil
}

func isRemoteMediaSource(source string) bool {
	lower := strings.ToLower(strings.TrimSpace(source))
	return strings.HasPrefix(lower, "https://") || strings.HasPrefix(lower, "http://")
}

func validateMultimodalImages(data [][]byte) error {
	if len(data) == 0 {
		return errors.New("multimodal image data is empty")
	}
	var totalBytes, totalPixels uint64
	for index, encoded := range data {
		if len(encoded) > maxImageBytes {
			return fmt.Errorf("multimodal_data image %d exceeds decoded image limit", index)
		}
		totalBytes += uint64(len(encoded))
		if totalBytes > maxMediaBytes {
			return errors.New("multimodal images exceed decoded media limit")
		}
		config, _, err := image.DecodeConfig(bytes.NewReader(encoded))
		if err != nil {
			return fmt.Errorf("multimodal_data image %d is unsupported", index)
		}
		if config.Width <= 0 || config.Height <= 0 ||
			config.Width > maxImageDimension || config.Height > maxImageDimension {
			return fmt.Errorf("multimodal_data image %d dimensions exceed limit", index)
		}
		pixels := uint64(config.Width) * uint64(config.Height)
		if pixels > maxImagePixels {
			return fmt.Errorf("multimodal_data image %d pixel count exceeds limit", index)
		}
		totalPixels += pixels
		if totalPixels > maxRequestImagePixels {
			return errors.New("multimodal images exceed aggregate pixel limit")
		}
	}
	return nil
}

func decodeMultimodalImages(data [][]byte) ([]image.Image, error) {
	if err := validateMultimodalImages(data); err != nil {
		return nil, err
	}
	images := make([]image.Image, len(data))
	for index, encoded := range data {
		input, _, err := image.Decode(bytes.NewReader(encoded))
		if err != nil {
			return nil, fmt.Errorf("multimodal_data image %d is unsupported", index)
		}
		images[index] = input
	}
	return images, nil
}

func (h *Handler) projectNativeMultimodalPrompt(
	ctx context.Context,
	prompt nativePrompt,
) (nativePrompt, inference.ProjectedInputs, error) {
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: generator cannot tokenize multimodal prompt")
	}
	var projected projector.MultimodalPrompt
	var err error
	thinking := true
	if prompt.Thinking != nil {
		thinking = *prompt.Thinking
	}
	if len(prompt.Video) > 0 {
		video := h.config.ImageProjector
		if video == nil || !video.Capabilities().Video {
			return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: video projector is unavailable")
		}
		fps := prompt.VideoFPS
		if fps == 0 {
			fps = h.config.VideoFPS
		}
		frames, decodeErr := media.DecodeEncodedVideo(
			ctx, prompt.Video, h.config.FFmpegPath, fps, h.config.VideoMaxFrames,
		)
		if decodeErr != nil {
			return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: decode video: %w", decodeErr)
		}
		if validationErr := validateVideoFrames(frames); validationErr != nil {
			return nativePrompt{}, inference.ProjectedInputs{}, validationErr
		}
		projected, err = video.BuildVideoPrompt(
			ctx, tokenizerAPI, frames, prompt.BeforeMedia, prompt.AfterMedia, fps, thinking,
		)
	} else if len(prompt.Media) > 0 {
		mixed := h.config.ImageProjector
		if mixed == nil || !mixed.Capabilities().MediaHistory {
			mixed = h.config.AudioProjector
		}
		if mixed == nil || !mixed.Capabilities().MediaHistory {
			return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: selected projector does not support ordered mixed media")
		}
		imageData := make([][]byte, 0, len(prompt.Media))
		for _, item := range prompt.Media {
			if item.Kind == projector.MediaImage {
				imageData = append(imageData, item.Image)
			}
		}
		var images []image.Image
		if len(imageData) != 0 {
			var decodeErr error
			images, decodeErr = decodeMultimodalImages(imageData)
			if decodeErr != nil {
				return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: %w", decodeErr)
			}
		}
		inputs := make([]projector.MediaInput, len(prompt.Media))
		imageIndex := 0
		for index, item := range prompt.Media {
			if item.Kind == projector.MediaImage {
				inputs[index] = projector.NewImageMediaInput(images[imageIndex])
				imageIndex++
			} else {
				inputs[index] = projector.NewAudioMediaInput(item.Audio)
			}
		}
		projected, err = mixed.BuildMediaHistoryPrompt(ctx, tokenizerAPI, inputs, prompt.MediaText)
	} else if len(prompt.Audio) > 0 {
		if h.config.AudioProjector == nil {
			return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: audio projector is unavailable")
		}
		projected, err = h.config.AudioProjector.BuildAudioPrompt(
			ctx, tokenizerAPI, prompt.Audio, prompt.BeforeMedia, prompt.AfterMedia,
		)
	} else {
		imageData := prompt.Images
		if len(imageData) == 0 && len(prompt.Image) > 0 {
			imageData = [][]byte{prompt.Image}
		}
		images, decodeErr := decodeMultimodalImages(imageData)
		if decodeErr != nil {
			return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: %w", decodeErr)
		}
		if prompt.MediaHistory {
			history := h.config.ImageProjector
			if history == nil {
				history = h.config.Qwen3VLProjector
			}
			if history == nil || !history.Capabilities().MultiImage {
				return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: selected projector does not support media history")
			}
			projected, err = history.BuildImagesPrompt(ctx, tokenizerAPI, images, prompt.MediaText, projector.PromptOptions{History: true})
		} else if len(images) > 1 {
			multi := h.config.ImageProjector
			if multi == nil || !multi.Capabilities().MultiImage {
				return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: selected projector does not support multiple images")
			}
			projected, err = multi.BuildImagesPrompt(ctx, tokenizerAPI, images, prompt.MediaText, projector.PromptOptions{Thinking: thinking})
		} else if h.config.ImageProjector != nil {
			projected, err = h.config.ImageProjector.BuildImagePrompt(
				ctx, tokenizerAPI, images[0], prompt.BeforeMedia, prompt.AfterMedia, thinking,
			)
		} else {
			projected, err = h.config.Qwen3VLProjector.BuildImagePrompt(
				ctx, tokenizerAPI, images[0], prompt.BeforeMedia, prompt.AfterMedia, thinking,
			)
		}
	}
	if err != nil {
		return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: project media: %w", err)
	}
	properties, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: model properties unavailable for projected prompt")
	}
	_, inputs, err := inference.CompileProjectedInputs(
		projected, properties.ModelProperties().EmbeddingLength,
	)
	if err != nil {
		return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: %w", err)
	}
	prompt.TokenIDs = projected.TokenIDs
	prompt.Image = nil
	prompt.Images = nil
	prompt.MediaText = nil
	prompt.Audio = nil
	prompt.MediaHistory = false
	prompt.Media = nil
	prompt.Video = nil
	return prompt, inputs, nil
}

func nativePromptHasMedia(prompt nativePrompt) bool {
	return len(prompt.Image) != 0 || len(prompt.Images) != 0 || len(prompt.Audio) != 0 ||
		len(prompt.Media) != 0 || len(prompt.Video) != 0
}

func validateVideoFrames(frames []image.Image) error {
	if len(frames) == 0 {
		return errors.New("server: video has no frames")
	}
	var totalPixels uint64
	for index, frame := range frames {
		if frame == nil {
			return fmt.Errorf("server: video frame %d is nil", index)
		}
		bounds := frame.Bounds()
		width, height := bounds.Dx(), bounds.Dy()
		if width <= 0 || height <= 0 || width > maxImageDimension || height > maxImageDimension {
			return fmt.Errorf("server: video frame %d dimensions exceed limit", index)
		}
		pixels := uint64(width) * uint64(height)
		if pixels > maxImagePixels {
			return fmt.Errorf("server: video frame %d pixel count exceeds limit", index)
		}
		totalPixels += pixels
		if totalPixels > maxRequestImagePixels {
			return errors.New("server: video frames exceed aggregate pixel limit")
		}
	}
	return nil
}

func (h *Handler) preparePrompt(
	ctx context.Context,
	prompt nativePrompt,
	tokenizeText bool,
) (preparedPrompt, error) {
	result := preparedPrompt{
		nativePrompt: prompt,
		Multimodal:   nativePromptHasMedia(prompt),
	}
	if result.Multimodal {
		projectedPrompt, projected, err := h.projectNativeMultimodalPrompt(ctx, prompt)
		if err != nil {
			return preparedPrompt{}, err
		}
		result.nativePrompt = projectedPrompt
		result.ProjectedInputs = &projected
		return result, nil
	}
	if !tokenizeText || prompt.TokenIDs != nil {
		return result, nil
	}
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		return preparedPrompt{}, errors.New("server: generator cannot tokenize prepared prompt")
	}
	tokens, err := tokenizerAPI.TokenizeText(prompt.Text, true, true)
	if err != nil {
		return preparedPrompt{}, err
	}
	if len(tokens) == 0 {
		return preparedPrompt{}, errors.New("server: prepared prompt produced no tokens")
	}
	result.TokenIDs = tokens
	return result, nil
}

func (h *Handler) parseNativePrompt(raw json.RawMessage) (nativePrompt, error) {
	if len(raw) == 0 {
		return nativePrompt{}, errors.New("prompt is required")
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nativePrompt{}, errors.New("prompt must not be empty")
		}
		return nativePrompt{Text: text, Response: text}, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nativePrompt{}, errors.New("prompt must be a string or token/string sequence")
	}
	if len(parts) == 0 {
		return nativePrompt{}, errors.New("prompt token/string sequence must not be empty")
	}
	allStrings := true
	for _, part := range parts {
		var partText string
		if json.Unmarshal(part, &partText) != nil {
			allStrings = false
			break
		}
	}
	if allStrings {
		return nativePrompt{}, errors.New("multiple string prompts are not supported")
	}
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		return nativePrompt{}, errors.New("generator cannot expand mixed prompt strings")
	}
	tokenIDs := make([]tokenizer.TokenID, 0, len(parts))
	for index, part := range parts {
		var token int64
		if err := json.Unmarshal(part, &token); err == nil {
			if token < 0 || token >= int64(api.SamplingVocabularySize()) {
				return nativePrompt{}, fmt.Errorf("prompt token %d is out of range", token)
			}
			tokenIDs = append(tokenIDs, tokenizer.TokenID(token))
		} else {
			var partText string
			if err := json.Unmarshal(part, &partText); err != nil {
				return nativePrompt{}, fmt.Errorf(
					"prompt element %d must be an integer token ID or string",
					index,
				)
			}
			expanded, err := api.TokenizeText(partText, index == 0, false)
			if err != nil {
				return nativePrompt{}, fmt.Errorf("tokenize prompt element %d: %w", index, err)
			}
			tokenIDs = append(tokenIDs, expanded...)
		}
	}
	if len(tokenIDs) == 0 {
		return nativePrompt{}, errors.New("prompt produced no tokens")
	}
	processed, err := api.Detokenize(tokenIDs, inference.RenderPrompt)
	if err != nil {
		return nativePrompt{}, fmt.Errorf("detokenize processed prompt: %w", err)
	}
	return nativePrompt{
		Text:     processed,
		TokenIDs: tokenIDs,
		Response: processed,
	}, nil
}
