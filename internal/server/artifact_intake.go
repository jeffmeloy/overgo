package server

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// attachmentSchema: stored form of a file the composer attached; media
// type = the file's own, identity = its bytes.
const attachmentSchema = "overgo/attachment/v1"

// attachmentIntake: the intake route's answer; an artifact-typed control
// takes ID.
type attachmentIntake struct {
	ID        string `json:"id"`
	MediaType string `json:"media_type"`
	Size      uint64 `json:"size"`
}

// artifactIntake stores an attached file of a media type the server
// decodes as a file document of that type (bounded by the media limits;
// images decoded against the image bounds) and answers its identity. The
// executor behind an artifact-typed control accepts or refuses the stored
// document by its type; the served chat model's projectors do not bound
// what a control can take.
func (h *Handler) artifactIntake(response http.ResponseWriter, request *http.Request) {
	if h.repository == nil {
		writeError(response, http.StatusServiceUnavailable, "repository_unavailable", "no artifact repository is configured")
		return
	}
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil {
		writeInvalidRequest(response, fmt.Errorf("attachment media type: %w", err))
		return
	}
	if !slices.Contains(h.acceptedMedia(true, true, true), mediaType) {
		writeError(response, http.StatusUnsupportedMediaType, "media_unsupported", "the server does not decode "+mediaType)
		return
	}
	image, limit := strings.HasPrefix(mediaType, "image/"), maxMediaBytes
	if image {
		limit = maxImageBytes
	}
	data, err := io.ReadAll(io.LimitReader(request.Body, int64(limit)+1))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if len(data) > limit {
		writeError(response, http.StatusRequestEntityTooLarge, "media_too_large", fmt.Sprintf("attachment exceeds the %d byte limit", limit))
		return
	}
	if len(data) == 0 {
		writeInvalidRequestMessage(response, "attachment is empty")
		return
	}
	if image {
		if err := validateMultimodalImages([][]byte{data}); err != nil {
			writeInvalidRequest(response, err)
			return
		}
	}
	content, err := artifact.DocumentContract{Kind: artifact.KindFile, MediaType: mediaType, Schema: attachmentSchema}.OwnedContentBytes(data)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	// Native transcription and other consumers may have already registered these
	// exact bytes. Preserve their immutable descriptor when attaching them again.
	if prior, found, err := h.repository.Artifact(request.Context(), content.Descriptor.ID); err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	} else if found {
		content.Descriptor = prior
	}
	if _, err := artifact.CommitBatch(request.Context(), h.repository, artifact.Batch{
		Key: "artifact-intake/" + content.Descriptor.ID.String(), Contents: []artifact.Content{content},
	}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, attachmentIntake{ID: content.Descriptor.ID.String(), MediaType: mediaType, Size: content.Descriptor.Size})
}
