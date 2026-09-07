package vqaserve

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

//go:embed serve_policy.json
var servePolicyJSON []byte

// ServePolicy is the serving declaration: the decode budget a request that
// names none runs under, the budget the activation evidence is produced
// with, so the harness and the page decode under one declared bound.
type ServePolicy struct {
	Version        int `json:"version"`
	DecodeMaxSteps int `json:"decode_max_steps"`
}

// Policy is the embedded serving declaration.
var Policy = func() ServePolicy {
	var policy ServePolicy
	if err := strictjson.DecodeBytes(servePolicyJSON, &policy); err != nil {
		panic(fmt.Errorf("vqaserve: serve policy: %w", err))
	}
	if policy.DecodeMaxSteps <= 0 {
		panic("vqaserve: serve policy declares no decode budget")
	}
	return policy
}()

// Request is the page form of a question about an image: the stored image
// artifact, the question, and the decode budget, the policy's when zero.
type Request struct {
	Image     artifact.ID `json:"image" label:"image" media:"image"`
	Question  string      `json:"question"`
	MaxTokens int         `json:"max_tokens,omitzero"`
}

// RequestContract identifies a request document, so a run keys on the
// exact request it answered.
var RequestContract = artifact.JSONContract(artifact.KindFile, "overgo.vqa-input.v1")

// ValidateRequest refuses a request without a stored image, without a
// question, or with a negative budget.
func ValidateRequest(request Request) error {
	switch {
	case !request.Image.Valid():
		return errors.New("vqa: image must name a stored artifact")
	case strings.TrimSpace(request.Question) == "":
		return errors.New("vqa: question is required")
	case request.MaxTokens < 0:
		return fmt.Errorf("vqa: max_tokens must not be negative, got %d", request.MaxTokens)
	}
	return nil
}

// Budget is the decode budget the request runs under.
func (request Request) Budget() int {
	if request.MaxTokens > 0 {
		return request.MaxTokens
	}
	return Policy.DecodeMaxSteps
}

// ReadImage reads the request's image document from the store, refusing
// an absent artifact or one whose media type is not an image's.
func ReadImage(ctx context.Context, reader artifact.Reader, id artifact.ID) (artifact.Content, error) {
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return artifact.Content{}, err
	}
	if !found {
		return artifact.Content{}, fmt.Errorf("vqa: image %s is absent from the store", id)
	}
	if !strings.HasPrefix(content.Descriptor.MediaType, "image/") {
		return artifact.Content{}, fmt.Errorf("vqa: image %s is %s, not an image", id, content.Descriptor.MediaType)
	}
	return content, nil
}
