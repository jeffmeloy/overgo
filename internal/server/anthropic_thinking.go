package server

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"

	"llamacpp2go/internal/strictjson"
)

const anthropicThinkingSignaturePrefix = "local_v1."

type anthropicThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
}

type anthropicThinkingSigner struct {
	key [32]byte
}

func newAnthropicThinkingSigner() (*anthropicThinkingSigner, error) {
	signer := &anthropicThinkingSigner{}
	if _, err := rand.Read(signer.key[:]); err != nil {
		return nil, fmt.Errorf("initialize Anthropic thinking signer: %w", err)
	}
	return signer, nil
}

func (signer *anthropicThinkingSigner) sign(thinking string) string {
	mac := hmac.New(sha256.New, signer.key[:])
	_, _ = mac.Write([]byte(thinking))
	return anthropicThinkingSignaturePrefix + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func (signer *anthropicThinkingSigner) verify(thinking, signature string) bool {
	if signer == nil || thinking == "" || signature == "" {
		return false
	}
	expected := signer.sign(thinking)
	return hmac.Equal([]byte(expected), []byte(signature))
}

func validateAnthropicThinking(
	raw json.RawMessage,
	maxTokens *int,
) (bool, error) {
	if !rawJSONConfigured(raw) {
		return false, nil
	}
	var config anthropicThinkingConfig
	if err := strictjson.DecodeBytes(raw, &config); err != nil {
		return false, errors.New("thinking must be an Anthropic thinking configuration")
	}
	switch config.Type {
	case "disabled":
		if config.BudgetTokens != 0 || config.Display != "" {
			return false, errors.New("disabled thinking cannot set budget_tokens or display")
		}
		return false, nil
	case "enabled":
		if config.BudgetTokens < 1024 {
			return false, errors.New("thinking budget_tokens must be at least 1024")
		}
		if maxTokens != nil && config.BudgetTokens >= *maxTokens {
			return false, errors.New("thinking budget_tokens must be less than max_tokens")
		}
		if config.Display != "" && config.Display != "summarized" {
			return false, errors.New("local thinking supports only summarized display")
		}
		return true, nil
	case "adaptive":
		return false, errors.New("adaptive thinking is unavailable for local GGUF models; use type enabled")
	default:
		return false, fmt.Errorf("unsupported thinking type %q", config.Type)
	}
}
