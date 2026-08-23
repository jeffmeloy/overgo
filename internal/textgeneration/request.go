// Package textgeneration owns the shared bounded text-generation request.
package textgeneration

import "errors"

// Request describes bounded text generation input.
type Request struct {
	Text      string `json:"text"`
	MaxTokens int    `json:"max_tokens"`
}

// SessionKey: shared text-generation residency.
func (Request) SessionKey() (string, error) { return "text-generation", nil }

// Validate rejects incomplete text-generation requests.
func Validate(request Request) error {
	if request.Text == "" || request.MaxTokens <= 0 {
		return errors.New("text generation requires text and a positive token count")
	}
	return nil
}
