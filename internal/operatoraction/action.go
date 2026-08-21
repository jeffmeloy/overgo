// Package operatoraction defines interface-neutral blocked-operation recovery.
// Commands are argv vectors, never shell strings, so CLI and GUI transports
// execute the same declared action without reparsing prose.
package operatoraction

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	maxTextBytes = 4 << 10
	maxActions   = 16
	maxArguments = 64
)

type Action struct {
	Code    string   `json:"code"`
	Summary string   `json:"summary"`
	Argv    []string `json:"argv"`
}

type Block struct {
	Subject  artifact.ID   `json:"subject"`
	Reason   string        `json:"reason"`
	Evidence []artifact.ID `json:"evidence"`
	Actions  []Action      `json:"actions"`
}

type Answer string

const (
	AnswerGrant   Answer = "grant"
	AnswerDecline Answer = "decline"
)

type Choice struct {
	Answer Answer `json:"answer"`
	Label  string `json:"label"`
	Effect string `json:"effect"`
	Action Action `json:"action"`
}

type Consent struct {
	Subject  artifact.ID   `json:"subject"`
	Headline string        `json:"headline"`
	Reason   string        `json:"reason"`
	Value    string        `json:"value"`
	Evidence []artifact.ID `json:"evidence"`
	Choices  []Choice      `json:"choices"`
	OffPath  Action        `json:"off_path"`
}

func (action Action) Validate() error {
	if !bounded(action.Code) || strings.ContainsAny(action.Code, " \t") || !bounded(action.Summary) ||
		len(action.Argv) == 0 || len(action.Argv) > maxArguments {
		return errors.New("operator action: invalid action identity or command")
	}
	for _, argument := range action.Argv {
		if argument == "" || len(argument) > maxTextBytes || strings.ContainsAny(argument, "\x00\r\n") {
			return errors.New("operator action: invalid argv token")
		}
	}
	return nil
}

func (block Block) Validate() error {
	if !block.Subject.Valid() || !bounded(block.Reason) || block.Evidence == nil ||
		len(block.Actions) == 0 || len(block.Actions) > maxActions {
		return errors.New("operator action: incomplete block")
	}
	seenEvidence := map[artifact.ID]bool{}
	for _, evidence := range block.Evidence {
		if !evidence.Valid() || seenEvidence[evidence] {
			return errors.New("operator action: invalid block evidence")
		}
		seenEvidence[evidence] = true
	}
	seenActions := map[string]bool{}
	for _, action := range block.Actions {
		if err := action.Validate(); err != nil || seenActions[action.Code] {
			return errors.New("operator action: invalid or duplicate recovery action")
		}
		seenActions[action.Code] = true
	}
	return nil
}

func (consent Consent) Validate() error {
	if !consent.Subject.Valid() || !bounded(consent.Headline) || !bounded(consent.Reason) ||
		!bounded(consent.Value) || consent.Evidence == nil || len(consent.Choices) != 2 {
		return errors.New("operator action: incomplete consent envelope")
	}
	for _, evidence := range consent.Evidence {
		if !evidence.Valid() {
			return errors.New("operator action: invalid consent evidence")
		}
	}
	want := [...]Answer{AnswerGrant, AnswerDecline}
	for index, choice := range consent.Choices {
		if choice.Answer != want[index] || !bounded(choice.Label) || !bounded(choice.Effect) || choice.Action.Validate() != nil {
			return errors.New("operator action: invalid consent choice")
		}
	}
	if err := consent.OffPath.Validate(); err != nil {
		return errors.New("operator action: invalid consent off path")
	}
	return nil
}

func (block Block) Clone() Block {
	block.Evidence = slices.Clone(block.Evidence)
	block.Actions = cloneActions(block.Actions)
	return block
}

func (consent Consent) Clone() Consent {
	consent.Evidence = slices.Clone(consent.Evidence)
	consent.Choices = slices.Clone(consent.Choices)
	for index := range consent.Choices {
		consent.Choices[index].Action.Argv = slices.Clone(consent.Choices[index].Action.Argv)
	}
	consent.OffPath.Argv = slices.Clone(consent.OffPath.Argv)
	return consent
}

type recoverableError struct {
	cause error
	block Block
}

func (failure *recoverableError) Error() string { return failure.cause.Error() }
func (failure *recoverableError) Unwrap() error { return failure.cause }

// Recoverable returns a typed error only when the cause and recovery envelope
// are complete. Invalid recovery metadata fails as an ordinary construction
// error and can never be advertised to an operator.
func Recoverable(cause error, block Block) error {
	if cause == nil {
		return errors.New("operator action: recoverable failure lacks a cause")
	}
	if err := block.Validate(); err != nil {
		return fmt.Errorf("operator action: recoverable failure: %w", err)
	}
	return &recoverableError{cause: cause, block: block.Clone()}
}

func Recovery(err error) (Block, bool) {
	var failure *recoverableError
	if !errors.As(err, &failure) || failure == nil {
		return Block{}, false
	}
	return failure.block.Clone(), true
}

func cloneActions(actions []Action) []Action {
	cloned := slices.Clone(actions)
	for index := range cloned {
		cloned[index].Argv = slices.Clone(cloned[index].Argv)
	}
	return cloned
}

func bounded(value string) bool {
	return value != "" && len(value) <= maxTextBytes && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n")
}
