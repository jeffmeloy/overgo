// Package operatoraction defines interface-neutral blocked-operation recovery.
// Commands are argv vectors, never shell strings, so CLI and GUI transports
// execute the same declared action without reparsing prose.
package operatoraction

import (
	"errors"
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

func (block Block) Clone() Block {
	block.Evidence = slices.Clone(block.Evidence)
	block.Actions = cloneActions(block.Actions)
	return block
}

// RecoverableError is a failure that carries the recovery an operator may run.
//   - Recovery finds it anywhere in an error chain
//   - Error and Unwrap answer for Cause.
type RecoverableError struct {
	Cause error
	Block Block
}

func (failure *RecoverableError) Error() string { return failure.Cause.Error() }
func (failure *RecoverableError) Unwrap() error { return failure.Cause }

func Recovery(err error) (Block, bool) {
	failure, ok := errors.AsType[*RecoverableError](err)
	if !ok || failure == nil {
		return Block{}, false
	}
	return failure.Block.Clone(), true
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
