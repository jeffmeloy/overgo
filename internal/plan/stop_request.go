package plan

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/atomicfile"
	"overgo/internal/processlock"
)

// stopRequestFile, in a worktree's process state, holds the words of the latest prompt there when it asked to stop.
// The prompt hook writes it for such a prompt and removes it for any other, so a requested stop rests on the latest.
const stopRequestFile = "stop_request"

// stopWords are the words a prompt asks the loop to stop with. The match is crude and deterministic: a prompt that
// says "don't stop" counts too, and the session still decides.
var stopWords = []string{"halt", "pause", "stop"}

// asksToStop says a prompt asks the loop to stop: one of its words, read without case or punctuation, is a stop word.
func asksToStop(prompt string) bool {
	words := strings.FieldsFunc(strings.ToLower(prompt), func(r rune) bool { return !unicode.IsLetter(r) })
	return slices.ContainsFunc(words, func(word string) bool { return slices.Contains(stopWords, word) })
}

// NoteStopRequest keeps prompt in root's process state when it asks to stop, and clears what an earlier prompt left
// when it does not.
func NoteStopRequest(root, prompt string) error {
	path := filepath.Join(root, processlock.StateDirectory, stopRequestFile)
	if !asksToStop(prompt) {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	return atomicfile.Write(path, []byte(prompt), stopFileMode)
}

// RequireStopRequest refuses a requested stop unless the latest prompt in root asked to stop: a session records one
// only when asked, never of its own accord.
func RequireStopRequest(root string) error {
	if _, err := os.Stat(filepath.Join(root, processlock.StateDirectory, stopRequestFile)); err != nil {
		return errors.New("plan: a requested stop needs a prompt that asked to stop (with stop, pause or halt), and the latest prompt here asked for none; an outside need is licence: or hardware:, and a choice between options is no stop")
	}
	return nil
}
