package main

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"

	"overgo/internal/gitauthority"
)

func gitOutput(root string, args ...string) ([]byte, error) {
	return commandOutput(root, "git", args...)
}

func commandOutput(root, name string, args ...string) ([]byte, error) {
	return commandInput(root, nil, name, args...)
}

// commandInput is commandOutput with the bytes the command reads.
func commandInput(root string, input []byte, name string, args ...string) ([]byte, error) {
	commandArgs := args
	if name == "git" {
		commandArgs = append([]string{"--no-replace-objects"}, args...)
	}
	command := exec.Command(name, commandArgs...)
	command.Dir = root
	command.Stdin = bytes.NewReader(input)
	if name == "git" {
		command.Env = gitauthority.RepositoryEnvironment()
	}
	output, err := command.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(output)))
	}
	return output, nil
}
