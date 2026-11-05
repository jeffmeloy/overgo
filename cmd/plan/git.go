package main

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"

	"overgo/internal/gitauthority"
)

// commandInput runs name with args in root, input the bytes it reads, and returns what it wrote.
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
