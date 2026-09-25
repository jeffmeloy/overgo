package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	maxPromptSuiteBytes   = 1 << 20
	maxPromptSuitePrompts = 1024
)

func benchmarkPrompts(options options) ([]string, error) {
	if options.PromptSuite == "" {
		if options.Prompt == "" {
			return nil, errors.New("benchmark: prompt must not be empty")
		}
		return []string{options.Prompt}, nil
	}
	file, err := os.Open(options.PromptSuite)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxPromptSuiteBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxPromptSuiteBytes {
		return nil, errors.New("benchmark: prompt suite exceeds byte limit")
	}
	var prompts []string
	if err := json.Unmarshal(data, &prompts); err != nil {
		return nil, fmt.Errorf("benchmark: parse prompt suite: %w", err)
	}
	if len(prompts) == 0 || len(prompts) > maxPromptSuitePrompts {
		return nil, errors.New("benchmark: prompt suite count is out of bounds")
	}
	for index, prompt := range prompts {
		if strings.TrimSpace(prompt) == "" {
			return nil, fmt.Errorf("benchmark: prompt %d is empty", index)
		}
	}
	return prompts, nil
}

func benchmarkPromptIndex(run, count int) int {
	if run < 0 {
		return 0
	}
	return run % count
}
