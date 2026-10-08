package evaluation

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
)

const (
	bfclSource = "bfcl-v3/simple"
	bfclSchema = "overgo/bfcl-v3-simple-function-calls/v1"
	// bfclAnswerTokens bounds the call a model writes for one case.
	bfclAnswerTokens = 256
)

// FunctionCallSpec names the two files of the Berkeley Function-Calling
// Leaderboard's simple category: its questions with their function
// definitions, and the answers each parameter accepts.
type FunctionCallSpec struct {
	Questions string
	Answers   string
}

type bfclQuestion struct {
	ID       string `json:"id"`
	Question [][]struct {
		Content string `json:"content"`
	} `json:"question"`
	Function []bfclFunction `json:"function"`
}

// bfclFunction is one function as the leaderboard defines it. Its properties
// are kept as written, so that a tool lists its parameters in the order the
// definition gives them.
type bfclFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  struct {
		Properties json.RawMessage `json:"properties"`
		Required   []string        `json:"required"`
	} `json:"parameters"`
}

// describedTools writes the function as the one tool of a tools list, the way
// a function-calling model's own examples and training data describe a tool:
// its name, its description, and for each parameter its type, its description
// and whether it is required. A model trained on tools so described reads the
// descriptions to decide what to pass; given a name and types alone, needle's
// own code answers 8 of the leaderboard's 400 simple cases acceptably, and
// given this, 104.
func describedTools(function bfclFunction) (string, error) {
	// quoted writes a text as a JSON string, its angle brackets and ampersands
	// as they are.
	quoted := func(text string) string {
		var written strings.Builder
		encoder := json.NewEncoder(&written)
		encoder.SetEscapeHTML(false)
		// A string always encodes; the encoder ends it with a line break.
		_ = encoder.Encode(text)
		return strings.TrimSuffix(written.String(), "\n")
	}
	var tools strings.Builder
	fmt.Fprintf(&tools, `[{"name":%s,"description":%s,"parameters":{`, quoted(function.Name), quoted(function.Description))
	if len(function.Parameters.Properties) != 0 {
		properties := json.NewDecoder(strings.NewReader(string(function.Parameters.Properties)))
		if _, err := properties.Token(); err != nil {
			return "", err
		}
		for first := true; properties.More(); first = false {
			name, err := properties.Token()
			if err != nil {
				return "", err
			}
			var property struct {
				Type        string `json:"type"`
				Description string `json:"description"`
			}
			if err := properties.Decode(&property); err != nil {
				return "", err
			}
			parameter, named := name.(string)
			if !named {
				return "", fmt.Errorf("evaluation: function %s lists a parameter that has no name", function.Name)
			}
			if !first {
				tools.WriteByte(',')
			}
			fmt.Fprintf(&tools, `%s:{"type":%s,"description":%s,"required":%t}`,
				quoted(parameter), quoted(property.Type), quoted(property.Description), slices.Contains(function.Parameters.Required, parameter))
		}
	}
	tools.WriteString(`}}]`)
	return tools.String(), nil
}

type bfclAnswer struct {
	ID          string                                    `json:"id"`
	GroundTruth []map[string]map[string][]json.RawMessage `json:"ground_truth"`
}

// BuildBFCLFunctionCalls builds the function-call suite of the leaderboard's
// simple category: each case gives the model a request and the one function it
// may call, in the form a function-calling model reads, the request then the
// tools, the tool described as describedTools describes it, and accepts the
// call the leaderboard accepts.
func BuildBFCLFunctionCalls(spec FunctionCallSpec) (FunctionCallSuite, error) {
	var questions []bfclQuestion
	if err := readJSONValues(spec.Questions, &questions); err != nil {
		return FunctionCallSuite{}, err
	}
	var answers []bfclAnswer
	if err := readJSONValues(spec.Answers, &answers); err != nil {
		return FunctionCallSuite{}, err
	}
	accepted := make(map[string]map[string]map[string][]json.RawMessage, len(answers))
	for _, answer := range answers {
		if len(answer.GroundTruth) != 1 {
			return FunctionCallSuite{}, fmt.Errorf("evaluation: case %s accepts %d calls, want 1", answer.ID, len(answer.GroundTruth))
		}
		accepted[answer.ID] = answer.GroundTruth[0]
	}
	suite := FunctionCallSuite{Kind: functionCallKind, Schema: bfclSchema, Source: bfclSource}
	for _, question := range questions {
		answer, answered := accepted[question.ID]
		if !answered || len(question.Function) != 1 || len(question.Question) != 1 || len(question.Question[0]) != 1 {
			return FunctionCallSuite{}, fmt.Errorf("evaluation: case %s is not one request of one function with one answer", question.ID)
		}
		function := question.Function[0]
		// The leaderboard names an answer by a method's own name, though the
		// function is defined by its qualified one.
		var written string
		var arguments map[string][]json.RawMessage
		for name, values := range answer {
			written, arguments = name, values
		}
		if len(answer) != 1 || function.Name != written && !strings.HasSuffix(function.Name, "."+written) {
			return FunctionCallSuite{}, fmt.Errorf("evaluation: case %s accepts a call of another function than %s", question.ID, function.Name)
		}
		tools, err := describedTools(function)
		if err != nil {
			return FunctionCallSuite{}, fmt.Errorf("evaluation: case %s: %w", question.ID, err)
		}
		suite.Cases = append(suite.Cases, FunctionCallCase{
			Name:      "bfcl-v3-simple/" + question.ID,
			Prompt:    fmt.Sprintf("%s <tools> %s", question.Question[0][0].Content, tools),
			MaxTokens: bfclAnswerTokens, Function: written, Arguments: arguments,
		})
	}
	if len(suite.Cases) == 0 {
		return FunctionCallSuite{}, errors.New("evaluation: the leaderboard file holds no case")
	}
	return suite, nil
}

// readJSONValues reads a file of JSON values, one per line or one after
// another, into the slice target points to.
func readJSONValues[T any](path string, target *[]T) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	for {
		var value T
		if err := decoder.Decode(&value); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return fmt.Errorf("evaluation: read %s: %w", path, err)
		}
		*target = append(*target, value)
	}
}
