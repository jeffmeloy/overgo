package evaluation

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
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
	Function []struct {
		Name       string `json:"name"`
		Parameters struct {
			Properties map[string]struct {
				Type string `json:"type"`
			} `json:"properties"`
		} `json:"parameters"`
	} `json:"function"`
}

type bfclAnswer struct {
	ID          string                                    `json:"id"`
	GroundTruth []map[string]map[string][]json.RawMessage `json:"ground_truth"`
}

// BuildBFCLFunctionCalls builds the function-call suite of the leaderboard's
// simple category: each case gives the model a request and the one function it
// may call, in the form a function-calling model reads, the request then the
// tools, a tool being a name and each parameter's type, and accepts the call
// the leaderboard accepts.
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
		parameters := make(map[string]string, len(function.Parameters.Properties))
		for name, property := range function.Parameters.Properties {
			parameters[name] = property.Type
		}
		tools, err := json.Marshal([]any{map[string]any{"name": function.Name, "parameters": parameters}})
		if err != nil {
			return FunctionCallSuite{}, err
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
