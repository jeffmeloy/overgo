package evaluation

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

type textGenerator string

func (g textGenerator) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, string(g), nil
}

func triangleCase() FunctionCallCase {
	return FunctionCallCase{
		Name: "triangle", Prompt: "Find the area. <tools> []", MaxTokens: 64, Function: "calculate_triangle_area",
		Arguments: map[string][]json.RawMessage{
			"base": {json.RawMessage(`10`)}, "height": {json.RawMessage(`5`)}, "unit": {json.RawMessage(`"units"`), json.RawMessage(`""`)},
		},
	}
}

// TestScoreFunctionCallAcceptsWhatTheLeaderboardAccepts holds the scorer to
// the call a case accepts and to each way a call can differ from it.
func TestScoreFunctionCallAcceptsWhatTheLeaderboardAccepts(t *testing.T) {
	t.Parallel()
	for name, test := range map[string]struct {
		output   string
		accepted bool
	}{
		"the call":                  {`[{"name":"calculate_triangle_area","arguments":{"base":10,"height":5,"unit":"units"}}]`, true},
		"the call with its marker":  {`<tool_call>[{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}]`, true},
		"a number written as float": {`[{"name":"calculate_triangle_area","arguments":{"base":10.0,"height":5}}]`, true},
		"a string by its standard":  {`[{"name":"calculate_triangle_area","arguments":{"base":10,"height":5,"unit":"Units."}}]`, true},
		"a qualified name":          {`[{"name":"geometry.calculate_triangle_area","arguments":{"base":10,"height":5}}]`, true},
		"another function":          {`[{"name":"calculate_area","arguments":{"base":10,"height":5}}]`, false},
		"a wrong value":             {`[{"name":"calculate_triangle_area","arguments":{"base":11,"height":5}}]`, false},
		"a string for a number":     {`[{"name":"calculate_triangle_area","arguments":{"base":"10","height":5}}]`, false},
		"a missing parameter":       {`[{"name":"calculate_triangle_area","arguments":{"base":10}}]`, false},
		"an unknown parameter":      {`[{"name":"calculate_triangle_area","arguments":{"base":10,"height":5,"depth":1}}]`, false},
		"a call repeated":           {`[{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}},{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}]`, true},
		"a wrong call first":        {`[{"name":"calculate_triangle_area","arguments":{"base":11,"height":5}},{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}]`, false},
		"an empty list":             {`<tool_call> []`, false},
		"no list":                   {`calculate_triangle_area(10, 5)`, false},
	} {
		if got := scoreFunctionCall(test.output, triangleCase()) == ""; got != test.accepted {
			t.Errorf("%s: accepted = %t, want %t", name, got, test.accepted)
		}
	}
}

// TestFunctionCallSuiteScoresAGeneratorAndRefusesWhatItCannotCompile runs a
// suite through a generator that always writes one call, and holds the
// compiler to a case that names no function or accepts no value.
func TestFunctionCallSuiteScoresAGeneratorAndRefusesWhatItCannotCompile(t *testing.T) {
	t.Parallel()
	wrong := triangleCase()
	wrong.Name = "wrong"
	wrong.Arguments = map[string][]json.RawMessage{"base": {json.RawMessage(`99`)}, "height": {json.RawMessage(`5`)}}
	suite := FunctionCallSuite{Kind: functionCallKind, Schema: "fixture/v1", Source: "fixture", Cases: []FunctionCallCase{triangleCase(), wrong}}
	compiled, err := compileFunctionCall(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindFunctionCall(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := evaluateFunctionCall(t.Context(), store, textGenerator(`[{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}]`), compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Accuracy != 0.5 || !report.Observations[0].Accepted || report.Observations[1].Accepted || report.Observations[1].Why == "" {
		t.Fatalf("report = %+v, want the first case accepted and the second refused with its reason", report)
	}
	for name, change := range map[string]func(*FunctionCallSuite){
		"no function":     func(s *FunctionCallSuite) { s.Cases[0].Function = "" },
		"no value":        func(s *FunctionCallSuite) { s.Cases[0].Arguments = map[string][]json.RawMessage{"base": nil} },
		"a repeated case": func(s *FunctionCallSuite) { s.Cases[1].Name = s.Cases[0].Name },
		"no cases":        func(s *FunctionCallSuite) { s.Cases = nil },
	} {
		broken := FunctionCallSuite{Kind: suite.Kind, Schema: suite.Schema, Source: suite.Source, Cases: append([]FunctionCallCase(nil), suite.Cases...)}
		broken.Cases[0].Arguments = suite.Cases[0].Arguments
		change(&broken)
		if _, err := compileFunctionCall(broken); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}
}

// TestBFCLFunctionCallsReadTheLeaderboardsFiles holds the builder to the form
// a function-calling model reads: the request, then the one tool with each
// parameter's type, and the accepted values as the answer file gives them.
func TestBFCLFunctionCallsReadTheLeaderboardsFiles(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	questions := filepath.Join(directory, "questions.json")
	answers := filepath.Join(directory, "answers.json")
	if err := os.WriteFile(questions, []byte(`{"id":"simple_0","question":[[{"role":"user","content":"Find the area of a triangle."}]],"function":[{"name":"calculate_triangle_area","parameters":{"type":"dict","properties":{"base":{"type":"integer"},"unit":{"type":"string"}}}}]}
{"id":"simple_1","question":[[{"role":"user","content":"Second."}]],"function":[{"name":"second","parameters":{"type":"dict","properties":{}}}]}
`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(answers, []byte(`{"id":"simple_0","ground_truth":[{"calculate_triangle_area":{"base":[10],"unit":["units",""]}}]}
{"id":"simple_1","ground_truth":[{"second":{}}]}
`), 0o600); err != nil {
		t.Fatal(err)
	}
	suite, err := BuildBFCLFunctionCalls(FunctionCallSpec{Questions: questions, Answers: answers})
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Cases) != 2 || suite.Cases[0].Name != "bfcl-v3-simple/simple_0" || suite.Cases[0].Function != "calculate_triangle_area" ||
		suite.Cases[0].Prompt != `Find the area of a triangle. <tools> [{"name":"calculate_triangle_area","parameters":{"base":"integer","unit":"string"}}]` ||
		len(suite.Cases[0].Arguments["unit"]) != 2 {
		t.Fatalf("suite = %+v", suite)
	}
	if _, err := compileFunctionCall(suite); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	if err := os.WriteFile(answers, []byte(`{"id":"simple_0","ground_truth":[{"another":{}}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildBFCLFunctionCalls(FunctionCallSpec{Questions: questions, Answers: answers}); err == nil {
		t.Fatal("a suite was built from answers that name another function or are missing")
	}
}
