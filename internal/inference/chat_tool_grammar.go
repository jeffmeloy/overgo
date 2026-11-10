package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/sampling"
	"overgo/internal/textcheck"
)

// chatToolNameCharacters are what a tool's name may hold.
const chatToolNameCharacters = textcheck.Alphanumeric + "_-"

// parameterClosing is what ends a parameter-form value, the only text a value cannot hold: the parser reads a value to
// its first closing tag (parseHermesParameters).
const parameterClosing = "</parameter>"

// parameterLessThan is the rule for a less-than sign within a value and what follows it, until the text parts from
// the closing tag: after each prefix of the tag, any byte but the tag's next one, or another less-than sign, which
// starts the rule again. A value may hold any text but the tag (loop-proof/tool-value-markup): one whose rule admitted
// no less-than sign at all left the local worker unable to write a Go comparison, a channel receive or <item>.
func parameterLessThan() string {
	const again = "tool-parameter-lt"
	// The tag's last byte would close it, so after the rest of it only another byte follows.
	last := len(parameterClosing) - 1
	tail := fmt.Sprintf(`[^%c<\x00] | %s`, parameterClosing[last], again)
	for index := last - 1; index >= 1; index-- {
		next := parameterClosing[index]
		tail = fmt.Sprintf(`[^%c<\x00] | %s | %s ( %s )`, next, again, strconv.Quote(string(next)), tail)
	}
	return again + ` ::= "<" ( ` + tail + ` )`
}

// ChatToolGrammar: returns lazy GBNF grammar that activates when model
// begins tool-call delimiter advertised by its GGUF template
func (r *Runner) ChatToolGrammar(
	tools []ChatTool,
	required bool,
	enableThinking bool,
	parallelToolCalls bool,
) (source, root string, triggerPatterns []string, err error) {
	if r == nil {
		return "", "", nil, errRunnerNil
	}
	if len(tools) == 0 {
		return "", "", nil, errors.New("inference: tool list is empty")
	}
	template := metadataString(r.file, "tokenizer.chat_template")
	if toolSource := metadataString(
		r.file,
		"tokenizer.chat_template.tool_use",
	); toolSource != "" {
		template = toolSource
	}
	hermes := strings.Contains(template, "<function=example_function_name>")
	if hermes {
		source, err = hermesToolGrammar(tools, parallelToolCalls)
	} else if strings.Contains(template, toolCallOpen) {
		source, err = jsonToolGrammar(tools, parallelToolCalls)
	} else {
		err = errors.New(
			"inference: GGUF chat template has no supported tool-call syntax",
		)
	}
	if err != nil {
		return "", "", nil, err
	}
	if required && enableThinking {
		if hermes {
			source += "\n" +
				`tool-generation-root ::= tool-reasoning "</think>" space tool-root` +
				"\n" + `tool-reasoning ::= [^<\x00]*`
		} else {
			source += "\n" +
				`tool-generation-root ::= ("<think>" tool-reasoning "</think>" space)? tool-root` +
				"\n" + `tool-reasoning ::= [^<\x00]*`
		}
		return source, "tool-generation-root", nil, nil
	}
	if required {
		return source, "tool-root", nil, nil
	}
	return source, "tool-root", []string{`(<tool_call>)`}, nil
}

func jsonToolGrammar(tools []ChatTool, parallelToolCalls bool) (string, error) {
	type nameSchema struct {
		Const string `json:"const"`
	}
	type propertiesSchema struct {
		Name      nameSchema      `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	type callSchema struct {
		Type                 string           `json:"type"`
		Properties           propertiesSchema `json:"properties"`
		Required             []string         `json:"required"`
		AdditionalProperties bool             `json:"additionalProperties"`
	}
	alternatives := make([]callSchema, len(tools))
	for index, tool := range tools {
		if err := validateGrammarTool(tool); err != nil {
			return "", fmt.Errorf("tool %d: %w", index, err)
		}
		parameters, err := json.Marshal(tool.Function.Parameters)
		if err != nil {
			return "", fmt.Errorf("tool %d parameters: %w", index, err)
		}
		alternatives[index] = callSchema{
			Type: "object",
			Properties: propertiesSchema{
				Name:      nameSchema{Const: tool.Function.Name},
				Arguments: parameters,
			},
			Required:             []string{"name", "arguments"},
			AdditionalProperties: false,
		}
	}
	var schema any = alternatives[0]
	if len(alternatives) > 1 {
		schema = struct {
			OneOf []callSchema `json:"oneOf"`
		}{OneOf: alternatives}
	}
	encoded, err := json.Marshal(schema)
	if err != nil {
		return "", err
	}
	jsonGrammar, err := sampling.JSONSchemaToGrammar(encoded)
	if err != nil {
		return "", fmt.Errorf("tool JSON schema: %w", err)
	}
	toolRoot := `tool-root ::= tool-call`
	if parallelToolCalls {
		toolRoot += ` (space tool-call)*`
	}
	return `tool-call ::= "<tool_call>" space root space "</tool_call>"` + "\n" +
		toolRoot + "\n" +
		jsonGrammar, nil
}

func hermesToolGrammar(
	tools []ChatTool,
	parallelToolCalls bool,
) (string, error) {
	rules := make([]string, 0, 2+len(tools)*2)
	functionRules := make([]string, len(tools))
	for index, tool := range tools {
		if err := validateGrammarTool(tool); err != nil {
			return "", fmt.Errorf("tool %d: %w", index, err)
		}
		functionName := fmt.Sprintf("tool-function-%d", index)
		functionRules[index] = functionName
		// Each parameter appears once: the required ones in the order the schema requires them, then each optional one,
		// sorted, or not at all. The parser refuses a call giving a parameter twice (parseHermesParameters) or missing a
		// required one (validateParsedToolArguments), so the grammar draws no call it would refuse.
		properties, _ := tool.Function.Parameters["properties"].(map[string]any)
		required := schemaRequiredNames(tool.Function.Parameters["required"])
		var optional []string
		for name := range properties {
			if !slices.Contains(required, name) {
				optional = append(optional, name)
			}
		}
		slices.Sort(optional)
		var sequence []string
		for propertyIndex, name := range append(slices.Clone(required), optional...) {
			if !textcheck.Only(name, chatToolNameCharacters) {
				return "", fmt.Errorf(
					"tool %d parameter name %q is not tag-safe",
					index,
					name,
				)
			}
			ruleName := fmt.Sprintf(
				"tool-%d-parameter-%d",
				index,
				propertyIndex,
			)
			rules = append(
				rules,
				ruleName+" ::= "+
					strconv.Quote("<parameter="+name+">")+
					` tool-parameter-value `+
					strconv.Quote("</parameter>")+" space",
			)
			if propertyIndex >= len(required) {
				ruleName += "?"
			}
			sequence = append(sequence, ruleName)
		}
		body := ""
		if len(sequence) != 0 {
			body = " " + strings.Join(sequence, " ")
		}
		rules = append(
			rules,
			functionName+" ::= "+
				strconv.Quote("<function="+tool.Function.Name+">")+
				" space"+body+" "+
				strconv.Quote("</function>"),
		)
	}
	toolRoot := `tool-root ::= tool-call`
	if parallelToolCalls {
		toolRoot += ` (space tool-call)*`
	}
	rules = append(
		rules,
		`space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		`tool-parameter-value ::= ( [^<\x00] | tool-parameter-lt )*`,
		parameterLessThan(),
		`tool-call ::= "<tool_call>" space (`+
			strings.Join(functionRules, " | ")+
			`) space "</tool_call>"`,
		toolRoot,
	)
	slices.Sort(rules)
	return strings.Join(rules, "\n"), nil
}

func validateGrammarTool(tool ChatTool) error {
	if err := validateChatTool(tool); err != nil {
		return err
	}
	if !textcheck.Only(tool.Function.Name, chatToolNameCharacters) {
		return fmt.Errorf(
			"function name %q is not tag-safe",
			tool.Function.Name,
		)
	}
	return nil
}
