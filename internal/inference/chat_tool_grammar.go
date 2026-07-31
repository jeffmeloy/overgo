package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"llamacpp2go/internal/sampling"
)

var chatToolNamePattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// ChatToolGrammar returns a lazy GBNF grammar that activates when the model
// begins the tool-call delimiter advertised by its GGUF template.
func (r *Runner) ChatToolGrammar(
	tools []ChatTool,
	required bool,
	enableThinking bool,
) (source, root string, triggerPatterns []string, err error) {
	if r == nil {
		return "", "", nil, errors.New("inference: runner is nil")
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
		source, err = hermesToolGrammar(tools)
	} else if strings.Contains(template, toolCallOpen) {
		source, err = jsonToolGrammar(tools)
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

func jsonToolGrammar(tools []ChatTool) (string, error) {
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
	return `tool-call ::= "<tool_call>" space root space "</tool_call>"` + "\n" +
		`tool-root ::= tool-call (space tool-call)*` + "\n" +
		jsonGrammar, nil
}

func hermesToolGrammar(tools []ChatTool) (string, error) {
	rules := make([]string, 0, 2+len(tools)*2)
	functionRules := make([]string, len(tools))
	for index, tool := range tools {
		if err := validateGrammarTool(tool); err != nil {
			return "", fmt.Errorf("tool %d: %w", index, err)
		}
		functionName := fmt.Sprintf("tool-function-%d", index)
		functionRules[index] = functionName
		properties, _ := tool.Function.Parameters["properties"].(map[string]any)
		propertyNames := make([]string, 0, len(properties))
		for name := range properties {
			if !chatToolNamePattern.MatchString(name) {
				return "", fmt.Errorf(
					"tool %d parameter name %q is not tag-safe",
					index,
					name,
				)
			}
			propertyNames = append(propertyNames, name)
		}
		sort.Strings(propertyNames)
		parameterRules := make([]string, len(propertyNames))
		for propertyIndex, name := range propertyNames {
			ruleName := fmt.Sprintf(
				"tool-%d-parameter-%d",
				index,
				propertyIndex,
			)
			parameterRules[propertyIndex] = ruleName
			rules = append(
				rules,
				ruleName+" ::= "+
					strconv.Quote("<parameter="+name+">")+
					` tool-parameter-value `+
					strconv.Quote("</parameter>")+" space",
			)
		}
		body := ""
		if len(parameterRules) != 0 {
			body = " (" + strings.Join(parameterRules, " | ") + ")*"
		}
		rules = append(
			rules,
			functionName+" ::= "+
				strconv.Quote("<function="+tool.Function.Name+">")+
				" space"+body+" "+
				strconv.Quote("</function>"),
		)
	}
	rules = append(
		rules,
		`space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		`tool-parameter-value ::= [^<\x00]*`,
		`tool-call ::= "<tool_call>" space (`+
			strings.Join(functionRules, " | ")+
			`) space "</tool_call>"`,
		`tool-root ::= tool-call (space tool-call)*`,
	)
	sort.Strings(rules)
	return strings.Join(rules, "\n"), nil
}

func validateGrammarTool(tool ChatTool) error {
	if err := validateChatTool(tool); err != nil {
		return err
	}
	if !chatToolNamePattern.MatchString(tool.Function.Name) {
		return fmt.Errorf(
			"function name %q is not tag-safe",
			tool.Function.Name,
		)
	}
	return nil
}
