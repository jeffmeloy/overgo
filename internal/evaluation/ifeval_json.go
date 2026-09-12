package evaluation

import (
	"encoding/json"
	"strings"
	"unicode"
)

// lm_eval 0.4.9.1 JsonFormat; separate identity from the strict JSON rule.
const ifevalJSONRule = "lm-eval/ifeval/json-format/v1"

func matchesIFEvalJSON(response string) bool {
	// Python str.strip includes the four ASCII information separators.
	strip := func(value string) string {
		return strings.TrimFunc(value, func(r rune) bool {
			return unicode.IsSpace(r) || r >= '\x1c' && r <= '\x1f'
		})
	}
	value := strip(response)
	for _, prefix := range []string{"```json", "```Json", "```JSON", "```"} {
		value = strings.TrimPrefix(value, prefix)
	}
	value = strip(strings.TrimSuffix(value, "```"))
	// Python json.loads also accepts these three constants. Replace only
	// unquoted literals; encoding/json still validates the complete grammar.
	var normalized strings.Builder
	quoted, escaped := false, false
	for index := 0; index < len(value); {
		current := value[index]
		if quoted {
			normalized.WriteByte(current)
			if escaped {
				escaped = false
			} else if current == '\\' {
				escaped = true
			} else if current == '"' {
				quoted = false
			}
			index++
			continue
		}
		constant := ""
		for _, candidate := range []string{"NaN", "Infinity", "-Infinity"} {
			if strings.HasPrefix(value[index:], candidate) {
				constant = candidate
				break
			}
		}
		if constant != "" {
			normalized.WriteString("null")
			index += len(constant)
		} else {
			normalized.WriteByte(current)
			quoted = current == '"'
			index++
		}
	}
	return json.Valid([]byte(normalized.String()))
}
