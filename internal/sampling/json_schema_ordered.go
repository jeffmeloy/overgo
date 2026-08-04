package sampling

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
)

const (
	maxJSONSchemaBytes      = 1 << 20
	maxJSONSchemaDepth      = 256
	maxJSONSchemaArrayItems = 1 << 20
)

type orderedJSONMember struct {
	Name  string
	Value any
}

type orderedJSONObject struct {
	Members []orderedJSONMember
}

func (object orderedJSONObject) get(name string) (any, bool) {
	for _, member := range object.Members {
		if member.Name == name {
			return member.Value, true
		}
	}
	return nil, false
}

func (object orderedJSONObject) has(name string) bool {
	_, ok := object.get(name)
	return ok
}

func (object orderedJSONObject) keys() []string {
	result := make([]string, len(object.Members))
	for index, member := range object.Members {
		result[index] = member.Name
	}
	return result
}

func parseOrderedJSON(input []byte) (any, error) {
	input = bytes.TrimPrefix(input, []byte{0xef, 0xbb, 0xbf})
	if len(input) == 0 {
		return nil, errors.New("JSON schema is empty")
	}
	if len(input) > maxJSONSchemaBytes {
		return nil, fmt.Errorf(
			"JSON schema exceeds %d bytes",
			maxJSONSchemaBytes,
		)
	}
	decoder := json.NewDecoder(bytes.NewReader(input))
	decoder.UseNumber()
	value, err := decodeOrderedJSONValue(decoder, 0)
	if err != nil {
		return nil, err
	}
	if token, trailingErr := decoder.Token(); trailingErr != io.EOF {
		if trailingErr != nil {
			return nil, trailingErr
		}
		return nil, fmt.Errorf("JSON schema has trailing token %v", token)
	}
	return value, nil
}

func decodeOrderedJSONValue(decoder *json.Decoder, depth int) (any, error) {
	if depth > maxJSONSchemaDepth {
		return nil, fmt.Errorf("JSON schema exceeds %d nesting levels", maxJSONSchemaDepth)
	}
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delimiter, isDelimiter := token.(json.Delim)
	if !isDelimiter {
		return token, nil
	}
	switch delimiter {
	case '{':
		object := orderedJSONObject{}
		seen := make(map[string]bool)
		for decoder.More() {
			nameToken, nameErr := decoder.Token()
			if nameErr != nil {
				return nil, nameErr
			}
			name, ok := nameToken.(string)
			if !ok {
				return nil, errors.New("JSON object key is not a string")
			}
			if seen[name] {
				return nil, fmt.Errorf("JSON object repeats key %q", name)
			}
			seen[name] = true
			value, valueErr := decodeOrderedJSONValue(decoder, depth+1)
			if valueErr != nil {
				return nil, valueErr
			}
			object.Members = append(object.Members, orderedJSONMember{
				Name:  name,
				Value: value,
			})
		}
		if end, endErr := decoder.Token(); endErr != nil || end != json.Delim('}') {
			if endErr != nil {
				return nil, endErr
			}
			return nil, errors.New("JSON object is not closed")
		}
		return object, nil
	case '[':
		array := make([]any, 0)
		for decoder.More() {
			value, valueErr := decodeOrderedJSONValue(decoder, depth+1)
			if valueErr != nil {
				return nil, valueErr
			}
			array = append(array, value)
			if len(array) > maxJSONSchemaArrayItems {
				return nil, fmt.Errorf("JSON array exceeds %d items", maxJSONSchemaArrayItems)
			}
		}
		if end, endErr := decoder.Token(); endErr != nil || end != json.Delim(']') {
			if endErr != nil {
				return nil, endErr
			}
			return nil, errors.New("JSON array is not closed")
		}
		return array, nil
	default:
		return nil, fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
}

func marshalOrderedJSON(value any) (string, error) {
	var output bytes.Buffer
	if err := appendOrderedJSON(&output, value, 0); err != nil {
		return "", err
	}
	return output.String(), nil
}

func appendOrderedJSON(output *bytes.Buffer, value any, depth int) error {
	if depth > 256 {
		return errors.New("JSON value exceeds 256 nesting levels")
	}
	switch typed := value.(type) {
	case orderedJSONObject:
		output.WriteByte('{')
		for index, member := range typed.Members {
			if index > 0 {
				output.WriteByte(',')
			}
			name, _ := json.Marshal(member.Name)
			output.Write(name)
			output.WriteByte(':')
			if err := appendOrderedJSON(output, member.Value, depth+1); err != nil {
				return err
			}
		}
		output.WriteByte('}')
	case []any:
		output.WriteByte('[')
		for index, item := range typed {
			if index > 0 {
				output.WriteByte(',')
			}
			if err := appendOrderedJSON(output, item, depth+1); err != nil {
				return err
			}
		}
		output.WriteByte(']')
	case string:
		encoded, _ := json.Marshal(typed)
		output.Write(encoded)
	case json.Number:
		if _, err := strconv.ParseFloat(string(typed), 64); err != nil {
			return fmt.Errorf("invalid JSON number %q", typed)
		}
		output.WriteString(string(typed))
	case bool:
		output.WriteString(strconv.FormatBool(typed))
	case nil:
		output.WriteString("null")
	default:
		return fmt.Errorf("unsupported ordered JSON value %T", value)
	}
	return nil
}
