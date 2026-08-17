package artifact

import (
	"encoding/json"
	"fmt"
)

func marshalEnumJSON(value int, names []string, kind string) ([]byte, error) {
	if value <= 0 || value >= len(names) || names[value] == "" {
		return nil, fmt.Errorf("artifact: invalid %s", kind)
	}
	return json.Marshal(names[value])
}

func unmarshalEnumJSON(data []byte, names []string, kind string) (int, error) {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return 0, fmt.Errorf("artifact: decode %s: %w", kind, err)
	}
	return parseEnum(value, names, kind)
}

func unmarshalEnumInto[T ~uint8](target *T, data []byte, names []string, kind string) error {
	if target == nil {
		return fmt.Errorf("artifact: nil %s target", kind)
	}
	value, err := unmarshalEnumJSON(data, names, kind)
	if err != nil {
		return err
	}
	*target = T(value)
	return nil
}

func parseEnum(value string, names []string, kind string) (int, error) {
	for index := 1; index < len(names); index++ {
		if value == names[index] {
			return index, nil
		}
	}
	return 0, fmt.Errorf("artifact: unknown %s %q", kind, value)
}
