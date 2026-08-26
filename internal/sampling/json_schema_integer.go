package sampling

import (
	"fmt"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
)

func buildSchemaIntegerRange(minimum, maximum int64) (string, error) {
	if minimum > maximum {
		return "", fmt.Errorf("integer minimum %d exceeds maximum %d", minimum, maximum)
	}
	if minimum < 0 && maximum < 0 {
		inner, err := buildSchemaIntegerRange(-maximum, -minimum)
		if err != nil {
			return "", err
		}
		return `"-" (` + inner + `)`, nil
	}
	if minimum < 0 {
		negative, err := buildSchemaIntegerRange(0, -minimum)
		if err != nil {
			return "", err
		}
		positive, err := buildSchemaIntegerRange(0, maximum)
		if err != nil {
			return "", err
		}
		return `"-" (` + negative + `) | ` + positive, nil
	}
	from := strconv.FormatInt(minimum, 10)
	to := strconv.FormatInt(maximum, 10)
	var alternatives []string
	for len(from) < len(to) {
		value, err := uniformSchemaIntegerRange(from, strings.Repeat("9", len(from)))
		if err != nil {
			return "", err
		}
		alternatives = append(alternatives, value)
		from = "1" + strings.Repeat("0", len(from))
	}
	value, err := uniformSchemaIntegerRange(from, to)
	if err != nil {
		return "", err
	}
	alternatives = append(alternatives, value)
	return strings.Join(alternatives, " | "), nil
}

func uniformSchemaIntegerRange(from, to string) (string, error) {
	if len(from) != len(to) || from > to {
		return "", fmt.Errorf("invalid uniform integer range %q..%q", from, to)
	}
	common := 0
	for common < len(from) && from[common] == to[common] {
		common++
	}
	prefix := ""
	if common > 0 {
		prefix = `"` + from[:common] + `"`
	}
	if common == len(from) {
		return prefix, nil
	}
	if prefix != "" {
		prefix += " "
	}
	remaining := len(from) - common - 1
	if remaining == 0 {
		return prefix + schemaDigitRange(from[common], to[common]), nil
	}
	fromTail := from[common+1:]
	toTail := to[common+1:]
	zeros := strings.Repeat("0", remaining)
	nines := strings.Repeat("9", remaining)
	var alternatives []string
	if fromTail == zeros {
		if from[common] < to[common] {
			alternatives = append(
				alternatives,
				schemaDigitRange(from[common], to[common]-1)+" "+schemaDigitCount(remaining),
			)
		}
	} else {
		tail, err := uniformSchemaIntegerRange(fromTail, nines)
		if err != nil {
			return "", err
		}
		alternatives = append(
			alternatives,
			schemaDigitRange(from[common], from[common])+" ("+tail+")",
		)
		toReached := false
		if from[common]+1 < to[common] {
			end := to[common] - 1
			if toTail == nines {
				end = to[common]
				toReached = true
			}
			alternatives = append(
				alternatives,
				schemaDigitRange(from[common]+1, end)+" "+schemaDigitCount(remaining),
			)
		}
		if toReached {
			return prefix + "(" + strings.Join(alternatives, " | ") + ")", nil
		}
	}
	tail, err := uniformSchemaIntegerRange(zeros, toTail)
	if err != nil {
		return "", err
	}
	alternatives = append(
		alternatives,
		schemaDigitRange(to[common], to[common])+" "+tail,
	)
	return prefix + "(" + strings.Join(alternatives, " | ") + ")", nil
}

func schemaDigitRange(from, to byte) string {
	if from == to {
		return "[" + string(from) + "]"
	}
	return "[" + string(from) + "-" + string(to) + "]"
}

func schemaDigitCount(count int) string {
	if count == 1 {
		return "[0-9]"
	}
	return fmt.Sprintf("[0-9]{%d}", count)
}

func buildSchemaIntegerMinimum(minimum int64) (string, error) {
	return buildSchemaIntegerMinimumDepth(minimum, 16, true)
}

func buildSchemaIntegerMinimumDepth(
	minimum int64,
	digitsLeft int,
	topLevel bool,
) (string, error) {
	lessDigits := max(digitsLeft-1, 1)
	if minimum < 0 {
		bounded, err := buildSchemaIntegerMaximumDepth(
			-minimum,
			digitsLeft,
			false,
		)
		if err != nil {
			return "", err
		}
		return `"-" (` + bounded + `) | [0] | [1-9] ` +
			schemaVariableDigitCount(0, digitsLeft-1), nil
	}
	if minimum == 0 {
		if topLevel {
			return `[0] | [1-9] ` +
				schemaVariableDigitCount(0, lessDigits), nil
		}
		return schemaVariableDigitCount(1, digitsLeft), nil
	}
	if minimum <= 9 {
		start := byte('0')
		if topLevel {
			start = '1'
		}
		current := byte('0' + minimum)
		var alternatives []string
		if current > start {
			alternatives = append(
				alternatives,
				schemaDigitRange(start, current-1)+" "+
					schemaVariableDigitCount(1, lessDigits),
			)
		}
		alternatives = append(
			alternatives,
			schemaDigitRange(current, '9')+" "+
				schemaVariableDigitCount(0, lessDigits),
		)
		return strings.Join(alternatives, " | "), nil
	}
	text := strconv.FormatInt(minimum, 10)
	first := text[0]
	var alternatives []string
	start := byte('0')
	if topLevel {
		start = '1'
	}
	if first > start {
		alternatives = append(
			alternatives,
			schemaDigitRange(start, first-1)+" "+
				schemaVariableDigitCount(len(text), lessDigits),
		)
	}
	tail, err := strconv.ParseInt(text[1:], 10, binaryschema.Width64Bits)
	if err != nil {
		return "", err
	}
	boundedTail, err := buildSchemaIntegerMinimumDepth(
		tail,
		lessDigits,
		false,
	)
	if err != nil {
		return "", err
	}
	alternatives = append(
		alternatives,
		schemaDigitRange(first, first)+" ("+boundedTail+")",
	)
	if first < '9' {
		alternatives = append(
			alternatives,
			schemaDigitRange(first+1, '9')+" "+
				schemaVariableDigitCount(len(text)-1, lessDigits),
		)
	}
	return strings.Join(alternatives, " | "), nil
}

func buildSchemaIntegerMaximum(maximum int64) (string, error) {
	return buildSchemaIntegerMaximumDepth(maximum, 16, true)
}

func buildSchemaIntegerMaximumDepth(
	maximum int64,
	digitsLeft int,
	topLevel bool,
) (string, error) {
	if maximum >= 0 {
		bounded, err := buildSchemaIntegerRange(0, maximum)
		if err != nil {
			return "", err
		}
		if topLevel {
			return `"-" [1-9] ` +
				schemaVariableDigitCount(0, max(digitsLeft-1, 1)) +
				` | ` + bounded, nil
		}
		return bounded, nil
	}
	bounded, err := buildSchemaIntegerMinimumDepth(
		-maximum,
		digitsLeft,
		false,
	)
	if err != nil {
		return "", err
	}
	return `"-" (` + bounded + `)`, nil
}

func schemaVariableDigitCount(minimum, maximum int) string {
	if minimum == maximum && minimum == 1 {
		return "[0-9]"
	}
	if minimum == maximum {
		return fmt.Sprintf("[0-9]{%d}", minimum)
	}
	return fmt.Sprintf("[0-9]{%d,%d}", minimum, maximum)
}
