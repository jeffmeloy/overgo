package testevidence

import (
	"cmp"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

type goTestSelector struct {
	pattern string
	paths   [][]*regexp.Regexp
}

// Go splits top-level alternatives and subtest paths before compiling each
// component. Escapes, character classes and groups protect their separators.
func goTestTargets(command string) ([]*goTestSelector, error) {
	matches := goTestRunFlag.FindAllStringSubmatch(command, -1)
	if len(matches) == 0 {
		return nil, fmt.Errorf("go test verifier must declare its acceptance target with -run")
	}
	var targets []*goTestSelector
	for _, match := range matches {
		target, err := compileGoTestSelector(cmp.Or(match[1], match[2], match[3]))
		if err != nil {
			return nil, err
		}
		targets = append(targets, target)
	}
	return targets, nil
}

func compileGoTestSelector(pattern string) (*goTestSelector, error) {
	target := &goTestSelector{pattern: pattern}
	var path []*regexp.Regexp
	start, classes, groups := 0, 0, 0
	for i := 0; i <= len(target.pattern); i++ {
		separator := byte('|') // Flush the last path at end of input.
		if i < len(target.pattern) {
			separator = target.pattern[i]
			switch separator {
			case '\\':
				if i+1 < len(target.pattern) {
					i++
				}
				continue
			case '[':
				classes++
			case ']':
				// An unmatched closing bracket is a literal in Go patterns.
				if classes > 0 {
					classes--
				}
			case '(':
				if classes == 0 {
					groups++
				}
			case ')':
				if classes == 0 {
					groups--
				}
			}
			if classes != 0 || groups != 0 || separator != '/' && separator != '|' {
				continue
			}
		}
		var normalized strings.Builder
		for _, r := range target.pattern[start:i] {
			switch {
			case unicode.IsSpace(r):
				normalized.WriteByte('_')
			case !strconv.IsPrint(r):
				quoted := strconv.QuoteRune(r)
				normalized.WriteString(quoted[1 : len(quoted)-1])
			default:
				normalized.WriteRune(r)
			}
		}
		component, err := regexp.Compile(normalized.String())
		if err != nil {
			return nil, fmt.Errorf("compile go test -run target: %w", err)
		}
		path = append(path, component)
		if separator == '|' {
			target.paths = append(target.paths, path)
			path = nil
		}
		start = i + 1
	}
	return target, nil
}

func (target *goTestSelector) matches(name string) bool {
	parts := strings.Split(name, "/")
	for _, path := range target.paths {
		// Go runs partial parents to discover children; they prove no child ran.
		if len(parts) < len(path) {
			continue
		}
		matched := true
		for i, component := range path {
			if !component.MatchString(parts[i]) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}
