package runrecord

import (
	"cmp"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// GoTestTarget is one compiled Go -run selector, including subtest paths.
type GoTestTarget struct {
	// Pattern retains the declared selector for diagnostics.
	Pattern string
	paths   [][]*regexp.Regexp
}

var goTestRunFlag = regexp.MustCompile(`(?:^|[ \t])-run(?:=|[ \t]+)(?:'([^']*)'|"([^"]*)"|([^ \t;&|]+))`)

// GoTestTargets compiles every Go -run selector in command. requireTarget
// rejects broad or non-Go declarations; false permits whole-package evidence.
func GoTestTargets(command string, requireTarget bool) ([]*GoTestTarget, error) {
	if requireTarget && !strings.Contains(command, "go test") {
		return nil, fmt.Errorf("verifier %q is not a go test command", command)
	}
	matches := goTestRunFlag.FindAllStringSubmatch(command, -1)
	if len(matches) == 0 && requireTarget {
		return nil, fmt.Errorf("go test verifier must declare its acceptance target with -run")
	}
	var targets []*GoTestTarget
	for _, match := range matches {
		target, err := compileGoTestSelector(cmp.Or(match[1], match[2], match[3]))
		if err != nil {
			return nil, err
		}
		targets = append(targets, target)
	}
	return targets, nil
}

func compileGoTestSelector(pattern string) (*GoTestTarget, error) {
	target := &GoTestTarget{Pattern: pattern}
	var path []*regexp.Regexp
	start, classes, groups := 0, 0, 0
	for i := 0; i <= len(target.Pattern); i++ {
		separator := byte('|') // Flush the last path at end of input.
		if i < len(target.Pattern) {
			separator = target.Pattern[i]
			switch separator {
			case '\\':
				if i+1 < len(target.Pattern) {
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
		for _, r := range target.Pattern[start:i] {
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

// MatchString requires a complete selected path; parent discovery is insufficient.
func (target *GoTestTarget) MatchString(name string) bool {
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
