package runrecord

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// VerifyCommand is a plan verifier parsed once: its segments in order.
// Execution, dependency discovery, evidence matching and memoisation read
// this form instead of each re-scanning the text, so quoting, flags and file
// guards mean the same thing to all of them.
type VerifyCommand struct {
	Segments []VerifySegment
	// Composite marks a command that pipes, redirects, sequences with ; or
	// ||, substitutes or expands a variable; the shell still runs it, but its
	// segments are no longer a plain && chain a caller can reason about.
	Composite bool
}

// SegmentKind names what one verifier segment runs; the zero kind is any
// command that is neither go test nor a file guard.
type SegmentKind int

const (
	// SegmentGoTest is one go test invocation.
	SegmentGoTest SegmentKind = iota + 1
	// SegmentFileGuard is `test -f <path>`.
	SegmentFileGuard
)

// VerifySegment is one command of a verifier, words unquoted.
type VerifySegment struct {
	Kind SegmentKind
	// Env holds leading NAME=value assignments, an `env` prefix included.
	Env []string
	// Fields are the command words after the environment assignments.
	Fields []string
	// Packages, Short and Flags describe a go test invocation; Flags keeps
	// every other flag in order, values attached. Target is the -run
	// selector of a go test or of a wrapper that forwards one.
	Packages []string
	Target   *GoTestTarget
	Short    bool
	Flags    []string
	// Path is a file guard's path.
	Path string
	// testEnd is the byte offset in the command just past `go test`.
	testEnd int
}

// goTestValueFlags are the go test flags whose value may follow as the next
// word. A value flag missing here reads its value as a package, which no
// caller mistakes for one: the memo finds no such package and refuses, and
// evidence matching reads only ./-relative arguments.
var goTestValueFlags = []string{"-bench", "-benchtime", "-C", "-count", "-cpu", "-exec", "-o", "-p", "-parallel", "-run", "-skip", "-tags", "-timeout"}

// ParseVerify parses a verifier. Words split on unquoted whitespace with
// POSIX quoting; segments split on unquoted shell operators. A plain
// verifier is an && chain; any other operator or an expansion marks it
// composite, which callers that must reason exactly refuse.
func ParseVerify(command string) (VerifyCommand, error) {
	words, composite, err := verifyWords(command)
	if err != nil {
		return VerifyCommand{}, err
	}
	parsed := VerifyCommand{Composite: composite}
	var current []verifyWord
	flush := func() error {
		if len(current) == 0 {
			if composite {
				return nil // a redirect or subshell leaves no words of its own
			}
			return errors.New("verifier has an empty command segment")
		}
		segment, err := parseVerifySegment(current)
		if err != nil {
			return err
		}
		parsed.Segments = append(parsed.Segments, segment)
		current = nil
		return nil
	}
	for _, word := range words {
		if word.operator {
			if err := flush(); err != nil {
				return VerifyCommand{}, err
			}
			continue
		}
		current = append(current, word)
	}
	if err := flush(); err != nil {
		return VerifyCommand{}, err
	}
	return parsed, nil
}

// GoTests returns the go test invocations in order.
func (c VerifyCommand) GoTests() []VerifySegment {
	var tests []VerifySegment
	for _, segment := range c.Segments {
		if segment.Kind == SegmentGoTest {
			tests = append(tests, segment)
		}
	}
	return tests
}

// Targets returns every segment's -run selector in order: each go test
// invocation's and each wrapper's that forwards one.
func (c VerifyCommand) Targets() []*GoTestTarget {
	var targets []*GoTestTarget
	for _, segment := range c.Segments {
		if segment.Target != nil {
			targets = append(targets, segment.Target)
		}
	}
	return targets
}

// Short reports whether any go test invocation runs in short mode.
func (c VerifyCommand) Short() bool {
	return slices.ContainsFunc(c.Segments, func(segment VerifySegment) bool { return segment.Short })
}

// JSONCommand returns command with -json added to every go test invocation
// that lacks it, inserted where the parse found each `go test`, so text a
// segment merely quotes is never rewritten.
func (c VerifyCommand) JSONCommand(command string) string {
	for _, test := range slices.Backward(c.GoTests()) {
		if slices.Contains(test.Flags, "-json") {
			continue
		}
		command = command[:test.testEnd] + " -json" + command[test.testEnd:]
	}
	return command
}

type verifyWord struct {
	text     string
	end      int
	operator bool
}

// verifyWords splits command into words and operators, reporting whether
// the command composes anything beyond an && chain.
func verifyWords(command string) (words []verifyWord, composite bool, err error) {
	var word strings.Builder
	inWord := false
	emit := func(end int) {
		if inWord {
			words = append(words, verifyWord{text: word.String(), end: end})
			word.Reset()
			inWord = false
		}
	}
	operator := func(end int) {
		emit(end)
		words = append(words, verifyWord{operator: true})
	}
	for i := 0; i < len(command); i++ {
		c := command[i]
		switch c {
		case ' ', '\t', '\n':
			emit(i)
		case '\'':
			closing := strings.IndexByte(command[i+1:], '\'')
			if closing < 0 {
				return nil, false, errors.New("verifier has an unbalanced single quote")
			}
			word.WriteString(command[i+1 : i+1+closing])
			inWord, i = true, i+1+closing
		case '"':
			i++
			for ; i < len(command) && command[i] != '"'; i++ {
				composite = composite || expands(command, i)
				if command[i] == '\\' && i+1 < len(command) && strings.IndexByte("$`\"\\\n", command[i+1]) >= 0 {
					i++
				}
				word.WriteByte(command[i])
			}
			if i == len(command) {
				return nil, false, errors.New("verifier has an unbalanced double quote")
			}
			inWord = true
		case '\\':
			if i+1 == len(command) {
				return nil, false, errors.New("verifier ends in a backslash")
			}
			i++
			word.WriteByte(command[i])
			inWord = true
		case '&', '|':
			// && is the plain chain; a single & or |, and ||, compose.
			double := i+1 < len(command) && command[i+1] == c
			composite = composite || c == '|' || !double
			operator(i)
			if double {
				i++
			}
		case ';', '<', '>', '`', '(', ')':
			composite = true
			operator(i)
		case '$':
			composite = composite || expands(command, i)
			word.WriteByte(c)
			inWord = true
		default:
			word.WriteByte(c)
			inWord = true
		}
	}
	emit(len(command))
	return words, composite, nil
}

// parseVerifySegment classifies one segment's words.
func parseVerifySegment(words []verifyWord) (VerifySegment, error) {
	var segment VerifySegment
	if words[0].text == "env" {
		words = words[1:]
	}
	for len(words) > 0 && isAssignment(words[0].text) {
		segment.Env = append(segment.Env, words[0].text)
		words = words[1:]
	}
	for _, word := range words {
		segment.Fields = append(segment.Fields, word.text)
	}
	// A file guard is the three words `test -f <path>`; a go test is its two
	// command words and then its arguments.
	const guardWords, goTestWords = 3, 2
	switch {
	case len(words) == guardWords && words[0].text == "test" && words[1].text == "-f":
		segment.Kind, segment.Path = SegmentFileGuard, words[guardWords-1].text
	case len(words) >= goTestWords && words[0].text == "go" && words[1].text == "test":
		segment.Kind, segment.testEnd = SegmentGoTest, words[goTestWords-1].end
		if err := segment.parseGoTest(segment.Fields[goTestWords:]); err != nil {
			return VerifySegment{}, err
		}
	case len(words) == 0:
		return VerifySegment{}, errors.New("verifier segment sets an environment and runs nothing")
	default:
		// A wrapper such as cmd/webui-lane forwards -run to its own go test,
		// so its selector is the segment's acceptance target.
		for i, field := range segment.Fields {
			value, attached := strings.CutPrefix(field, "-run=")
			if !attached && (field != "-run" || i+1 == len(segment.Fields)) {
				continue
			}
			if !attached {
				value = segment.Fields[i+1]
			}
			target, err := compileGoTestSelector(value)
			if err != nil {
				return VerifySegment{}, err
			}
			segment.Target = target
			break
		}
	}
	return segment, nil
}

// parseGoTest reads a go test invocation's flags and packages.
func (segment *VerifySegment) parseGoTest(arguments []string) error {
	for i := 0; i < len(arguments); i++ {
		argument := arguments[i]
		if argument == "-args" {
			break
		}
		if !strings.HasPrefix(argument, "-") {
			segment.Packages = append(segment.Packages, argument)
			continue
		}
		name, value, attached := strings.Cut(argument, "=")
		if !attached && slices.Contains(goTestValueFlags, name) {
			if i+1 == len(arguments) {
				return fmt.Errorf("go test flag %s has no value", name)
			}
			i++
			value, attached = arguments[i], true
		}
		switch {
		case name == "-run":
			if segment.Target != nil {
				return errors.New("go test invocation declares more than one -run selector")
			}
			target, err := compileGoTestSelector(value)
			if err != nil {
				return err
			}
			segment.Target = target
		case name == "-short" && (!attached || value == "true"):
			segment.Short = true
		case attached:
			segment.Flags = append(segment.Flags, name+"="+value)
		default:
			segment.Flags = append(segment.Flags, name)
		}
	}
	return nil
}

// expands reports whether the $ at i starts a shell expansion; a $ before
// anything but a name, a brace or a parenthesis is a literal.
func expands(command string, i int) bool {
	if command[i] != '$' || i+1 == len(command) {
		return false
	}
	next := command[i+1]
	return next == '{' || next == '(' || next == '_' || next >= 'A' && next <= 'Z' || next >= 'a' && next <= 'z'
}

// isAssignment reports a NAME=value word.
func isAssignment(word string) bool {
	name, _, found := strings.Cut(word, "=")
	if !found || name == "" {
		return false
	}
	for i, r := range name {
		if r != '_' && (r < 'A' || r > 'Z') && (r < 'a' || r > 'z') && (i == 0 || r < '0' || r > '9') {
			return false
		}
	}
	return true
}
