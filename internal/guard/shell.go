package guard

import (
	"path"
	"slices"
	"strings"
)

// word is one word of a shell command with its quoting taken off. spelled keeps the backslashes an unquoted word was
// written with, which the shell takes off and a Windows path means.
type word struct {
	text    string
	spelled string
	quoted  bool
}

// shellCommand is one simple command: its words, the files and streams its output is sent to, and whether it reads the
// output of the command before it.
type shellCommand struct {
	words []word
	sinks []string
	piped bool
}

// script is what one command line runs: every simple command, those inside a substitution among them, and the bodies of
// its heredocs, which are data.
type script struct {
	commands []shellCommand
	heredocs []string
}

// shellReader walks a command line the way a shell reads it. It accepts anything, since a line the shell would refuse
// runs nothing: an unclosed quote or substitution ends at the end of the line.
type shellReader struct {
	text    string
	at      int
	out     *script
	current shellCommand
	piped   bool
	pending []heredocMark
}

// heredocMark is a heredoc begun on a line, whose body starts on the next one.
type heredocMark struct {
	delimiter string
	indented  bool
}

// readShell reads a command line into what it runs.
func readShell(text string) script {
	var parsed script
	reader := shellReader{text: text, out: &parsed}
	reader.run(0)
	return parsed
}

// run reads commands up to the closing byte of a substitution, or the end of the line.
func (r *shellReader) run(closing byte) {
	for r.at < len(r.text) {
		character := r.text[r.at]
		switch {
		case closing != 0 && character == closing:
			r.at++
			r.endCommand(false)
			return
		case character == ' ' || character == '\t' || character == '\r':
			r.at++
		case character == '\n':
			r.at++
			r.endCommand(false)
			r.readHeredocs()
		case character == '#':
			r.skipComment()
		case character == ';' || character == '(' || character == ')':
			r.at++
			r.endCommand(false)
		case character == '|':
			r.at++
			double := r.peek('|')
			if double {
				r.at++
			}
			r.endCommand(!double)
		case character == '&':
			r.at++
			switch {
			case r.peek('>'):
				r.readRedirect()
			case r.peek('&'):
				r.at++
				r.endCommand(false)
			default:
				r.endCommand(false)
			}
		case character == '>' || character == '<':
			r.readRedirect()
		default:
			read := r.readWord()
			// A number against a redirect names the stream it redirects.
			if !read.quoted && (r.peek('>') || r.peek('<')) && strings.Trim(read.text, "0123456789") == "" {
				continue
			}
			r.current.words = append(r.current.words, read)
		}
	}
	r.endCommand(false)
}

func (r *shellReader) peek(character byte) bool {
	return r.at < len(r.text) && r.text[r.at] == character
}

// endCommand closes the command being read, and says whether the next one reads its output.
func (r *shellReader) endCommand(piping bool) {
	if len(r.current.words) > 0 || len(r.current.sinks) > 0 {
		r.current.piped = r.piped
		r.out.commands = append(r.out.commands, r.current)
	}
	r.current, r.piped = shellCommand{}, piping
}

func (r *shellReader) skipComment() {
	if end := strings.IndexByte(r.text[r.at:], '\n'); end >= 0 {
		r.at += end
		return
	}
	r.at = len(r.text)
}

// readRedirect reads a redirection and what it names: an output's target is a sink, and a heredoc waits for its line to end.
func (r *shellReader) readRedirect() {
	operator := strings.TrimRight(r.text[r.at:], "")
	end := 0
	for end < len(operator) && strings.IndexByte("<>&|-", operator[end]) >= 0 {
		end++
	}
	operator = operator[:end]
	// A dash, a bar or an ampersand belongs to the operator only where a shell spells it so.
	for _, spelled := range []string{"<<<", "<<-", "<<", ">>", ">&", ">|", "<&", "<>", ">", "<"} {
		if strings.HasPrefix(operator, spelled) {
			operator = spelled
			break
		}
	}
	r.at += len(operator)
	for r.at < len(r.text) && (r.text[r.at] == ' ' || r.text[r.at] == '\t') {
		r.at++
	}
	target := r.readWord()
	switch operator {
	case ">", ">>", ">|":
		r.current.sinks = append(r.current.sinks, target.text)
	case ">&":
		r.current.sinks = append(r.current.sinks, "&"+target.text)
	case "<<", "<<-":
		r.pending = append(r.pending, heredocMark{delimiter: target.text, indented: operator == "<<-"})
	}
}

// readHeredocs reads the bodies a finished line opened, each up to the line that is its delimiter.
func (r *shellReader) readHeredocs() {
	for _, mark := range r.pending {
		var body strings.Builder
		for r.at < len(r.text) {
			line, _, ended := strings.Cut(r.text[r.at:], "\n")
			r.at += len(line)
			if ended {
				r.at++
			}
			closing := strings.TrimRight(line, "\r")
			if mark.indented {
				closing = strings.TrimLeft(closing, "\t")
			}
			if closing == mark.delimiter {
				break
			}
			body.WriteString(line + "\n")
		}
		r.out.heredocs = append(r.out.heredocs, body.String())
	}
	r.pending = nil
}

// substitution reads the command a $( or a backtick opens, which the shell runs wherever it stands, a double quote included.
func (r *shellReader) substitution(closing byte) {
	inner := shellReader{text: r.text, at: r.at, out: r.out}
	inner.run(closing)
	r.at = inner.at
}

// readWord reads one word, taking its quotes and escapes off.
func (r *shellReader) readWord() word {
	var read word
	var text strings.Builder
	escapes := map[int]bool{}
	finished := func() word {
		read.text = text.String()
		var spelled strings.Builder
		for index := range len(read.text) {
			if escapes[index] {
				spelled.WriteByte('\\')
			}
			spelled.WriteByte(read.text[index])
		}
		read.spelled = spelled.String()
		return read
	}
	for r.at < len(r.text) {
		character := r.text[r.at]
		switch {
		case strings.IndexByte(" \t\r\n;&|()<>", character) >= 0:
			return finished()
		case character == '\'':
			read.quoted = true
			r.at++
			quoted, _, _ := strings.Cut(r.text[r.at:], "'")
			text.WriteString(quoted)
			r.at = min(r.at+len(quoted)+1, len(r.text))
		case character == '"':
			read.quoted = true
			r.at++
			r.readDoubleQuoted(&text)
		case character == '\\':
			r.at++
			if r.at < len(r.text) && r.text[r.at] != '\n' {
				escapes[text.Len()] = true
				text.WriteByte(r.text[r.at])
			}
			r.at++
		case character == '`':
			r.at++
			r.substitution('`')
		case character == '$' && strings.HasPrefix(r.text[r.at:], "$("):
			r.at += len("$(")
			r.substitution(')')
		default:
			text.WriteByte(character)
			r.at++
		}
	}
	return finished()
}

// readDoubleQuoted reads up to the closing quote: a backslash escapes only what a shell lets it, and a substitution still runs.
func (r *shellReader) readDoubleQuoted(text *strings.Builder) {
	for r.at < len(r.text) {
		character := r.text[r.at]
		switch {
		case character == '"':
			r.at++
			return
		case character == '\\' && r.at+1 < len(r.text) && strings.IndexByte("\"\\$`", r.text[r.at+1]) >= 0:
			text.WriteByte(r.text[r.at+1])
			r.at += len(`\"`)
		case character == '`':
			r.at++
			r.substitution('`')
		case character == '$' && strings.HasPrefix(r.text[r.at:], "$("):
			r.at += len("$(")
			r.substitution(')')
		default:
			text.WriteByte(character)
			r.at++
		}
	}
}

// commandPrefixes are the words that stand before a command and run it, a shell's keywords among them.
var commandPrefixes = []string{"!", "{", "command", "do", "elif", "else", "env", "exec", "if", "nohup", "sudo", "then", "time", "until", "while"}

// flagsWithValues are the options of a prefix or of xargs whose value is the next word.
var flagsWithValues = []string{"-C", "-E", "-I", "-L", "-P", "-c", "-d", "-n", "-s", "-u"}

// invocation is a command's words from the program it runs: assignments and the prefixes that run it are behind that.
func (c shellCommand) invocation() []word {
	words := c.words
	for len(words) > 0 {
		first := words[0]
		name, _, assigned := strings.Cut(first.text, "=")
		switch {
		case !first.quoted && assigned && name != "" && !strings.ContainsAny(name, `/\-`):
			words = words[1:]
		case slices.Contains(commandPrefixes, strings.ToLower(first.text)):
			words = skipFlags(words[1:])
		default:
			return words
		}
	}
	return nil
}

// skipFlags leaves the words after a run of options.
func skipFlags(words []word) []word {
	for len(words) > 0 && strings.HasPrefix(words[0].text, "-") {
		if slices.Contains(flagsWithValues, words[0].text) && len(words) > 1 {
			words = words[1:]
		}
		words = words[1:]
	}
	return words
}

// programName is a program's name in lower case, without its directory or .exe.
func programName(text string) string {
	return strings.TrimSuffix(strings.ToLower(path.Base(strings.ReplaceAll(text, `\`, "/"))), ".exe")
}

// run is what a command runs on behalf of another: the words after xargs, or those after find's -exec up to their end.
func runsFor(words []word) [][]word {
	var runs [][]word
	for index, current := range words {
		if current.quoted {
			continue
		}
		switch {
		case programName(current.text) == "xargs":
			if rest := skipFlags(words[index+1:]); len(rest) > 0 {
				runs = append(runs, rest)
			}
		case current.text == "-exec" || current.text == "-execdir":
			rest := words[index+1:]
			if end := slices.IndexFunc(rest, func(next word) bool { return next.text == ";" || next.text == "+" }); end >= 0 {
				rest = rest[:end]
			}
			if len(rest) > 0 {
				runs = append(runs, rest)
			}
		}
	}
	return runs
}

// programs is every command a script runs, what xargs and find's -exec run for it among them.
func (s script) programs() []shellCommand {
	var commands []shellCommand
	for _, command := range s.commands {
		commands = append(commands, command)
		for _, run := range runsFor(command.words) {
			commands = append(commands, shellCommand{words: run})
		}
	}
	return commands
}

// isPackage says a word names a command's package or one beneath it.
func isPackage(text, name string) bool {
	lower := strings.ToLower(text)
	return lower == name || strings.HasPrefix(lower, name+"/")
}

// names says a command holds a word that names the package.
func (c shellCommand) names(name string) bool {
	return slices.ContainsFunc(c.words, func(current word) bool { return isPackage(current.text, name) })
}

// holds says a command holds one of the flags, however a shell quoted it, alone or with its value after an equals sign.
func (c shellCommand) holds(flags ...string) bool {
	return slices.ContainsFunc(c.words, func(current word) bool {
		flag, _, _ := strings.Cut(current.text, "=")
		return slices.Contains(flags, flag)
	})
}
