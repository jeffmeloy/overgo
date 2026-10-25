package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unicode"
)

// personWords are the words that state a rule as a person's: the doctrine states every rule as the system's own.
var personWords = map[string]bool{
	"user": true, "users": true, "human": true, "humans": true,
	"owner": true, "owners": true, "operator": true, "operators": true,
}

// doctrineSources are the Go files whose comments and strings print the doctrine to sessions: the hook's contract
// and the plan command's task prompt and help.
var doctrineSources = []string{"cmd/loophook/doctrine.go", "cmd/loophook/main.go", "cmd/plan/main.go"}

// TestDoctrineNamesNoPerson holds the system's doctrine (skill.md, the plan's standing and campaign paragraphs, and the
// text the hook and the plan command print) to rules stated as the system's own, wherever it lives.
//   - no word names a user, human, owner or operator; a path's segments are names
//   - no line holds OS security settings as something the system never changes: the doctrine changes them as needed,
//     and a session that cannot is the session's limit, never the system's rule
func TestDoctrineNamesNoPerson(t *testing.T) {
	root := filepath.Join("..", "..")
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	texts := map[string]string{"skill.md": read("skill.md")}
	var plan struct {
		Paragraphs []struct {
			Governs string `json:"governs"`
			Text    string `json:"text"`
		} `json:"paragraphs"`
	}
	if err := json.Unmarshal([]byte(read("docs/plan.json")), &plan); err != nil {
		t.Fatal(err)
	}
	for index, paragraph := range plan.Paragraphs {
		if paragraph.Governs == "standing" || paragraph.Governs == "campaign" {
			texts[fmt.Sprintf("docs/plan.json paragraph %d", index)] = paragraph.Text
		}
	}
	for _, path := range doctrineSources {
		files := token.NewFileSet()
		file, err := parser.ParseFile(files, filepath.Join(root, path), nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		for _, group := range file.Comments {
			texts[files.Position(group.Pos()).String()] = group.Text()
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if literal, ok := node.(*ast.BasicLit); ok && literal.Kind == token.STRING {
				value, err := strconv.Unquote(literal.Value)
				if err != nil {
					t.Fatal(err)
				}
				texts[files.Position(literal.Pos()).String()] = value
			}
			return true
		})
	}
	for source, text := range texts {
		for number, line := range strings.Split(text, "\n") {
			lower := strings.ToLower(line)
			if strings.Contains(lower, "security setting") && strings.Contains(lower, "never") {
				t.Errorf("%s line %d holds security settings as never changed: %s", source, number+1, line)
			}
			for _, word := range strings.FieldsFunc(lower, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_' && r != '/'
			}) {
				if strings.Contains(word, "/") {
					continue
				}
				for part := range strings.SplitSeq(word, "-") {
					if personWords[part] {
						t.Errorf("%s line %d names a person (%q): %s", source, number+1, part, line)
					}
				}
			}
		}
	}
}
