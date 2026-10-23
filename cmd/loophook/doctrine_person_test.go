package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"
)

// recordVocabulary is the stored control records' reason codes, which name a person until the person-free-records row
// renames them with their records.
var recordVocabulary = map[string]bool{"user-stop": true, "operator-resume": true, "operator-maintenance": true}

// personWords are the words that state a rule as a person's: the doctrine states every rule as the system's own.
var personWords = map[string]bool{
	"user": true, "users": true, "human": true, "humans": true,
	"owner": true, "owners": true, "operator": true, "operators": true,
}

// TestDoctrineNamesNoPerson holds the system's doctrine (skill.md, the plan's standing and campaign paragraphs, and the
// contract the hook prints) to rules stated as the system's own, wherever it lives.
//   - no word names a user, human, owner or operator; a path's segments and the stored record vocabulary are names
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
	texts := map[string]string{
		"skill.md":                 read("skill.md"),
		"cmd/loophook/doctrine.go": read("cmd/loophook/doctrine.go"),
		"cmd/loophook/main.go":     read("cmd/loophook/main.go"),
	}
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
	for source, text := range texts {
		for number, line := range strings.Split(text, "\n") {
			lower := strings.ToLower(line)
			if strings.Contains(lower, "security setting") && strings.Contains(lower, "never") {
				t.Errorf("%s line %d holds security settings as never changed: %s", source, number+1, line)
			}
			for _, word := range strings.FieldsFunc(lower, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_' && r != '/'
			}) {
				if strings.Contains(word, "/") || recordVocabulary[word] {
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
