package sampling

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const pinnedJSONSchemaUpstreamCommit = "42fc243060709331ff9b158a9ed2cbe37219ae83"

type pinnedJSONSchemaFixture struct {
	UpstreamCommit string `json:"upstream_commit"`
	Source         string `json:"source"`
	Cases          []struct {
		Status  string `json:"status"`
		Name    string `json:"name"`
		Schema  string `json:"schema"`
		Grammar string `json:"grammar"`
	} `json:"cases"`
}

func TestJSONSchemaGrammarMatchesPinnedUpstreamCorpus(t *testing.T) {
	input, err := os.ReadFile(filepath.Join("testdata", "json-schema-to-grammar.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture pinnedJSONSchemaFixture
	if err := json.Unmarshal(input, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.UpstreamCommit != pinnedJSONSchemaUpstreamCommit {
		t.Fatalf("fixture commit = %q", fixture.UpstreamCommit)
	}
	if len(fixture.Cases) != 70 {
		t.Fatalf("fixture case count = %d, want 70", len(fixture.Cases))
	}
	for _, test := range fixture.Cases {
		test := test
		t.Run(test.Name, func(t *testing.T) {
			got, err := JSONSchemaToGrammar([]byte(test.Schema))
			if test.Status == "failure" {
				if err == nil {
					t.Fatalf("expected conversion failure, got:\n%s", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if normalizePinnedGrammar(got) != normalizePinnedGrammar(test.Grammar) {
				t.Fatalf(
					"grammar mismatch\nwant:\n%s\n\ngot:\n%s",
					normalizePinnedGrammar(test.Grammar),
					normalizePinnedGrammar(got),
				)
			}
			if _, err := NewGBNFGrammar(
				got,
				"root",
				[][]byte{[]byte("x"), nil},
				[]int{1},
			); err != nil {
				t.Fatalf("generated grammar does not compile: %v\n%s", err, got)
			}
		})
	}
}

func normalizePinnedGrammar(value string) string {
	lines := strings.Split(strings.ReplaceAll(strings.TrimSpace(value), "\r\n", "\n"), "\n")
	for index := range lines {
		lines[index] = strings.TrimLeft(lines[index], " \t")
	}
	return strings.Join(lines, "\n")
}

func TestJSONSchemaGrammarMatchesPinnedCoreOracles(t *testing.T) {
	cases := []struct {
		name   string
		schema string
		want   string
	}{
		{
			name: "ordered object",
			schema: `{"type":"object","properties":{"z":{"const":1},` +
				`"a":{"type":"string"}},"required":["z"],"additionalProperties":false}`,
			want: `a-kv ::= "\"a\"" space ":" space string
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
root ::= "{" space z-kv ( "," space ( a-kv ) )? space "}"
space ::= | " " | "\n"{1,2} [ \t]{0,20}
string ::= "\"" char* "\""
z ::= "1"
z-kv ::= "\"z\"" space ":" space z`,
		},
		{
			name:   "bounded array",
			schema: `{"type":"array","items":{"enum":["a","b"]},"minItems":1,"maxItems":3}`,
			want: `item ::= ("\"a\"" | "\"b\"")
root ::= "[" space item ("," space item){0,2} space "]"
space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		},
		{
			name:   "union",
			schema: `{"anyOf":[{"type":"boolean"},{"type":"null"}]}`,
			want: `boolean ::= ("true" | "false")
null ::= "null"
root ::= boolean | null
space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		},
		{
			name: "recursive ref",
			schema: `{"$defs":{"node":{"type":"object","properties":{"next":{"anyOf":[` +
				`{"$ref":"#/$defs/node"},{"type":"null"}]}},"additionalProperties":false}},` +
				`"$ref":"#/$defs/node"}`,
			want: `null ::= "null"
ref-defs-node ::= "{" space  (ref-defs-node-next-kv )? space "}"
ref-defs-node-next ::= ref-defs-node-next-0 | null
ref-defs-node-next-0 ::= ref-defs-node
ref-defs-node-next-kv ::= "\"next\"" space ":" space ref-defs-node-next
root ::= ref-defs-node
space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		},
		{
			name: "additional property exclusion",
			schema: `{"type":"object","properties":{"foo":{"type":"integer"},` +
				`"bar":{"type":"boolean"}},"additionalProperties":{"type":"string"}}`,
			want: `additional-k ::= ["] ( [b] ([a] ([r] char+ | [^"r] char*) | [^"a] char*) | [f] ([o] ([o] char+ | [^"o] char*) | [^"o] char*) | [^"bf] char* )? ["]
additional-kv ::= additional-k ":" space string
bar-kv ::= "\"bar\"" space ":" space boolean
bar-rest ::= ( "," space additional-kv )*
boolean ::= ("true" | "false")
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
foo-kv ::= "\"foo\"" space ":" space integer
foo-rest ::= ( "," space bar-kv )? bar-rest
integer ::= ("-"? integral-part)
integral-part ::= [0] | [1-9] [0-9]{0,15}
root ::= "{" space  (foo-kv foo-rest | bar-kv bar-rest | additional-kv ( "," space additional-kv )* )? space "}"
space ::= | " " | "\n"{1,2} [ \t]{0,20}
string ::= "\"" char* "\""`,
		},
		{
			name: "allOf object composition",
			schema: `{"allOf":[{"type":"object","properties":{"a":{"type":"string"}}},` +
				`{"type":"object","properties":{"b":{"type":"integer"}}}]}`,
			want: `a-kv ::= "\"a\"" space ":" space string
b-kv ::= "\"b\"" space ":" space integer
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
integer ::= ("-"? integral-part)
integral-part ::= [0] | [1-9] [0-9]{0,15}
root ::= "{" space a-kv "," space b-kv space "}"
space ::= | " " | "\n"{1,2} [ \t]{0,20}
string ::= "\"" char* "\""`,
		},
		{
			name:   "allOf enum intersection",
			schema: `{"type":"string","allOf":[{"enum":["a","b","c"]},{"enum":["b","c","d"]}]}`,
			want: `root ::= ("\"b\"" | "\"c\"")
space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			got, err := JSONSchemaToGrammar([]byte(test.schema))
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("grammar:\n%s\nwant:\n%s", got, test.want)
			}
		})
	}
}

func TestJSONSchemaIntegerBoundsMatchPinnedOracles(t *testing.T) {
	cases := []struct {
		schema string
		root   string
	}{
		{
			`{"type":"integer","minimum":42}`,
			`([1-3] [0-9]{2,15} | [4] ([0-1] [0-9]{1,14} | [2-9] [0-9]{0,14}) | [5-9] [0-9]{1,15})`,
		},
		{
			`{"type":"integer","maximum":42}`,
			`("-" [1-9] [0-9]{0,15} | [0-9] | ([1-3] [0-9] | [4] [0-2]))`,
		},
		{
			`{"type":"integer","minimum":-42}`,
			`("-" ([0-9] | ([1-3] [0-9] | [4] [0-2])) | [0] | [1-9] [0-9]{0,15})`,
		},
		{
			`{"type":"integer","maximum":-42}`,
			`("-" ([0-3] [0-9]{2,15} | [4] ([0-1] [0-9]{1,14} | [2-9] [0-9]{0,14}) | [5-9] [0-9]{1,15}))`,
		},
		{
			`{"type":"integer","minimum":-12,"maximum":345}`,
			`("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1-2] [0-9]{2} | [3] ([0-3] [0-9] | [4] [0-5])))`,
		},
		{
			`{"type":"integer","exclusiveMinimum":9,"exclusiveMaximum":21}`,
			`(([1] [0-9] | [2] "0"))`,
		},
	}
	for _, test := range cases {
		grammar, err := JSONSchemaToGrammar([]byte(test.schema))
		if err != nil {
			t.Fatal(err)
		}
		want := "root ::= " + test.root + "\n" +
			`space ::= | " " | "\n"{1,2} [ \t]{0,20}`
		if grammar != want {
			t.Fatalf("schema %s grammar:\n%s\nwant:\n%s", test.schema, grammar, want)
		}
	}
}

func TestJSONSchemaPatternConstraintIsTranslated(t *testing.T) {
	grammar, err := JSONSchemaToGrammar(
		[]byte(`{"type":"string","pattern":"^[a-z]+$"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(grammar, `[a-z]+`) {
		t.Fatalf("pattern constraint missing from grammar:\n%s", grammar)
	}
}

var pinnedJSONSchemaPatternOracles = []struct {
	pattern string
	grammar string
}{
	{"^abc$", `root ::= "\"" ("abc") "\""
space ::= | " " | "\n"{1,2} [ \t]{0,20}`},
	{"^[a-z]+$", `root ::= "\"" ([a-z]+) "\""
space ::= | " " | "\n"{1,2} [ \t]{0,20}`},
	{"^(ab|cd)?$", `root ::= "\"" (("ab" | "cd")?) "\""
space ::= | " " | "\n"{1,2} [ \t]{0,20}`},
	{"^a.{2,4}z$", `dot ::= [^\x0A\x0D]
root ::= "\"" ("a" root-1{2,4} "z") "\""
root-1 ::= dot
space ::= | " " | "\n"{1,2} [ \t]{0,20}`},
	{"^[A-Z][0-9]{3}$", `root ::= "\"" ([A-Z] root-1{3,3}) "\""
root-1 ::= [0-9]
space ::= | " " | "\n"{1,2} [ \t]{0,20}`},
}

func TestPinnedPatternCorpus(t *testing.T) {
	for _, oracle := range pinnedJSONSchemaPatternOracles {
		oracle := oracle
		t.Run(oracle.pattern, func(t *testing.T) {
			schema, err := json.Marshal(map[string]any{
				"type":    "string",
				"pattern": oracle.pattern,
			})
			if err != nil {
				t.Fatal(err)
			}
			grammar, err := JSONSchemaToGrammar(schema)
			if err != nil {
				t.Fatal(err)
			}
			if grammar != oracle.grammar {
				t.Fatalf("grammar mismatch\nwant:\n%s\n\ngot:\n%s", oracle.grammar, grammar)
			}
		})
	}
}

func TestPinnedNonCapturingPatternCorpus(t *testing.T) {
	cases := []struct {
		pattern string
		grammar string
	}{
		{
			pattern: `^(?:foo|bar)baz$`,
			grammar: `root ::= "\"" (("foo" | "bar") "baz") "\""
space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		},
		{
			pattern: `^(?:(?:ab)+c)?d$`,
			grammar: `root ::= "\"" ((("ab")+ "c")? "d") "\""
space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
		},
	}
	for _, test := range cases {
		schema, err := json.Marshal(map[string]any{
			"type":    "string",
			"pattern": test.pattern,
		})
		if err != nil {
			t.Fatal(err)
		}
		grammar, err := JSONSchemaToGrammar(schema)
		if err != nil {
			t.Fatalf("%s: %v", test.pattern, err)
		}
		if grammar != test.grammar {
			t.Fatalf("%s grammar mismatch\nwant:\n%s\n\ngot:\n%s", test.pattern, test.grammar, grammar)
		}
	}
}

func TestJSONSchemaPatternEdgeCases(t *testing.T) {
	cases := []struct {
		pattern  string
		contains string
		wantErr  bool
	}{
		{pattern: `^$`, contains: `root ::= "\"" ("") "\""`},
		{pattern: `^a{,3}$`, contains: `"a"{0,3}`},
		{pattern: `^a{3,}$`, contains: `"a"{3,}`},
		{pattern: `^a{3,2}$`, wantErr: true},
		{pattern: `^a\$`, wantErr: true},
	}
	for _, test := range cases {
		schema, err := json.Marshal(map[string]any{
			"type":    "string",
			"pattern": test.pattern,
		})
		if err != nil {
			t.Fatal(err)
		}
		grammar, err := JSONSchemaToGrammar(schema)
		if test.wantErr {
			if err == nil {
				t.Fatalf("%s: expected error, got:\n%s", test.pattern, grammar)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", test.pattern, err)
		}
		if !strings.Contains(grammar, test.contains) {
			t.Fatalf("%s grammar lacks %q:\n%s", test.pattern, test.contains, grammar)
		}
		if _, err := NewGBNFGrammar(
			grammar,
			"root",
			[][]byte{[]byte("a"), nil},
			[]int{1},
		); err != nil {
			t.Fatalf("%s grammar does not compile: %v", test.pattern, err)
		}
	}
}
