package jinja

import "testing"

func TestRenderCoreConstructs(t *testing.T) {
	cases := []struct {
		name string
		src  string
		ctx  map[string]any
		want string
	}{
		{"text", `hello`, nil, "hello"},
		{"output", `{{ x }}`, map[string]any{"x": "hi"}, "hi"},
		{"bool_render", `{{ true }}|{{ false }}|{{ none }}`, nil, "True|False|"},
		{"int_concat", `{{ 'a' + 'b' }}{{ 1 + 2 }}{{ 'n' ~ 3 }}`, nil, "ab3n3"},
		{"escape_nl", `{{ 'a\nb' }}`, nil, "a\nb"},
		{"trim_left", "x  {%- if true %}Y{% endif %}", nil, "xY"},
		{"trim_right", "{% if true -%}  Y{% endif %}", nil, "Y"},
		{"for_loop", `{% for i in [1,2,3] %}{{ i }}{{ loop.index }}{{ loop.last }};{% endfor %}`, nil,
			"11False;22False;33True;"},
		{"for_map_sorted", `{% for k in d %}{{ k }};{% endfor %}`, map[string]any{"d": map[string]any{"b": 1, "A": 2, "c": 3}}, "A;b;c;"},
		{"if_elif_else", `{% if x == 1 %}one{% elif x == 2 %}two{% else %}other{% endif %}`, map[string]any{"x": 2}, "two"},
		{"set_and_use", `{% set y = 'z' %}{{ y }}`, nil, "z"},
		{"block_set", `{% set b %}a{{ 1 }}{% endset %}{{ b }}`, nil, "a1"},
		{"namespace", `{% set ns = namespace(v=0) %}{% for i in [1,2,3] %}{% set ns.v = ns.v + i %}{% endfor %}{{ ns.v }}`, nil, "6"},
		{"ternary", `{{ 'y' if x else 'n' }}`, map[string]any{"x": true}, "y"},
		{"ternary_noelse", `[{{ 'z' if x }}]`, map[string]any{"x": false}, "[]"},
		{"tests", `{{ x is defined }}{{ y is none }}{{ x is string }}{{ n is number }}`,
			map[string]any{"x": "s", "n": 5}, "TrueTrueTrueTrue"},
		{"is_not_in", `{{ 'a' in ['a','b'] }}{{ 'c' not in ['a','b'] }}`, nil, "TrueTrue"},
		{"slice_step", `{% for i in [1,2,3,4][::-1] %}{{ i }}{% endfor %}`, nil, "4321"},
		{"slice_range", `{{ 'hello'[1:3] }}`, nil, "el"},
		{"filters", `{{ '  x  '|trim|upper }}{{ [3,1,2]|min }}{{ ['a','b']|map('upper')|list }}`, nil, "X1['A', 'B']"},
		{"default", `{{ missing|default('D') }}{{ ''|default('E', true) }}`, nil, "DE"},
		{"macro", `{% macro g(a, b='!') %}{{ a }}{{ b }}{% endmacro %}{{ g('hi') }}{{ g('yo','?') }}`, nil, "hi!yo?"},
		{"method_split", `{{ 'a</t>b'.split('</t>')[-1] }}`, nil, "b"},
		{"method_strip", `[{{ '\n x \n'.strip() }}][{{ '\nx\n'.strip('\n') }}]`, nil, "[x][x]"},
		{"comment", `a{# hidden #}b`, nil, "ab"},
		{"dict_get", `{{ d.get('k', 'def') }}{{ d.get('missing', 'def') }}`, map[string]any{"d": map[string]any{"k": "v"}}, "vdef"},
	}
	env := New()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := env.Render(c.src, c.ctx)
			if err != nil {
				t.Fatalf("render error: %v", err)
			}
			if got != c.want {
				t.Fatalf("got %q, want %q", got, c.want)
			}
		})
	}
}
