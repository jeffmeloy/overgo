package seq2seq

import (
	"maps"
	"testing"
)

// TestToolNamesAreRewrittenAsTheModelsOwnCodeRewritesThem holds the two
// rewrites of tool names to what the model's own code returns for the same
// text, recorded from that code: the snake-case rule; a tools list with its
// names rewritten, written compactly with keys in the order they came, a key
// given twice keeping its last value, characters outside ASCII escaped and
// numbers as Python writes them, and any other value or text left without
// names; and an answer with the caller's names written back, by name where it
// is JSON and by replacement, the longest name first, where it is not.
func TestToolNamesAreRewrittenAsTheModelsOwnCodeRewritesThem(t *testing.T) {
	t.Parallel()
	// escaped is a character outside ASCII as the rewritten text spells it:
	// a backslash, a u and four hex digits.
	escaped := func(digits string) string { return string(rune('\\')) + "u" + digits }
	for name, want := range map[string]string{
		"getWeather": "get_weather", "GetHTTPResponse": "get_http_response", "math.factorial": "math_factorial",
		"calc_area": "calc_area", "XMLHttpRequest2Send": "xml_http_request2_send", "__init__": "init",
		"a.b-c d": "a_b_c_d", "already_snake_case": "already_snake_case", "ABC": "abc", "userID": "user_id",
	} {
		if got := snakeCase(name); got != want {
			t.Errorf("%q in snake case is %q, want %q", name, got, want)
		}
	}
	for _, test := range []struct {
		tools, normalized string
		names             map[string]string
	}{
		{
			`[{"name":"math.factorial","parameters":{"number":"integer"}}]`,
			`[{"name":"math_factorial","parameters":{"number":"integer"}}]`, map[string]string{"math_factorial": "math.factorial"},
		},
		{
			`[ {"parameters": {"b": "string", "a": "integer"}, "name": "getWeather"}, {"name": "sendEMail", "name": "SendEmail"} ]`,
			`[{"parameters":{"b":"string","a":"integer"},"name":"get_weather"},{"name":"send_email"}]`,
			map[string]string{"get_weather": "getWeather", "send_email": "SendEmail"},
		},
		{`{"name":"getWeather"}`, `{"name":"getWeather"}`, nil},
		{
			"[{\"name\":\"café.Lookup\",\"parameters\":{\"q\":\"str\\ting \\\"x\\\" / ☃ \U0001F600\"}}]",
			`[{"name":"caf_lookup","parameters":{"q":"str\ting \"x\" / ` + escaped("2603") + " " + escaped("d83d") + escaped("de00") + `"}}]`, map[string]string{"caf_lookup": "café.Lookup"},
		},
		{
			`[{"name":"n","parameters":{"x":1.50,"y":1e5,"z":-0,"w":1e-7,"v":12345678901234567890,"u":1E16,"t":0.1,"s":true,"r":null}}]`,
			`[{"name":"n","parameters":{"x":1.5,"y":100000.0,"z":0,"w":1e-07,"v":12345678901234567890,"u":1e+16,"t":0.1,"s":true,"r":null}}]`,
			map[string]string{"n": "n"},
		},
		{`not json at all`, `not json at all`, nil},
		{`[]`, `[]`, nil},
	} {
		normalized, names := normalizeTools(test.tools)
		if normalized != test.normalized || !maps.Equal(names, test.names) && (len(names) != 0 || len(test.names) != 0) {
			t.Errorf("tools %s\nnormalize to %s with names %v\nwant         %s with names %v", test.tools, normalized, names, test.normalized, test.names)
		}
	}
	both := map[string]string{"get_weather": "getWeather", "get_weather_now": "getWeatherNow"}
	one := map[string]string{"get_weather": "getWeather"}
	for _, test := range []struct {
		answer, restored string
		names            map[string]string
	}{
		{` [{"name":"math_factorial","arguments":{"number":5}}]`, `[{"name":"math.factorial","arguments":{"number":5}}]`, map[string]string{"math_factorial": "math.factorial"}},
		{"{\"name\":\"get_weather\",\"arguments\":{\"location\":\"São Paulo\"}}", `{"name":"getWeather","arguments":{"location":"S` + escaped("00e3") + `o Paulo"}}`, one},
		{` [{"name":"get_weather","arguments":{"a":"1","a":"2"}},{"name":"other"}]`, `[{"name":"getWeather","arguments":{"a":"2"}},{"name":"other"}]`, one},
		{` [{"name":"get_weather","arguments":{"base":"height":"5"}},{"name":"get_weather_now"}]`, ` [{"name":"getWeather","arguments":{"base":"height":"5"}},{"name":"getWeatherNow"}]`, both},
		{` []`, `[]`, one},
		{` [{"name":"math_factorial"}]`, ` [{"name":"math_factorial"}]`, nil},
	} {
		if restored := restoreToolNames(test.answer, test.names); restored != test.restored {
			t.Errorf("answer %s\nrestores to %s\nwant        %s", test.answer, restored, test.restored)
		}
	}
}
