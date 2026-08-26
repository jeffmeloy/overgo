package agenttool

import (
	"strings"
	"testing"
)

const openAPIFixture = `{"openapi":"3.1.0","info":{"title":"Weather","version":"1"},"paths":{"/forecast":{"post":{"summary":"Forecast weather.","operationId":"weather.forecast","responses":{"200":{"description":"ok"}},"x-overgo-effect":"inspection","requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object","properties":{"days":{"type":"integer","description":"forecast days"},"place":{"type":"string"}},"required":["place"]}}}}}}}}`

func TestCompileOpenAPICanonicalCandidateManuals(t *testing.T) {
	first, err := CompileOpenAPI(strings.NewReader(openAPIFixture), "https://api.example.test/v1")
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileOpenAPI(strings.NewReader(openAPIFixture), "https://api.example.test/v1")
	if err != nil {
		t.Fatal(err)
	}
	if first.Source != second.Source || len(first.Manuals) != 1 || first.Manuals[0].ID != second.Manuals[0].ID {
		t.Fatalf("compilations differ: %+v %+v", first, second)
	}
	manual := first.Manuals[0]
	if manual.Name != "weather.forecast" || manual.Effect != EffectInspection ||
		manual.Transport.URL != "https://api.example.test/v1/forecast" || len(manual.Arguments) != 2 ||
		manual.Arguments[0].Name != "days" || manual.Arguments[1].Name != "place" || !manual.Arguments[1].Required {
		t.Fatalf("manual = %+v", manual)
	}
}

func TestCompileOpenAPIFailsClosed(t *testing.T) {
	cases := map[string]string{
		"effect":    strings.Replace(openAPIFixture, `,"x-overgo-effect":"inspection"`, "", 1),
		"method":    strings.Replace(openAPIFixture, `"post":`, `"get":`, 1),
		"reference": strings.Replace(openAPIFixture, `"type":"object"`, `"$ref":"#/components/schemas/Input","type":"object"`, 1),
		"unknown":   strings.Replace(openAPIFixture, `"openapi":"3.1.0"`, `"openapi":"3.1.0","mystery":true`, 1),
	}
	for name, document := range cases {
		if _, err := CompileOpenAPI(strings.NewReader(document), "https://api.example.test"); err == nil {
			t.Fatalf("%s: invalid document compiled", name)
		}
	}
}
