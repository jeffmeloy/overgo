package agenttool

import "testing"

func validManual() Manual {
	return Manual{
		Name:        "repo.search",
		Description: "Search the repository for a pattern.",
		Effect:      EffectInspection,
		Arguments: []Field{
			{Name: "pattern", Kind: FieldString, Required: true, Description: "regular expression"},
			{Name: "limit", Kind: FieldInteger},
		},
		Transport: Transport{Kind: TransportBuiltin},
	}
}

func TestNewManualCanonicalizesAndIdentifies(t *testing.T) {
	manual, err := NewManual(validManual())
	if err != nil {
		t.Fatal(err)
	}
	if !manual.ID.Valid() || manual.Version != ManualVersion {
		t.Fatalf("manual = %+v, want identified version %d", manual, ManualVersion)
	}
	content, err := manual.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := manualCodec.Parse(content)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != manual.ID || parsed.Name != manual.Name || parsed.Effect != EffectInspection {
		t.Fatalf("parsed = %+v, want the round-tripped manual", parsed)
	}
}

func TestNewManualRefusesInvalidDeclarations(t *testing.T) {
	cases := map[string]func(*Manual){
		"uppercase name":     func(m *Manual) { m.Name = "Repo.Search" },
		"empty description":  func(m *Manual) { m.Description = "" },
		"undeclared effect":  func(m *Manual) { m.Effect = "sideways" },
		"duplicate argument": func(m *Manual) { m.Arguments = append(m.Arguments, m.Arguments[0]) },
		"undeclared kind":    func(m *Manual) { m.Arguments[0].Kind = "tuple" },
		"builtin with url":   func(m *Manual) { m.Transport.URL = "https://example.test/run" },
		"http without host":  func(m *Manual) { m.Transport = Transport{Kind: TransportHTTP, URL: "http://"} },
		"http with program": func(m *Manual) {
			m.Transport = Transport{Kind: TransportHTTP, URL: "https://example.test/run", Program: "sh"}
		},
		"argv without program": func(m *Manual) { m.Transport = Transport{Kind: TransportArgv} },
		"mcp without target": func(m *Manual) {
			m.Transport = Transport{Kind: TransportMCPHTTP, URL: "https://example.test/mcp", Protocol: "2025-06-18"}
		},
		"undeclared transport": func(m *Manual) { m.Transport = Transport{Kind: "carrier-pigeon"} },
	}
	for name, mutate := range cases {
		manual := validManual()
		mutate(&manual)
		if _, err := NewManual(manual); err == nil {
			t.Fatalf("%s: manual accepted", name)
		}
	}
}

func TestManualTransportsDeclareCompletely(t *testing.T) {
	http := validManual()
	http.Name = "weather.lookup"
	http.Transport = Transport{Kind: TransportHTTP, URL: "https://example.test/weather"}
	if _, err := NewManual(http); err != nil {
		t.Fatalf("http manual refused: %v", err)
	}
	stream := validManual()
	stream.Name = "tokens.stream"
	stream.Transport = Transport{Kind: TransportHTTPJSONStream, URL: "https://example.test/stream"}
	if _, err := NewManual(stream); err != nil {
		t.Fatalf("stream manual refused: %v", err)
	}
	argv := validManual()
	argv.Name = "git.status"
	argv.Effect = EffectInspection
	argv.Transport = Transport{Kind: TransportArgv, Program: "git", Args: []string{"status", "--porcelain"}}
	if _, err := NewManual(argv); err != nil {
		t.Fatalf("argv manual refused: %v", err)
	}
	mcp := validManual()
	mcp.Name = "weather.remote"
	mcp.Transport = Transport{
		Kind: TransportMCPHTTP, URL: "https://example.test/mcp",
		Target: "weather.lookup", Protocol: "2025-06-18",
	}
	if _, err := NewManual(mcp); err != nil {
		t.Fatalf("mcp manual refused: %v", err)
	}
}
