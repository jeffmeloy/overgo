package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"overgo/internal/discovery"
)

func TestCapabilityCensusWireCompatibility(t *testing.T) {
	data, err := os.ReadFile("testdata/capability_census_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	base := string(bytes.TrimSpace(data))
	for _, test := range []struct {
		name, field string
	}{
		{"original", ""},
		{"explicit-empty", `,"KeyEnvironment":""`},
		{"provider-key-variable", `,"KeyEnvironment":"TEST_PROVIDER_KEY"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire := []byte(strings.Replace(base, `}],"verifications"`, test.field+`}],"verifications"`, 1))
			value, err := capabilityCensusCodec.Parse(wire)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := capabilityCensusCodec.Encode(value)
			if err != nil || !bytes.Equal(wire, encoded) {
				t.Fatalf("stored representation changed: %s: %v", encoded, err)
			}
			identified, err := capabilityCensusCodec.Contract.Identify(wire)
			if err != nil || value.ID != identified {
				t.Fatalf("stored identity changed: %v", err)
			}
			cloned := capabilityCensusCodec.Clone(value)
			cloned.Models[0].Capabilities[0].Stale = "changed"
			if value.Models[0].Capabilities[0].Stale != "" {
				t.Fatal("clone shares mutable capability state")
			}
			if cloned.Models[0].KeyEnvironment != nil {
				*cloned.Models[0].KeyEnvironment = "CHANGED"
				if *value.Models[0].KeyEnvironment == "CHANGED" {
					t.Fatal("clone shares optional field state")
				}
			}
		})
	}
	for _, field := range []string{`,"Unknown":true`, `,"KeyEnvironment":null`} {
		wire := []byte(strings.Replace(base, `}],"verifications"`, field+`}],"verifications"`, 1))
		if _, err := capabilityCensusCodec.Parse(wire); err == nil {
			t.Fatalf("noncanonical or unknown field accepted: %s", field)
		}
	}
	var view struct {
		Models []discovery.CatalogEntry `json:"models"`
	}
	if err := json.Unmarshal([]byte(base), &view); err != nil {
		t.Fatal(err)
	}
	view.Models[0].KeyEnvironment = "TEST_PROVIDER_KEY"
	current, err := json.Marshal(view.Models)
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := json.Marshal(censusModels(view.Models))
	if err != nil || !bytes.Equal(current, frozen) {
		t.Fatalf("current publication changed: %s: %v", frozen, err)
	}
}
