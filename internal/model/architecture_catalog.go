package model

import (
	_ "embed"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/strictjson"
)

//go:embed architecture_profiles.json
var architectureProfileCatalog []byte

func mustLoadArchitectureRegistry() map[string]ArchitectureProfile {
	registry, err := parseArchitectureRegistry(architectureProfileCatalog)
	if err != nil {
		panic(err)
	}
	return registry
}

func parseArchitectureRegistry(content []byte) (map[string]ArchitectureProfile, error) {
	var profiles []ArchitectureProfile
	if err := strictjson.DecodeBytes(content, &profiles); err != nil {
		return nil, fmt.Errorf("model: decode architecture catalog: %w", err)
	}
	if len(profiles) == 0 {
		return nil, errors.New("model: architecture catalog is empty")
	}
	registry := make(map[string]ArchitectureProfile, len(profiles))
	previous := ""
	for index, profile := range profiles {
		if err := ValidateArchitectureProfile(profile); err != nil {
			return nil, fmt.Errorf("model: architecture catalog entry %d: %w", index, err)
		}
		if profile.Name <= previous {
			return nil, fmt.Errorf("model: architecture catalog is not strictly ordered at %q", profile.Name)
		}
		registry[profile.Name] = profile
		previous = profile.Name
	}
	return registry, nil
}

// ValidateArchitectureName checks the persisted profile key syntax.
func ValidateArchitectureName(name string) error {
	if name == "" || strings.TrimSpace(name) != name {
		return errors.New("invalid architecture name")
	}
	for _, character := range name {
		if character != '-' && character != '_' && (character < 'a' || character > 'z') &&
			(character < '0' || character > '9') {
			return errors.New("invalid architecture name")
		}
	}
	return nil
}
