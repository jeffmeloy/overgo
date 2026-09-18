package dataroot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestResolvedEnvironment(t *testing.T) {
	t.Setenv(Env, "")
	t.Setenv(boundEnvironment, "")
	base := t.TempDir()
	config := filepath.Join(base, ConfigFile)
	if err := os.WriteFile(config, []byte(`{"models":"original-models"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	want, bindings, err := ResolveEnvironment(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, binding := range bindings {
		name, value, ok := strings.Cut(binding, "=")
		if !ok {
			t.Fatalf("invalid environment binding %q", binding)
		}
		t.Setenv(name, value)
	}
	if os.Getenv(Env) != base {
		t.Fatal("data-root directory semantics changed")
	}
	if err := os.WriteFile(config, []byte(`{"models":"later-models"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	want.Source = Env
	got, rebound, err := ResolveEnvironment(t.TempDir())
	if err != nil || got != want || !slices.Equal(rebound, bindings) {
		t.Fatalf("bound child followed later config: %+v, %v", got, err)
	}
	other := t.TempDir()
	t.Setenv(Env, other)
	got, err = Resolve(base)
	if err != nil || got.Models != filepath.Join(other, "models") {
		t.Fatalf("explicit override ignored: %+v %v", got, err)
	}
	t.Setenv(Env, "")
	got, err = Resolve(base)
	if err != nil || got.Models != filepath.Join(base, "later-models") {
		t.Fatalf("new scope retained old binding: %+v %v", got, err)
	}
}

func TestResolvedEnvironmentRejectsInvalidBinding(t *testing.T) {
	base := t.TempDir()
	t.Setenv(Env, base)
	for _, raw := range []string{`{`, `{"unknown":true}`, `{}`, `{} {}`} {
		t.Setenv(boundEnvironment, raw)
		if _, err := Resolve(base); err == nil {
			t.Fatalf("invalid binding accepted: %s", raw)
		}
	}
	for _, path := range []string{"", "relative"} {
		roots := fallback(base)
		roots.Models = path
		raw, err := json.Marshal(environmentBinding{Base: base, Roots: roots})
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv(boundEnvironment, string(raw))
		if _, err := Resolve(base); err == nil {
			t.Fatalf("incomplete binding accepted: %s", raw)
		}
	}
}
