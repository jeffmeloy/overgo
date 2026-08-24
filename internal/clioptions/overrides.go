package clioptions

import (
	"flag"
	"time"
)

// IntOverride registers an unset integer override.
func IntOverride(flags *flag.FlagSet, name, usage string) *int {
	value := new(int)
	flags.IntVar(value, name, *value, usage)
	return value
}

// Int64Override registers an unset signed 64-bit override.
func Int64Override(flags *flag.FlagSet, name, usage string) *int64 {
	value := new(int64)
	flags.Int64Var(value, name, *value, usage)
	return value
}

// Uint64Override registers an unset unsigned 64-bit override.
func Uint64Override(flags *flag.FlagSet, name, usage string) *uint64 {
	value := new(uint64)
	flags.Uint64Var(value, name, *value, usage)
	return value
}

// Float64Override registers an unset floating-point override.
func Float64Override(flags *flag.FlagSet, name, usage string) *float64 {
	value := new(float64)
	flags.Float64Var(value, name, *value, usage)
	return value
}

// BoolOverride registers an unset Boolean override.
func BoolOverride(flags *flag.FlagSet, name, usage string) *bool {
	value := new(bool)
	flags.BoolVar(value, name, *value, usage)
	return value
}

// DurationOverride registers an unset duration override.
func DurationOverride(flags *flag.FlagSet, name, usage string) *time.Duration {
	value := new(time.Duration)
	flags.DurationVar(value, name, *value, usage)
	return value
}

// Overrides records explicitly supplied command options.
type Overrides map[string]struct{}

// ExplicitOverrides captures parsed options.
func ExplicitOverrides(flags *flag.FlagSet) Overrides {
	values := make(Overrides)
	flags.Visit(func(option *flag.Flag) { values[option.Name] = struct{}{} })
	return values
}

// ApplyDefault selects authority when the option is absent.
func ApplyDefault[T any](overrides Overrides, name string, destination *T, authority T) {
	if _, explicit := overrides[name]; !explicit {
		*destination = authority
	}
}
