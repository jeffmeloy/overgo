package main

import (
	"flag"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/sampling"
)

// TestSamplingOptionsCoverThePolicyAndOverrideOnlyTheirField holds samplingOptions to the sampling policy.
//   - every policy field except Samplers and DryBreakers has one row, because those two parse lists through flags of their own
//   - an explicit flag writes its value into its own field and leaves every other field at the policy's value
func TestSamplingOptionsCoverThePolicyAndOverrideOnlyTheirField(t *testing.T) {
	t.Parallel()
	// overrideValue is positive and every field of the base policy is negative, so a write to the wrong field shows.
	const overrideValue, overrideText = 7, "7"
	policyType := reflect.TypeFor[sampling.Policy]()
	var base sampling.Policy
	fields := reflect.ValueOf(&base).Elem()
	for index := range policyType.NumField() {
		switch field := fields.Field(index); field.Kind() {
		case reflect.Float32:
			field.SetFloat(-float64(index + 1))
		case reflect.Int, reflect.Int64:
			field.SetInt(-int64(index + 1))
		}
	}
	var covered, want []string
	for _, option := range samplingOptions {
		pointer := reflect.ValueOf(option.field(&base)).Pointer()
		for index := range policyType.NumField() {
			if fields.Field(index).Addr().Pointer() == pointer {
				covered = append(covered, policyType.Field(index).Name)
			}
		}
		flags := flag.NewFlagSet("generate", flag.ContinueOnError)
		override := samplingOverrides(flags)
		if err := flags.Parse([]string{"-" + option.name, overrideText}); err != nil {
			t.Fatalf("-%s %s: %v", option.name, overrideText, err)
		}
		got, expected := base, base
		override(clioptions.ExplicitOverrides(flags), &got)
		switch field := option.field(&expected).(type) {
		case *float32:
			*field = overrideValue
		case *int:
			*field = overrideValue
		case *int64:
			*field = overrideValue
		}
		if !reflect.DeepEqual(got, expected) {
			t.Errorf("-%s %s gives policy %+v, want %+v", option.name, overrideText, got, expected)
		}
	}
	for field := range policyType.Fields() {
		if name := field.Name; name != "Samplers" && name != "DryBreakers" {
			want = append(want, name)
		}
	}
	slices.Sort(covered)
	slices.Sort(want)
	if !slices.Equal(covered, want) {
		t.Errorf("samplingOptions cover %v, want each policy field except Samplers and DryBreakers once: %v", covered, want)
	}
}
