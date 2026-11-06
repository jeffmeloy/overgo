package hybridtrain

import (
	"reflect"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/devicemath"
	"overgo/internal/testcheck"
)

func TestResidentMatrixGradientViews(t *testing.T) {
	model := testcheck.Must(BuildModel(smallHybrid(), testSeed))(t)
	plans := testcheck.Must(model.matrixPlans())(t)
	// Simulated disjoint allocations; no device acquisition is needed to test
	// the optimizer-layout-to-view binding.
	for _, base := range []driver.DevicePtr{1, driver.DevicePtr(model.MatrixParamCount()*4 + 1)} {
		for _, plan := range plans {
			want := map[driver.DevicePtr]bool{}
			for _, slot := range plan.slots() {
				want[devicemath.ResidentPtr(base, slot.Off)] = true
			}
			views := plan.residentViews(base)
			if views.IsLinear != plan.IsLinear {
				t.Fatal("matrix mix changed")
			}
			fields := reflect.ValueOf(views)
			for i := range fields.NumField() {
				field := fields.Field(i)
				if field.Kind() == reflect.Bool || field.Uint() == 0 {
					continue
				}
				pointer := driver.DevicePtr(field.Uint())
				if !want[pointer] {
					t.Fatalf("duplicate or unmapped matrix view %s=%d", fields.Type().Field(i).Name, pointer)
				}
				delete(want, pointer)
			}
			if len(want) != 0 {
				t.Fatalf("%d optimizer matrices lack a resident view", len(want))
			}
		}
	}
}
