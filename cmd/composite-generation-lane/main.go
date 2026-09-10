//overgo:runtime-inputs caller

// composite-generation-lane validates an active composition against its
// model-native baseline on identical catalog artifacts and held-out inputs.
package main

import "overgo/internal/clioptions"

func main() {
	clioptions.MainNamed("composite-generation-lane", run)
}
