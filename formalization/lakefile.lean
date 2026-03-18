import Lake
open Lake DSL

package «HDIM» where
  version := v!"0.1.0"

@[default_target]
lean_lib «HDIM» where
  srcDir := "."
