import Lake
open Lake DSL

package «HDIM» where
  version := v!"0.1.0"

lean_lib «Core» where
  srcDir := "."

lean_lib «Extensions» where
  srcDir := "."

@[default_target]
lean_lib «HDIM» where
  srcDir := "."
