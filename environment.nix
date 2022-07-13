{ }:

let pkgs = import <nixpkgs> { overlays = [  ]; };
in with pkgs;
  let
    APPEND_LIBRARY_PATH = "${lib.makeLibraryPath [  ] }";
    myLibraries = writeText "libraries" ''
      export LD_LIBRARY_PATH="${APPEND_LIBRARY_PATH}:$LD_LIBRARY_PATH"
    '';
  in
    buildEnv {
      name = "env";
      paths = [
        (runCommand "libraries" { } ''
          mkdir -p $out/etc/profile.d
          cp ${myLibraries} $out/etc/profile.d/libraries.sh
        '')
        stack
      ];
    }
