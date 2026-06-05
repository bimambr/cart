{
  description = "Agentic Translation Workflow environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        llamaCppCuda = pkgs.llama-cpp.override {
          cudaSupport = true;
        };

        pythonEnv = pkgs.python3.withPackages (ps:
          with ps; [
            aiohttp
            pandas
            ruff
            (
              scipy-stubs.overrideAttrs (oldAttrs: {
                postPatch = "sed -i -E 's/uv_build[><=0-9.,]*/uv_build/g' pyproject.toml";
              })
            )
            (sentence-transformers.overrideAttrs
              (old: {
                postInstall =
                  (old.postInstall or "")
                  + ''
                    # https://github.com/microsoft/pylance-release/issues/7615
                    rm -f $out/lib/python*/site-packages/transformers/py.typed
                  '';
              }))
          ]);
      in {
        packages = {inherit llamaCppCuda pythonEnv;};

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [pkgs.makeWrapper];

          packages = [
            pythonEnv
            llamaCppCuda
            pkgs.cudatoolkit
            pkgs.basedpyright
            pkgs.just
            pkgs.aria2
            pkgs.tmux
          ];

          shellHook = ''
            echo "Environment loaded with CUDA support."
            echo "Python version: $(python --version)"
            llama-server --version
          '';
        };
      }
    );
}
