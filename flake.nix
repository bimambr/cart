{
  description = "CART environment";

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
      in {
        packages = {inherit llamaCppCuda;};

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [pkgs.makeWrapper];

          packages = [
            pkgs.python3
            pkgs.uv

            pkgs.ruff
            pkgs.basedpyright

            llamaCppCuda

            pkgs.just
            pkgs.aria2
            pkgs.tmux

            (pkgs.rWrapper.override {
              packages = with pkgs.rPackages; [
                brms
                tidybayes
                bayestestR
                emmeans
                performance
                ggplot2
                loo

                styler
                lintr
                languageserver
              ];
            })
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.cudatoolkit
            pkgs.zlib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH

            echo "Environment loaded with CUDA support."
            echo "Python version: $(python --version)"
            llama-server --version
          '';
        };
      }
    );
}
